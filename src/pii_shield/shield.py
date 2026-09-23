"""The Shield facade: the one entry point consumers touch.

Contract in one line: ``anonymize`` returns text that is safe to send outbound, or
raises. It never returns partially-filtered text as if it were clean.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import time
from collections import OrderedDict

from . import secrets as secrets_layer
from .engine import (
    NER_ENTITIES,
    NerUnavailableError,
    PresidioDetector,
    SurrogateFactory,
    contact_patterns,
    finance_patterns,
    merge_findings,
    name_scan,
    national_scan,
)
from .languages import SECRET_ENTITIES, get_profile
from .memory import assess, estimated_load_bytes
from .policy import Policy, Scope
from .restore import restore as restore_inflected
from .session import SessionStore
from .types import (
    Action,
    AnonymizeResult,
    BlockedError,
    Finding,
    RedactionUnavailableError,
    UnknownSessionError,
)

logger = logging.getLogger(__name__)

# Entities whose stand-in is free text and can therefore be declined.
_INFLECTABLE_ENTITIES = frozenset({"PERSON", "ORGANIZATION", "LOCATION", "NRP", "RU_FULL_NAME"})

_MISSING = object()


class Shield:
    """Detects and replaces sensitive spans, and restores them on the way back."""

    def __init__(
        self,
        policy: Policy | None = None,
        *,
        store: SessionStore | None = None,
        detector: PresidioDetector | None = None,
        use_faker: bool = True,
        download_models: bool = False,
        max_loaded_languages: int | None = None,
        evict_primary: bool = False,
        detection_cache: int = 0,
    ) -> None:
        self.policy = policy or Policy.ru_default()
        self.store = store or SessionStore()
        self._use_faker = use_faker
        # Opt-in: fetch a missing language pipeline instead of refusing. Useful on a
        # workstation or a long-lived server that should pick up a new language
        # without a rebuild; wrong for a container, which should ship what it needs.
        self._download_models = download_models
        # One detector per language, built on demand. A request may carry its own
        # policy — the sidecar accepts one per call — and that policy may name a
        # different language than the shield was configured with. Holding a single
        # detector meant an English request was scanned by the Russian pipeline and
        # came back clean, which is the exact failure this project exists to prevent.
        # A pipeline is about a gigabyte resident. Loading a second one inside a
        # container sized for one gets the process killed, which leaves the agent with
        # no shield at all — worse than a refusal. Least-recently-used pipelines are
        # dropped to make room, and when that is not enough the load is refused.
        self._max_languages = max_loaded_languages
        # The language the deployment was configured for is not evicted by default.
        # Dropping it to serve one request in another language means reloading it for
        # the next one, at forty seconds a time — a pod that alternates would spend
        # its life loading pipelines. Refusing the odd foreign request keeps the
        # language the deployment exists for fast. Set this when a mixed-language
        # deployment would rather pay the reload than refuse.
        self._evict_primary = evict_primary
        self._last_used: dict[str, float] = {}
        # How many analyzed texts to remember. Off by default: it is only worth the
        # memory where the same text arrives repeatedly, which is exactly what an
        # agent does and exactly what a one-shot library call does not.
        self._detection_cache_size = int(detection_cache)
        self._cache: OrderedDict[str, tuple[Finding, ...]] = OrderedDict()
        self._detectors: dict[str, PresidioDetector | None] = {}
        if detector is not None:
            self._detectors[self.policy.language] = detector
        self._check_capabilities()

    # -- construction-time safety ------------------------------------------
    def _check_capabilities(self) -> None:
        """Refuse to start misconfigured.

        If the policy asks for PERSON but no NER model is loadable, every request
        would sail through with names intact while the caller believes they are being
        scrubbed. That silent downgrade is worse than a failed startup, so under
        ``fail_closed`` it is fatal here rather than invisible later.
        """
        if not self.policy.requires_presidio():
            # Nothing but this project's own patterns and checksums. No Presidio, no
            # language model, no load at all — this is the configuration that fits in
            # a container sized for the agent rather than for a language model.
            return
        # Warm explicitly rather than through ``detector_for``: that method returns a
        # cached entry untouched, and an injected detector is already cached, so the
        # startup check would pass without ever loading anything.
        existing = self._detectors.get(self.policy.language)
        try:
            if existing is not None:
                existing.warm()
            else:
                self.detector_for(self.policy.language)
        except NerUnavailableError:
            if self.policy.fail_closed:
                raise
            self._detectors[self.policy.language] = None

    @property
    def _detector(self):
        """Detector for the shield's own language. Assignable, which tests rely on."""
        return self._detectors.get(self.policy.language)

    @_detector.setter
    def _detector(self, value) -> None:
        self._detectors[self.policy.language] = value

    def detector_for(self, language: str, pol: Policy | None = None):
        """Return (building if needed) the detector for *language*.

        A policy that does not ask for names is served by a blank pipeline, which
        needs no language model. One that does ask for them requires a real pipeline,
        and its absence raises under ``fail_closed`` rather than returning None and
        quietly scanning nothing.
        """
        policy = pol or self.policy
        require_ner = policy.requires_ner()

        cached = self._detectors.get(language, _MISSING)
        if cached is not _MISSING:
            self._last_used[language] = time.monotonic()
            # A blank detector cached for a cheaper policy must not silently serve a
            # request that needs names; rebuild instead of under-detecting.
            if cached is None or not require_ner or getattr(cached, "ner_available", True):
                if cached is None and require_ner and policy.fail_closed:
                    raise NerUnavailableError(
                        f"no spaCy pipeline loadable for language {language!r}"
                    )
                return cached

        detector = PresidioDetector(
            language, allow_blank=not require_ner, download=self._download_models
        )
        if require_ner:
            self._make_room_for(detector, language)
        self._last_used[language] = time.monotonic()
        try:
            detector.warm(require_person=require_ner)
        except NerUnavailableError:
            if policy.fail_closed:
                raise
            detector = None
        self._detectors[language] = detector
        return detector

    def _loaded_languages(self) -> list[str]:
        return [lang for lang, det in self._detectors.items() if det is not None]

    def _evict_least_recently_used(self, keep: str) -> str | None:
        """Drop the least recently used pipeline. Returns the language dropped."""
        order = self._eviction_order(keep=keep)
        if not order:
            return None
        victim = order[0][0]
        self._evict(victim)
        return victim

    def _evictable_bytes(self, keep: str) -> int:
        """How much dropping every other loaded pipeline could plausibly free."""
        total = 0
        for lang in self._loaded_languages():
            if lang == keep:
                continue
            loaded = getattr(self._detectors[lang], "loaded_model", None)
            total += estimated_load_bytes(loaded) or 0 if loaded else 0
        return total

    def _make_room_for(self, detector: PresidioDetector, language: str) -> None:
        """Evict until the pipeline fits, or refuse — but never do both.

        Evicting first and refusing afterwards is the worst outcome available: a
        working pipeline is destroyed to attempt a load that was never going to fit,
        and the shield ends up with nothing loaded. So the decision is made up front,
        on estimates, and eviction happens only once it is known to be enough.
        """
        model = detector.planned_model()
        if model is None:
            return

        cap = self._max_languages
        while cap is not None and len(self._loaded_languages()) >= cap:
            if self._evict_least_recently_used(keep=language) is None:
                break

        free, needed, reason = assess(model)
        if free is None or needed is None or free >= needed:
            return

        victims = self._eviction_order(keep=language)
        recoverable = sum(size for _, size in victims)
        if free + recoverable < needed:
            raise NerUnavailableError(
                f"not enough memory to load {model} alongside {self.policy.language}: "
                f"{reason}. Give the container more memory, use --pattern-only, which "
                "needs no pipeline at all, or set evict_primary to trade the reload "
                f"cost for the extra language."
            )

        for victim, size in victims:
            self._evict(victim)
            free += size
            if free >= needed:
                return

    def _eviction_order(self, keep: str) -> list[tuple[str, int]]:
        """Loaded pipelines other than *keep*, least recently used first, with sizes."""
        out = []
        for lang in sorted(self._loaded_languages(), key=lambda x: self._last_used.get(x, 0.0)):
            if lang == keep:
                continue
            if lang == self.policy.language and not self._evict_primary:
                continue
            loaded = getattr(self._detectors[lang], "loaded_model", None)
            out.append((lang, estimated_load_bytes(loaded) or 0 if loaded else 0))
        return out

    def _evict(self, language: str) -> None:
        self._detectors.pop(language, None)
        self._last_used.pop(language, None)
        gc.collect()
        logger.warning("pii-shield: unloaded the %s pipeline to make room", language)

    @property
    def ner_ready(self) -> bool:
        return self._detector is not None

    @staticmethod
    def _ner_scope(detector) -> frozenset[str]:
        """Entities the given detector can actually answer for.

        Taken from the detector rather than a module constant, because the set is
        language-dependent: asking an ``en`` analyzer for ``US_SSN`` works, asking a
        ``ru`` one for it does not.
        """
        supported = getattr(detector, "supported_entities", None)
        return frozenset(supported) if supported else NER_ENTITIES

    # -- detection ----------------------------------------------------------
    def detect(self, text: str, policy: Policy | None = None) -> list[Finding]:
        """Run every enabled layer and return merged spans, most-authoritative first."""
        pol = policy or self.policy
        wanted = set(pol.entities)

        secret_spans = secrets_layer.scan(text, entities=wanted) if pol.secrets else []
        national_spans = national_scan(pol.language, text, wanted)
        finance_spans = finance_patterns.scan(text, entities=wanted)
        detector = self.detector_for(pol.language, pol) if pol.requires_presidio() else None

        # An agent resends its whole conversation on every turn, so the language model
        # spends most of its time re-reading text it has already read. Detection is
        # deterministic — the same text under the same policy and the same pipeline
        # yields the same spans — so the answer can simply be remembered. The key is a
        # hash and the value is offsets and entity kinds, so nothing here holds the
        # text it was computed from.
        key = self._cache_key(text, pol, detector)
        if key is not None:
            remembered = self._cache.get(key)
            if remembered is not None:
                self._cache.move_to_end(key)
                return list(remembered)
        ner_spans: list[Finding] = []
        if detector is not None and wanted & self._ner_scope(detector):
            # The analyzer must be asked for the LOWEST threshold any rule uses, not
            # the default: Presidio drops results below the threshold it is given, so
            # passing the default silently discards a per-entity rule that lowered it.
            # The per-entity filter below then applies the real thresholds.
            ner_spans = detector.detect(text, wanted, self._floor_threshold(pol))
            ner_spans.sort(key=lambda f: self._ner_priority(f, pol))

        # Names found by pattern share the NER layer rather than outranking it: where
        # both fire on the same person, the longer span must win, and that is what
        # _ner_priority already decides. Outranking would let a two-word pattern match
        # shorten a three-word span the model got right, which is a leak, not a fix.
        name_spans = name_scan(pol.language, text, wanted)
        if name_spans:
            ner_spans = sorted(ner_spans + name_spans, key=lambda f: self._ner_priority(f, pol))

        # Contact details only when Presidio is not here to do it better: it checks
        # numbers against real numbering plans, which this layer cannot.
        contact_spans = (
            contact_patterns.scan(text, entities=wanted) if detector is None else []
        )
        merged = merge_findings(
            secret_spans, national_spans, finance_spans, contact_spans, ner_spans
        )
        findings = [
            f
            for f in merged
            if f.score >= pol.threshold_for(f.entity)
            and not self._is_noise(f, text, pol)
        ]
        if key is not None:
            self._remember_detection(key, findings)
        return findings

    def _cache_key(self, text: str, pol: Policy, detector) -> str | None:
        """What makes two analyses interchangeable, or None when caching is off.

        The policy, the pipeline actually loaded and whether it can find names all
        change the answer, so all three are in the key. Getting that wrong would serve
        a blank pipeline's result — names not looked for — to a request that asked for
        them, which is the one failure this project exists to prevent.
        """
        if self._detection_cache_size <= 0 or not text:
            return None
        fingerprint = "\x00".join((
            pol.model_dump_json(),
            str(getattr(detector, "loaded_model", None)),
            str(getattr(detector, "ner_available", False)),
        ))
        digest = hashlib.blake2b(digest_size=16)
        digest.update(fingerprint.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    # A text carrying more findings than this is not a conversation turn, it is a data
    # dump; remembering it would spend the whole cache on one entry.
    _CACHE_FINDING_LIMIT = 2000

    def _remember_detection(self, key: str, findings: list[Finding]) -> None:
        if len(findings) > self._CACHE_FINDING_LIMIT:
            return
        self._cache[key] = tuple(findings)
        self._cache.move_to_end(key)
        while len(self._cache) > self._detection_cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _ner_priority(finding: Finding, pol: Policy) -> tuple:
        """Ordering for overlapping NER spans, most important first.

        Presidio returns overlapping spans and the merge keeps whichever comes first,
        so this ordering is not cosmetic. Without it a junk ``DATE_TIME`` scoring 0.85
        swallows a real ``US_SSN`` scoring 0.4 — an entity the policy *allows* beating
        one it *blocks*, which is exactly backwards for a tool whose job is to stop the
        second one from leaving.

        Order: what the policy blocks, then what it does not allow, then the longer
        span (more specific), then the higher score.
        """
        action = pol.action_for(finding.entity)
        return (
            action is not Action.BLOCK,
            action is Action.ALLOW,
            -(finding.end - finding.start),
            -finding.score,
        )

    @staticmethod
    def _floor_threshold(pol: Policy) -> float:
        """Lowest threshold in play, so no rule is pre-filtered away by the analyzer."""
        thresholds = [pol.default_threshold]
        wanted = set(pol.entities or [])
        thresholds += [r.threshold for r in pol.rules if r.entity in wanted]
        return min(thresholds)

    @staticmethod
    def _is_noise(finding: Finding, text: str, pol: Policy) -> bool:
        """Drop spans that are almost certainly not PII.

        Both guards apply to NER output only. A checksum-verified INN or a vendor-prefixed
        API key is structurally certain, so no heuristic gets to overrule it — and a short
        span is not evidence against either one.
        """
        if finding.entity not in NER_ENTITIES:
            return False
        span = text[finding.start : finding.end]
        if len(span.strip()) < pol.effective_min_ner_span:
            return True
        return pol.is_allowlisted(span)

    # -- anonymize ----------------------------------------------------------
    def names_analyzed(self, policy: Policy | None = None) -> bool:
        """Whether a model capable of finding names is available for this policy."""
        pol = policy or self.policy
        if not pol.requires_ner():
            return False
        detector = self._detectors.get(pol.language)
        return bool(detector is not None and getattr(detector, "ner_available", True))

    def anonymize(
        self,
        text: str,
        *,
        session_id: str | None = None,
        policy: Policy | None = None,
    ) -> AnonymizeResult:
        """Return outbound-safe text plus the handle needed to reverse it.

        Raises :class:`BlockedError` if a BLOCK-policy entity is present, and
        :class:`RedactionUnavailableError` if detection itself failed under
        ``fail_closed``.
        """
        pol = policy or self.policy
        # A session id the store never minted used to be accepted, substituted into,
        # and then not written anywhere: the caller got a 200, a session id back, and
        # no way to restore any of it. Continuing a session is the only reason to pass
        # one, so an id this process does not know is an error.
        if session_id and not self.store.knows(session_id):
            raise UnknownSessionError(
                f"unknown session {session_id!r}: it was never issued here, or it has "
                "expired. Omit it to start a new one."
            )
        if pol.scope is Scope.OFF or not text:
            return AnonymizeResult(
                text=text, session_id=session_id or "", findings=[], names_analyzed=False
            )

        try:
            findings = self.detect(text, pol)
        except Exception as exc:
            if pol.fail_closed:
                raise RedactionUnavailableError(
                    f"detection failed, refusing to emit unfiltered text: {exc}"
                ) from exc
            return AnonymizeResult(
                text=text, session_id=session_id or "", findings=[],
                names_analyzed=self.names_analyzed(pol),
            )

        # Resolve each span to the action the policy actually mandates.
        resolved = [f.model_copy(update={"action": pol.action_for(f.entity)}) for f in findings]

        blocked = [f for f in resolved if f.action is Action.BLOCK]
        if blocked:
            raise BlockedError(blocked)

        redacted = any(
            f.action is Action.MASK and f.entity in SECRET_ENTITIES for f in resolved
        )

        replaceable = [f for f in resolved if f.action is not Action.ALLOW]
        if not replaceable:
            return AnonymizeResult(
                text=text, session_id=session_id or "", findings=resolved,
                names_analyzed=self.names_analyzed(pol),
                credentials_redacted=redacted,
            )

        sid = session_id or self.store.new_session()
        factory = SurrogateFactory(
            seed=sid, locale=pol.effective_surrogate_locale, use_faker=self._use_faker
        )
        factory.reserve(self.store.mapping(sid).keys())

        # Built in one forward pass rather than spliced per finding. Splicing copies
        # the whole payload every time, which is invisible on a paragraph and ruinous
        # on an agent's context: 8 MB carrying 80 000 findings took 351 seconds that
        # way, against about two here. Spans never overlap — merge_findings has
        # already resolved that — so a single pass is enough.
        pieces: list[str] = []
        cursor = 0
        for finding in sorted(replaceable, key=lambda f: f.start):
            original = text[finding.start : finding.end]
            pieces.append(text[cursor : finding.start])
            pieces.append(self._replacement(finding, original, sid, factory, pol))
            cursor = finding.end
        pieces.append(text[cursor:])

        return AnonymizeResult(
            text="".join(pieces), session_id=sid, findings=resolved,
            names_analyzed=self.names_analyzed(pol),
            credentials_redacted=redacted,
        )

    def _replacement(
        self,
        finding: Finding,
        original: str,
        session_id: str,
        factory: SurrogateFactory,
        pol: Policy,
    ) -> str:
        if finding.action is Action.MASK:
            return f"<{finding.entity}>"
        if finding.action is Action.HASH:
            digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
            return f"<{finding.entity.lower()}_{digest}>"
        # SURROGATE: reuse this session's existing choice so one real person stays
        # one fake person across every call in the conversation.
        existing = self.store.surrogate_for(session_id, original)
        if existing is not None:
            return existing
        surrogate = factory.make(finding.entity, original)
        self.store.remember(
            session_id, original, surrogate, inflectable=self._inflectable(finding.entity, pol)
        )
        return surrogate

    @staticmethod
    def _inflectable(entity: str, pol: Policy) -> bool:
        """Whether this stand-in can come back in another grammatical form.

        The question is about the *stand-in*, not the text it was taken from: the loose
        restore matches what was written out, so a deployment putting English stand-ins
        into Russian text has nothing to match loosely. Reading it off the request's
        policy rather than the server's matters for the same reason the tier does — a
        request naming its own language must not be answered with the daemon's.

        An identifier is never included: it comes back verbatim or not at all.
        """
        if entity not in _INFLECTABLE_ENTITIES:
            return False
        return get_profile(pol.surrogate_language or pol.language).inflects_names

    # -- deanonymize --------------------------------------------------------
    def deanonymize(self, text: str, session_id: str, *, consume: bool = False) -> str:
        """Put the real values back into a model's answer.

        Longest surrogate first, so a surrogate that is a prefix of another one
        cannot corrupt the substitution.
        """
        if not text:
            return text
        # An expired or evicted session used to come back as the answer with the
        # stand-ins still in it, which reads like "the model named nobody" rather than
        # "the real values are gone". A session that exists and holds nothing is a
        # different thing and still answers normally.
        if not self.store.knows(session_id):
            raise UnknownSessionError(
                f"unknown session {session_id!r}: it was never issued here, or it has "
                "expired. The original values cannot be restored."
            )
        pairs = self.store.inflectable_pairs(session_id)
        mapping = self.store.pop_session(session_id) if consume else self.store.mapping(session_id)
        if not mapping:
            return text
        # Both passes happen in one place, because the second must know what the first
        # wrote. Substituting the exact forms here and asking for the inflected ones
        # afterwards let a stem pattern match a real name that had just been restored
        # and replace it with another word of the same person's name.
        return restore_inflected(text, pairs, exact=mapping)
