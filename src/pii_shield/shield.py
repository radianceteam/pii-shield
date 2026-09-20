"""The Shield facade: the one entry point consumers touch.

Contract in one line: ``anonymize`` returns text that is safe to send outbound, or
raises. It never returns partially-filtered text as if it were clean.
"""

from __future__ import annotations

import hashlib

from . import secrets as secrets_layer
from .engine import (
    NER_ENTITIES,
    NerUnavailableError,
    PresidioDetector,
    SurrogateFactory,
    finance_patterns,
    merge_findings,
    national_scan,
)
from .policy import Policy, Scope
from .session import SessionStore
from .types import (
    Action,
    AnonymizeResult,
    BlockedError,
    Finding,
    RedactionUnavailableError,
)

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
        try:
            detector.warm(require_person=require_ner)
        except NerUnavailableError:
            if policy.fail_closed:
                raise
            detector = None
        self._detectors[language] = detector
        return detector

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
        ner_spans: list[Finding] = []
        if detector is not None and wanted & self._ner_scope(detector):
            # The analyzer must be asked for the LOWEST threshold any rule uses, not
            # the default: Presidio drops results below the threshold it is given, so
            # passing the default silently discards a per-entity rule that lowered it.
            # The per-entity filter below then applies the real thresholds.
            ner_spans = detector.detect(text, wanted, self._floor_threshold(pol))
            ner_spans.sort(key=lambda f: self._ner_priority(f, pol))

        merged = merge_findings(secret_spans, national_spans, finance_spans, ner_spans)
        return [
            f
            for f in merged
            if f.score >= pol.threshold_for(f.entity)
            and not self._is_noise(f, text, pol)
        ]

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

        replaceable = [f for f in resolved if f.action is not Action.ALLOW]
        if not replaceable:
            return AnonymizeResult(
                text=text, session_id=session_id or "", findings=resolved,
                names_analyzed=self.names_analyzed(pol),
            )

        sid = session_id or self.store.new_session()
        factory = SurrogateFactory(
            seed=sid, locale=pol.effective_surrogate_locale, use_faker=self._use_faker
        )
        factory.reserve(self.store.mapping(sid).keys())

        # Walk backwards so each splice leaves earlier offsets valid.
        out = text
        for finding in sorted(replaceable, key=lambda f: f.start, reverse=True):
            original = text[finding.start : finding.end]
            replacement = self._replacement(finding, original, sid, factory)
            out = out[: finding.start] + replacement + out[finding.end :]

        return AnonymizeResult(
            text=out, session_id=sid, findings=resolved,
            names_analyzed=self.names_analyzed(pol),
        )

    def _replacement(
        self, finding: Finding, original: str, session_id: str, factory: SurrogateFactory
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
        self.store.remember(session_id, original, surrogate)
        return surrogate

    # -- deanonymize --------------------------------------------------------
    def deanonymize(self, text: str, session_id: str, *, consume: bool = False) -> str:
        """Put the real values back into a model's answer.

        Longest surrogate first, so a surrogate that is a prefix of another one
        cannot corrupt the substitution.
        """
        mapping = self.store.pop_session(session_id) if consume else self.store.mapping(session_id)
        if not mapping or not text:
            return text
        for surrogate in sorted(mapping, key=len, reverse=True):
            text = text.replace(surrogate, mapping[surrogate])
        return text
