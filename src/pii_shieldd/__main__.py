"""``pii-shieldd`` entry point."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path

# ``.app`` and ``.proxy`` import FastAPI, which lives in the optional ``server``
# extra. Importing them here would turn a missing extra into a traceback before
# ``main`` gets a chance to explain it, so both are imported lazily.
AUTH_ENV = "PII_SHIELD_TOKEN"

# Keys accepted in a config file, each mirroring a command-line flag.
CONFIG_KEYS = frozenset({
    "host", "port", "language", "surrogate_language", "upstream", "allow_remote",
    "download_models", "pattern_only", "redact_credentials", "cache_analysis",
    "parallel", "workers",
})

# Every setting also readable from the environment, which is how a worker process
# learns what to build: uvicorn starts workers by importing a factory, not by
# repeating the command line.
ENV_KEYS = {
    "host": "PII_SHIELD_HOST",
    "port": "PII_SHIELD_PORT",
    "language": "PII_SHIELD_LANGUAGE",
    "surrogate_language": "PII_SHIELD_SURROGATE_LANGUAGE",
    "upstream": "PII_SHIELD_UPSTREAM",
    "pattern_only": "PII_SHIELD_PATTERN_ONLY",
    "redact_credentials": "PII_SHIELD_REDACT_CREDENTIALS",
    "cache_analysis": "PII_SHIELD_CACHE_ANALYSIS",
    "download_models": "PII_SHIELD_DOWNLOAD_MODELS",
    "allow_remote": "PII_SHIELD_ALLOW_REMOTE",
    "parallel": "PII_SHIELD_PARALLEL",
    "workers": "PII_SHIELD_WORKERS",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pii-shieldd", description="pii-shield HTTP sidecar")
    parser.add_argument("--config", default=None, help="TOML config file; flags override it")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--language", default=None, help="Language of the text being read")
    parser.add_argument(
        "--surrogate-language",
        default=None,
        help=(
            "Language to pseudonymize INTO, if different from the one being read. "
            "Detection is unaffected; only the stand-ins change."
        ),
    )
    parser.add_argument(
        "--upstream",
        default=None,
        help=(
            "Enable the OpenAI-compatible proxy and forward to this base URL "
            "(e.g. https://api.openai.com/v1). Also settable via PII_SHIELD_UPSTREAM."
        ),
    )
    parser.add_argument(
        "--pattern-only",
        action="store_true",
        default=None,
        help=(
            "Run without Presidio and without a language model: national identifiers, "
            "banking codes, payment cards and credentials only. Names, organizations "
            "and places are not detected, and every response says so via "
            "names_analyzed. Fits a container sized for the agent rather than for a "
            "language model."
        ),
    )
    parser.add_argument(
        "--redact-credentials",
        action="store_true",
        default=None,
        help=(
            "Replace credentials with a placeholder and forward the request, instead "
            "of refusing it. For a coding agent, which resends its whole history: one "
            "connection string in there otherwise refuses every following request "
            "forever. The placeholder is one-way — the real value never comes back."
        ),
    )
    parser.add_argument(
        "--cache-analysis",
        action="store_true",
        default=None,
        help=(
            "Remember what was found in a text, so the same text is not analyzed "
            "twice. An agent resends its whole conversation every turn, so most of "
            "what arrives has already been read: on a growing session this turns a "
            "cost that climbs with the conversation into one that only covers what is "
            "new. Detection is deterministic, so the answer is the same either way."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help=(
            "Read a large payload in this many blocks at once. The language model's "
            "arithmetic happens inside numpy, which releases the interpreter lock, so "
            "the blocks genuinely overlap. Blocks are split on blank lines, which an "
            "entity never spans. 0 (the default) reads everything in one pass."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Serve from this many processes. One process analyzes one request at a "
            "time — the work is CPU-bound and the interpreter lock is not negotiable — "
            "so several developers behind one deployment wait in line. Each worker "
            "holds its own language pipeline, about a gigabyte. Sessions are per "
            "process, so /v1/anonymize and /v1/deanonymize must reach the same one; "
            "the proxy is unaffected, since its sessions never outlive a request."
        ),
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        default=None,
        help=(
            "Fetch a missing language pipeline instead of refusing to start. Off by "
            "default: it installs code at run time and blocks for minutes. Sensible on "
            "a workstation, wrong for a container, which should ship what it needs."
        ),
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        default=None,
        help="Permit binding a non-loopback address (requires an auth token)",
    )
    return parser


DEFAULTS = {
    "host": "127.0.0.1",
    "port": 8099,
    "language": "ru",
    "surrogate_language": None,
    "allow_remote": False,
    "download_models": False,
    "pattern_only": False,
    "redact_credentials": False,
    "cache_analysis": False,
    "parallel": 0,
    "workers": 1,
}


def load_config(path: str | None) -> dict:
    """Read a TOML config file. Unknown keys are an error, not a silent no-op."""
    if not path:
        return {}
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    section = data.get("pii-shield", data)
    unknown = set(section) - CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"unknown config keys: {', '.join(sorted(unknown))}. "
            f"Accepted: {', '.join(sorted(CONFIG_KEYS))}"
        )
    return section


def from_environment() -> dict:
    """Settings taken from the environment.

    The container has always mapped these onto flags in its entrypoint; reading them
    here as well is what lets a worker rebuild exactly the same app, because uvicorn
    starts workers by importing a factory, which gets no command line.
    """
    out: dict = {}
    for key, name in ENV_KEYS.items():
        value = os.environ.get(name)
        if not value:
            continue
        default = DEFAULTS.get(key)
        if isinstance(default, bool):
            out[key] = value.lower() in ("1", "true", "yes")
        elif isinstance(default, int):
            try:
                out[key] = int(value)
            except ValueError:
                continue
        else:
            out[key] = value
    return out


def resolve(args, config: dict) -> argparse.Namespace:
    """Merge the sources: defaults, then environment, then file, then flags."""
    merged = {**DEFAULTS, **from_environment(), **config}
    for key in CONFIG_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value
    return argparse.Namespace(**merged)


def check_bind(args) -> str | None:
    """Return a refusal message, or None if the address is safe to bind.

    This process holds raw PII in memory by definition. Exposing it off-host without
    authentication turns it into an anonymization oracle for anyone on the network,
    so that combination is refused rather than warned about.
    """
    if args.host in ("127.0.0.1", "::1", "localhost"):
        return None
    if not args.allow_remote:
        return f"refusing to bind {args.host}: pass --allow-remote if intended"
    if not os.environ.get(AUTH_ENV):
        return f"refusing to bind {args.host} without {AUTH_ENV} set"
    return None


def _enabled(args, key: str, env: str) -> bool:
    """An opt-in switch: the flag, or the environment variable, in that order."""
    return bool(
        getattr(args, key, False) or os.environ.get(env, "").lower() in ("1", "true", "yes")
    )


def build_app(args):
    """Assemble the app from resolved settings.

    Split out of ``main`` so a test can assert the proxy is actually mounted. It once
    silently was not, and no unit test could have noticed: the wiring lived inside
    ``main``, which cannot be called without starting a server.
    """
    from pii_shield import Policy, Shield

    from .app import create_app

    pattern_only = _enabled(args, "pattern_only", "PII_SHIELD_PATTERN_ONLY")
    if pattern_only:
        policy = Policy.pattern_only(args.language)
        print(
            "pattern-only mode: no language model; names are not detected "
            "(responses carry names_analyzed=false)",
            file=sys.stderr,
        )
    else:
        policy = Policy.for_language(args.language)

    surrogate = getattr(args, "surrogate_language", None) or os.environ.get(
        "PII_SHIELD_SURROGATE_LANGUAGE"
    )
    if surrogate:
        policy.surrogate_language = surrogate
        print(f"surrogates in: {surrogate}", file=sys.stderr)

    if _enabled(args, "redact_credentials", "PII_SHIELD_REDACT_CREDENTIALS"):
        policy.redact_credentials = True
        print(
            "credential redaction enabled: a credential is replaced by a placeholder "
            "and the request is forwarded, instead of being refused. Cards and "
            "passports are still refused.",
            file=sys.stderr,
        )

    from .proxy import UPSTREAM_ENV, ProxyConfig

    upstream = getattr(args, "upstream", None) or os.environ.get(UPSTREAM_ENV)
    proxy_config = None
    if upstream:
        proxy_config = ProxyConfig.from_env()
        proxy_config.upstream = upstream.rstrip("/")
        print(f"proxy enabled: /v1/chat/completions -> {proxy_config.upstream}", file=sys.stderr)

    download = _enabled(args, "download_models", "PII_SHIELD_DOWNLOAD_MODELS")
    if download:
        print("model download enabled: a missing pipeline will be fetched", file=sys.stderr)

    cache = 2048 if _enabled(args, "cache_analysis", "PII_SHIELD_CACHE_ANALYSIS") else 0
    if cache:
        print(f"analysis cache: up to {cache} texts remembered", file=sys.stderr)

    parallel = int(getattr(args, "parallel", 0) or os.environ.get("PII_SHIELD_PARALLEL", 0) or 0)
    if parallel > 1:
        print(f"parallel analysis: up to {parallel} blocks of a payload at once", file=sys.stderr)

    return create_app(
        Shield(policy, download_models=download, detection_cache=cache, parallel=parallel),
        proxy_config=proxy_config,
    )


def startup_error(exc: Exception, language: str) -> str:
    """Turn a startup failure into something a person can act on.

    A traceback is the wrong answer here: the two realistic failures — the NER extra
    missing, and no pipeline installed for the chosen language — both have a one-line
    fix, and printing a stack trace buries it.
    """
    from pii_shield import get_profile
    from pii_shield.engine.presidio_engine import NerUnavailableError
    from pii_shield.languages import UnsupportedLanguageError

    if isinstance(exc, UnsupportedLanguageError):
        return f"pii-shieldd: {exc}"
    if isinstance(exc, NerUnavailableError):
        model = get_profile(language).model_name()
        return (
            f"pii-shieldd: cannot start for language {language!r} — {exc}\n"
            f"\n"
            f"  Named-entity detection needs two things:\n"
            f"    pip install 'pii-shield[ner]'\n"
            f"    pip install "
            f"'https://github.com/explosion/spacy-models/releases/download/"
            f"{model}-3.8.0/{model}-3.8.0-py3-none-any.whl'\n"
            f"\n"
            f"  Refusing to start rather than run without it: the policy asks for personal\n"
            f"  names, and starting anyway would pass every one of them through while\n"
            f"  appearing to work."
        )
    return f"pii-shieldd: {type(exc).__name__}: {exc}"


def check_server_extra() -> str | None:
    """Return an actionable message if the HTTP stack is missing, else None."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        return (
            f"pii-shieldd: the HTTP server extra is not installed ({exc.name} is missing).\n"
            "\n"
            "  pip install 'pii-shield[ner,server,surrogates]'\n"
            "\n"
            "  'ner' finds names, 'server' runs this sidecar, 'surrogates' makes the\n"
            "  stand-ins realistic. The library itself needs none of them."
        )
    return None


def main(argv: list[str] | None = None) -> int:
    missing = check_server_extra()
    if missing:
        print(missing, file=sys.stderr)
        return 2

    args = build_parser().parse_args(argv)
    try:
        settings = resolve(args, load_config(args.config))
    except (OSError, ValueError) as exc:
        print(f"pii-shieldd: config: {exc}", file=sys.stderr)
        return 2

    refusal = check_bind(settings)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    import uvicorn

    workers = max(1, int(getattr(settings, "workers", 1) or 1))
    if workers > 1:
        # Each worker builds the app for itself, from the environment: uvicorn starts
        # them by importing a factory, so the command line does not reach them.
        export_environment(settings)
        print(
            f"serving from {workers} processes, one language pipeline each. "
            "A session belongs to the process that made it, so /v1/anonymize and "
            "/v1/deanonymize must reach the same worker; the proxy is unaffected.",
            file=sys.stderr,
        )
        uvicorn.run(
            "pii_shieldd.__main__:app_from_environment",
            factory=True,
            host=settings.host,
            port=settings.port,
            workers=workers,
            access_log=False,
        )
        return 0

    try:
        app = build_app(settings)
    except Exception as exc:
        print(startup_error(exc, settings.language), file=sys.stderr)
        return 1

    uvicorn.run(app, host=settings.host, port=settings.port, access_log=False)
    return 0


def export_environment(settings) -> None:
    """Put the resolved settings where a worker process will find them."""
    for key, name in ENV_KEYS.items():
        value = getattr(settings, key, None)
        if value is None or value == "":
            continue
        os.environ[name] = "1" if value is True else ("0" if value is False else str(value))


def app_from_environment():
    """Build the app in a worker. Its only input is the environment."""
    return build_app(resolve(build_parser().parse_args([]), {}))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
