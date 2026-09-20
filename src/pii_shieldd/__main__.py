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
    "download_models",
})


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


def resolve(args, config: dict) -> argparse.Namespace:
    """Merge the three sources, flags winning over file winning over defaults."""
    merged = {**DEFAULTS, **config}
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


def build_app(args):
    """Assemble the app from resolved settings.

    Split out of ``main`` so a test can assert the proxy is actually mounted. It once
    silently was not, and no unit test could have noticed: the wiring lived inside
    ``main``, which cannot be called without starting a server.
    """
    from pii_shield import Policy, Shield

    from .app import create_app

    policy = Policy.for_language(args.language)
    surrogate = getattr(args, "surrogate_language", None) or os.environ.get(
        "PII_SHIELD_SURROGATE_LANGUAGE"
    )
    if surrogate:
        policy.surrogate_language = surrogate
        print(f"surrogates in: {surrogate}", file=sys.stderr)

    from .proxy import UPSTREAM_ENV, ProxyConfig

    upstream = getattr(args, "upstream", None) or os.environ.get(UPSTREAM_ENV)
    proxy_config = None
    if upstream:
        proxy_config = ProxyConfig.from_env()
        proxy_config.upstream = upstream.rstrip("/")
        print(f"proxy enabled: /v1/chat/completions -> {proxy_config.upstream}", file=sys.stderr)

    download = bool(
        getattr(args, "download_models", False)
        or os.environ.get("PII_SHIELD_DOWNLOAD_MODELS", "").lower() in ("1", "true", "yes")
    )
    if download:
        print("model download enabled: a missing pipeline will be fetched", file=sys.stderr)

    return create_app(
        Shield(policy, download_models=download), proxy_config=proxy_config
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

    try:
        app = build_app(settings)
    except Exception as exc:
        print(startup_error(exc, settings.language), file=sys.stderr)
        return 1

    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, access_log=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
