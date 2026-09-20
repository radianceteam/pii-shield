"""CLI wiring: settings resolution, route mounting, bind safety, startup failures.

These exist because of a real miss: the proxy silently failed to mount for a while and
every unit test still passed, because the wiring lived inside ``main()`` where no test
could reach it. Only a live curl caught it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pii_shield.engine.presidio_engine import NerUnavailableError
from pii_shield.languages import UnsupportedLanguageError
from pii_shieldd.__main__ import (
    build_app,
    build_parser,
    check_bind,
    load_config,
    main,
    resolve,
    startup_error,
)
from pii_shieldd.app import AUTH_ENV


def settings(argv=None, config=None):
    return resolve(build_parser().parse_args(argv or []), config or {})


def _paths(app) -> set[str]:
    """Every mounted path.

    Read from the OpenAPI schema rather than ``app.routes``: an included router shows
    up there as an opaque ``_IncludedRouter`` with no ``path``, so walking the route
    list would report the proxy as missing even when it is mounted.
    """
    return set(app.openapi()["paths"])


# --- route mounting ---------------------------------------------------------
def test_proxy_is_not_mounted_without_upstream(monkeypatch):
    monkeypatch.delenv("PII_SHIELD_UPSTREAM", raising=False)
    app = build_app(settings())
    assert "/healthz" in _paths(app)
    assert "/v1/chat/completions" not in _paths(app)


def test_upstream_flag_mounts_the_proxy(monkeypatch):
    monkeypatch.delenv("PII_SHIELD_UPSTREAM", raising=False)
    app = build_app(settings(["--upstream", "https://api.openai.com/v1"]))
    assert "/v1/chat/completions" in _paths(app)


def test_upstream_env_var_mounts_the_proxy(monkeypatch):
    monkeypatch.setenv("PII_SHIELD_UPSTREAM", "https://api.openai.com/v1")
    assert "/v1/chat/completions" in _paths(build_app(settings()))


def test_trailing_slash_on_upstream_is_normalized(monkeypatch):
    """Otherwise every forwarded URL grows a double slash."""
    monkeypatch.delenv("PII_SHIELD_UPSTREAM", raising=False)
    from pii_shieldd import app as app_module

    captured = {}
    real_create_app = app_module.create_app

    def spy(shield, proxy_config=None):
        captured["upstream"] = proxy_config.upstream if proxy_config else None
        return real_create_app(shield, proxy_config=proxy_config)

    monkeypatch.setattr(app_module, "create_app", spy)
    build_app(settings(["--upstream", "https://api.openai.com/v1/"]))
    assert captured["upstream"] == "https://api.openai.com/v1"


# --- settings resolution ----------------------------------------------------
def test_defaults():
    resolved = settings()
    assert (resolved.host, resolved.port, resolved.language) == ("127.0.0.1", 8099, "ru")


def test_config_file_is_read(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[pii-shield]\nport = 9001\nlanguage = "ja"\n', encoding="utf-8")
    resolved = settings(["--config", str(path)], load_config(str(path)))
    assert resolved.port == 9001
    assert resolved.language == "ja"


def test_config_without_a_section_header(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('port = 9002\n', encoding="utf-8")
    assert load_config(str(path)) == {"port": 9002}


def test_flags_beat_the_config_file(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[pii-shield]\nport = 9001\nlanguage = "ja"\n', encoding="utf-8")
    resolved = settings(["--config", str(path), "--port", "9999"], load_config(str(path)))
    assert resolved.port == 9999
    assert resolved.language == "ja"      # untouched keys survive


def test_unknown_config_key_is_an_error_not_a_no_op(tmp_path):
    """A typo in a config file must not silently do nothing."""
    path = tmp_path / "c.toml"
    path.write_text('[pii-shield]\nprot = 9001\n', encoding="utf-8")
    with pytest.raises(ValueError, match="prot"):
        load_config(str(path))


def test_missing_config_file_exits_cleanly(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.toml")]) == 2
    assert "config" in capsys.readouterr().err


# --- bind safety ------------------------------------------------------------
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_is_always_allowed(host):
    assert check_bind(settings(["--host", host])) is None


def test_remote_bind_needs_the_flag():
    refusal = check_bind(settings(["--host", "0.0.0.0"]))
    assert refusal and "--allow-remote" in refusal


def test_remote_bind_needs_a_token(monkeypatch):
    monkeypatch.delenv(AUTH_ENV, raising=False)
    refusal = check_bind(settings(["--host", "0.0.0.0", "--allow-remote"]))
    assert refusal and AUTH_ENV in refusal


def test_remote_bind_allowed_with_flag_and_token(monkeypatch):
    monkeypatch.setenv(AUTH_ENV, "s3cret")
    assert check_bind(settings(["--host", "0.0.0.0", "--allow-remote"])) is None


def test_main_refuses_a_bad_bind(capsys):
    assert main(["--host", "0.0.0.0"]) == 2
    assert "refusing to bind" in capsys.readouterr().err


# --- startup errors ---------------------------------------------------------
def test_missing_ner_message_is_actionable():
    """A traceback buries a one-line fix; this is the message that replaces it."""
    message = startup_error(NerUnavailableError("presidio-analyzer is not installed"), "ja")
    assert "pip install 'pii-shield[ner]'" in message
    assert "ja_core_news_lg" in message
    assert "Refusing to start" in message


def test_unsupported_language_message_lists_the_options():
    message = startup_error(UnsupportedLanguageError("unsupported language 'xx'"), "ru")
    assert "unsupported language" in message
    assert "Traceback" not in message


def test_main_reports_a_startup_failure_without_a_traceback(monkeypatch, capsys):
    import pii_shieldd.__main__ as cli

    monkeypatch.setattr(cli, "build_app", lambda _a: (_ for _ in ()).throw(
        NerUnavailableError("presidio-analyzer is not installed")
    ))
    assert main([]) == 1
    err = capsys.readouterr().err
    assert "pip install" in err
    assert "Traceback" not in err


# --- pattern-only, the tier the daemon could not run ------------------------
def test_pattern_only_flag_builds_a_cheap_policy(monkeypatch):
    """The daemon always built the full policy, so a container without a language
    model could not start at all — the library supported the tier and the daemon
    did not."""
    monkeypatch.delenv("PII_SHIELD_UPSTREAM", raising=False)
    app = build_app(settings(["--pattern-only"]))
    from pii_shieldd.app import get_shield

    with TestClient(app):
        policy = get_shield().policy
        assert not policy.requires_presidio()
        assert not policy.requires_ner()


def test_pattern_only_via_environment(monkeypatch):
    monkeypatch.delenv("PII_SHIELD_UPSTREAM", raising=False)
    monkeypatch.setenv("PII_SHIELD_PATTERN_ONLY", "1")
    app = build_app(settings())
    from pii_shieldd.app import get_shield

    with TestClient(app):
        assert not get_shield().policy.requires_presidio()


def test_pattern_only_via_config_file(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[pii-shield]\npattern_only = true\n", encoding="utf-8")
    assert settings(["--config", str(path)], load_config(str(path))).pattern_only is True


def test_healthz_admits_there_is_no_ner(monkeypatch):
    monkeypatch.delenv("PII_SHIELD_UPSTREAM", raising=False)
    with TestClient(build_app(settings(["--pattern-only"]))) as client:
        assert client.get("/healthz").json()["ner_ready"] is False
