"""Credential-shaped patterns.

Provenance: the pattern set and the naming below are derived from ``agent/redact.py``
of Hermes (hermes-agent), Copyright (c) 2025 Nous Research, MIT licensed. That file
is a secrets redactor for logs and tool output; this is a narrowed port that emits
spans instead of masking in place, because the shield needs offsets to apply policy.
See NOTICE.

Why this layer exists at all, next to Presidio: Presidio detects *identity*, and has
no recognizer for an API key. Credentials reach a model through a different door —
source files and tool output, not prose — and the two layers never overlap.
"""

from __future__ import annotations

import re

# Vendor key prefixes. A prefix match is high-confidence by construction: these
# strings do not occur in prose, so the score is set high and the threshold in
# Policy.ru_default() (0.4) always admits them.
_PREFIX_PATTERNS = (
    r"sk-ant-[A-Za-z0-9_-]{20,}",           # Anthropic
    r"sk-proj-[A-Za-z0-9_-]{20,}",          # OpenAI project
    r"sk-[A-Za-z0-9]{32,}",                 # OpenAI classic / generic
    r"gh[pousr]_[A-Za-z0-9]{36,}",          # GitHub PAT / OAuth / server / refresh
    r"github_pat_[A-Za-z0-9_]{40,}",        # GitHub fine-grained PAT
    r"xox[abprs]-[A-Za-z0-9-]{10,}",        # Slack
    r"AKIA[0-9A-Z]{16}",                    # AWS access key id
    r"ASIA[0-9A-Z]{16}",                    # AWS temporary access key id
    r"AIza[0-9A-Za-z_-]{35}",               # Google API key
    r"ya29\.[0-9A-Za-z_-]{20,}",            # Google OAuth token
    r"hf_[A-Za-z0-9]{30,}",                 # Hugging Face
    r"gsk_[A-Za-z0-9]{40,}",                # Groq
    r"r8_[A-Za-z0-9]{35,}",                 # Replicate
    r"glpat-[A-Za-z0-9_-]{20,}",            # GitLab PAT
    r"dop_v1_[a-f0-9]{64}",                 # DigitalOcean
    r"SG\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}",  # SendGrid
    r"shpat_[a-f0-9]{32}",                  # Shopify
    r"npm_[A-Za-z0-9]{36}",                 # npm
)

# The lookarounds keep a key that is merely a substring of a longer token from
# matching, which otherwise fires on hashes and base64 blobs.
API_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)

# Telegram bot tokens have no vendor prefix; the <digits>:<35+ chars> shape is the tell.
TELEGRAM_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:bot)?\d{8,}:[A-Za-z0-9_-]{30,}(?![A-Za-z0-9_-])"
)

JWT_RE = re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_=-]{4,}){2}")

PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

AUTH_HEADER_RE = re.compile(
    r"(?:Proxy-)?Authorization\s*:\s*(?:[A-Za-z][\w.+-]*\s+)?[^\s\"']{8,}", re.IGNORECASE
)

# Named secret headers carry the value directly after the colon.
_SECRET_HEADER_NAMES = (
    r"(?:X-Api-Key|X-Auth-Token|X-Access-Token|Api-Key|Private-Token|"
    r"X-Goog-Api-Key|Anthropic-Api-Key|OpenAI-Api-Key)"
)
SECRET_HEADER_RE = re.compile(rf"{_SECRET_HEADER_NAMES}\s*:\s*\S{{6,}}", re.IGNORECASE)

# A DSN with an inline password. The password group is what matters, but the whole
# span is reported: a connection string minus its password is still a host inventory.
CONNECTION_STRING_RE = re.compile(
    r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|rediss|amqps?|clickhouse|mssql)"
    r"://[^\s:/@]+:[^\s/@]+@[^\s]+",
    re.IGNORECASE,
)

# Credentials in the userinfo part of any other URL.
URL_CREDENTIAL_RE = re.compile(r"(?:https?|wss?|ftp)://[^\s:/@]+:[^\s/@]+@[^\s]+", re.IGNORECASE)

# ``BEARER`` with an opaque value: no vendor shape to key on, so it needs a length
# floor. Hermes settled on 20 chars because a lower floor turns the English word
# "bearer" in ordinary prose into a false positive on every chat reply.
BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*")

# (entity, compiled pattern, score). Order matters: the scanner keeps the first
# match on any given span, so the most specific pattern must come first.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("SECRET_PRIVATE_KEY", PRIVATE_KEY_RE, 1.0),
    ("SECRET_CONNECTION_STRING", CONNECTION_STRING_RE, 0.95),
    ("SECRET_URL_CREDENTIAL", URL_CREDENTIAL_RE, 0.9),
    ("SECRET_AUTH_HEADER", AUTH_HEADER_RE, 0.9),
    ("SECRET_AUTH_HEADER", SECRET_HEADER_RE, 0.9),
    ("SECRET_JWT", JWT_RE, 0.95),
    ("SECRET_API_KEY", API_KEY_RE, 0.95),
    ("SECRET_API_KEY", TELEGRAM_TOKEN_RE, 0.9),
    ("SECRET_AUTH_HEADER", BEARER_RE, 0.8),
)
