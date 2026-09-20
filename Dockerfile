# pii-shield sidecar.
#
# Language pipelines are baked into the image rather than downloaded at startup.
# That is deliberate: Presidio reacts to a missing spaCy model by trying to install
# it at runtime, which blocks for minutes and writes into whatever environment it
# happens to resolve. A container that starts must already have what it needs.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

RUN pip install --no-cache-dir ".[ner,server,surrogates]"

# Space-separated language codes, e.g. "ru en ja". Each adds a few hundred MB.
ARG PII_SHIELD_LANGUAGES="ru"
# lg is accurate, sm is small. Person-name recall differs noticeably between them.
ARG SPACY_MODEL_SIZE="lg"
ARG SPACY_MODELS_VERSION="3.8.0"

# The pipeline name is resolved by the package itself, so this cannot drift from the
# language registry (en and zh use _core_web_, the other 22 use _core_news_).
RUN set -eux; \
    for lang in ${PII_SHIELD_LANGUAGES}; do \
        model="$(python -c "from pii_shield import get_profile; \
print(get_profile('${lang}').model_name('${SPACY_MODEL_SIZE}'))")"; \
        pip install --no-cache-dir \
            "https://github.com/explosion/spacy-models/releases/download/${model}-${SPACY_MODELS_VERSION}/${model}-${SPACY_MODELS_VERSION}-py3-none-any.whl"; \
    done

# Runs unprivileged: this process holds raw PII in memory, and nothing it does needs root.
RUN useradd --create-home --uid 10001 shield
USER shield

ENV PII_SHIELD_HOST=0.0.0.0 \
    PII_SHIELD_PORT=8099 \
    PII_SHIELD_LANGUAGE=ru

EXPOSE 8099

# No curl in slim, and adding it for a health check is not worth the surface.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8099/healthz', timeout=4).status==200 else 1)"

# 0.0.0.0 is required for the port mapping to work, so --allow-remote is passed here.
# The safety property is preserved elsewhere: the sidecar still refuses a non-loopback
# bind unless PII_SHIELD_TOKEN is set, and the compose file publishes to 127.0.0.1 only.
ENTRYPOINT ["sh", "-c", "exec pii-shieldd --host \"$PII_SHIELD_HOST\" --port \"$PII_SHIELD_PORT\" --language \"$PII_SHIELD_LANGUAGE\" --allow-remote \"$@\"", "--"]
