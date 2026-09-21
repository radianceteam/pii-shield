# pii-shield

Reversible pseudonymization for text on its way to a cloud LLM.

It finds identifying data in a prompt — people, organizations, places, phone numbers,
emails, national identifiers — replaces each one with a plausible stand-in **in the same
language**, sends the harmless version to the model, and puts the real values back into
the answer. Credential shapes (API keys, JWTs, connection strings) are blocked outright
rather than replaced.

**24 languages**, including Chinese, Japanese and Korean.

```
Thomas Müller von der Siemens AG anrufen: +49 30 12345678
                    │
                    ▼  anonymize
Sigmund Haering von der Jungfer Stiftung & Co. KG anrufen: +49(0) 996202061
                    │
                    ▼  the model answers, naming the surrogate
Ich habe Sigmund Haering den Entwurf geschickt.
                    │
                    ▼  deanonymize
Ich habe Thomas Müller den Entwurf geschickt.
```

## Ways to use it

| | |
|---|---|
| **OpenAI-compatible proxy** | Point your client's base URL at the shield. Nothing in your code changes, and the provider never sees the real data. Start here — see [Quickstart](#quickstart--proxy-mode) |
| **Python library** | `Shield.anonymize()` / `deanonymize()` in your own process. No network hop, no serialization, nothing on a socket |
| **HTTP sidecar** | `/v1/anonymize` and `/v1/deanonymize` for consumers that cannot import Python |
| **Behind a gateway** | The proxy plus LiteLLM for per-client keys, rate limits and budgets, for anything reachable from the internet — see [Deploying on a public network](#deploying-on-a-public-network) |
| **Inside an agent framework** | Adapters for spec-editor, Hermes and OpenClaw in [`examples/`](examples/) |

It is also being added to **[AgentsPodium.com](https://agentspodium.com)**, so personal agents
running there can use it without deploying anything themselves.

Built on [Presidio](https://github.com/data-privacy-stack/presidio) for named-entity
detection, with a checksum-verified pattern layer for Russian identifiers and a
credential layer that Presidio does not cover.

## Quickstart — proxy mode

Run the shield as an OpenAI-compatible endpoint, point your client's base URL at it, and
nothing else changes. Pick one:

```bash
# Docker — nothing else to install
printf 'PII_SHIELD_TOKEN=%s\nPII_SHIELD_UPSTREAM=https://api.openai.com/v1\nPII_SHIELD_LANGUAGES=ru en\n' \
  "$(openssl rand -base64 24 | tr -d /+=)" > .env && docker compose up -d --build
```

```bash
# pip — one command, installs the package, the Russian pipeline, and starts the proxy
pip install "pii-shield[ner,server,surrogates] @ git+https://github.com/radianceteam/pii-shield" \
  "https://github.com/explosion/spacy-models/releases/download/ru_core_news_lg-3.8.0/ru_core_news_lg-3.8.0-py3-none-any.whl" \
  && pii-shieldd --upstream https://api.openai.com/v1
```

Then point the client at it and check it works:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8099/v1     # most OpenAI SDKs read this

curl -s http://127.0.0.1:8099/healthz
# {"status":"ok","ner_ready":true,"language":"ru"}

curl -s http://127.0.0.1:8099/v1/chat/completions -H 'content-type: application/json' \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"gpt-5.5","messages":[{"role":"user","content":"Thomas Müller anrufen: +49 30 12345678"}]}'
```

Your provider sees something like `Sigmund Haering anrufen: +49(0) 996202061` — a different
cast each session — and the reply you get back names Thomas Müller again.

The Docker form also needs `-H "X-Pii-Shield-Token: <the token from .env>"`, because a
token is set there. `Authorization` is passed through to the provider untouched, so the
shield never holds your API key.

**The first start takes about half a minute** — loading a language pipeline from a cold
Python environment. Subsequent starts take a couple of seconds, and the container's
healthcheck waits this out on its own.

Not on PyPI yet, so the pip form installs from git. Everything below explains what this is
doing and how to change it.

## Using a hosted shield

If someone has already deployed this for you, they will hand you two things: a **base
URL** and an **API key**. Nothing else changes — the endpoint speaks the OpenAI API, so
point your existing client at it and carry on.

```bash
export OPENAI_BASE_URL="https://pii.example.com/v1"
export OPENAI_API_KEY="sk-...."          # the key you were given, not a provider key
```

```python
from openai import OpenAI

client = OpenAI()          # reads both variables above
reply = client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "Thomas Müller anrufen: +49 30 12345678"}],
)
```

```bash
curl -s "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H 'content-type: application/json' \
  -d '{"model":"gpt-5.5","messages":[{"role":"user","content":"Thomas Müller anrufen"}]}'
```

**What actually happens.** Your text is pseudonymized before it leaves the server, so the
model provider never sees the real names, phone numbers or identifiers. The answer comes
back with the real values restored, so from your side the round trip is transparent. You
do not call `/v1/anonymize` yourself and you do not manage sessions — that is the proxy's
job.

Streaming works normally (`stream=True`); a stand-in split across chunks is reassembled
before it is restored.

**What you may see that you would not see from the provider directly:**

| | Meaning |
|---|---|
| `401` | Wrong or missing key |
| `400` with `"type": "pii_shield_blocked"` | Your prompt contained something the policy refuses to send at all — an API key, a passport or card number. Remove it; it was never forwarded |
| `429` | You hit the rate limit or budget on your key |
| `503` | The shield could not analyze the text, so it refused to forward it. Never a silent pass-through |

**What to tell whoever runs it:** which models you need, roughly how many requests per
minute, and which language your text is in — the language decides which detector runs, and
a language nobody configured will fail closed rather than pass your data through.

## Why surrogates instead of `<PERSON>`

A placeholder changes the shape of the text the model reasons over. *"Bitte `<PERSON>`
bis `<DATE>` informieren"* reads as a template, and models answer templates with templates.
A real-looking name keeps the sentence a sentence.

## Install

```bash
pip install pii-shield                              # library: national IDs + credentials
pip install "pii-shield[ner,surrogates]"            # + names/places, realistic stand-ins
pip install "pii-shield[ner,server,surrogates]"     # + the pii-shieldd sidecar and proxy
```

Then a pipeline per language you will process:

```bash
pip install "https://github.com/explosion/spacy-models/releases/download/ru_core_news_lg-3.8.0/ru_core_news_lg-3.8.0-py3-none-any.whl"
pip install "https://github.com/explosion/spacy-models/releases/download/ja_core_news_lg-3.8.0/ja_core_news_lg-3.8.0-py3-none-any.whl"
```

Install the wheel directly rather than running `python -m spacy download`: that command
resolves the target environment itself and has been observed here installing into a
*different* virtualenv than the active one.

`lg` is preferred; `md` and `sm` are accepted as fallbacks, and a size that is not
installed is skipped instantly rather than triggering a download.

### Docker

```bash
cp .env.example .env            # set PII_SHIELD_TOKEN, pick languages
docker compose up -d
curl -s localhost:8099/healthz
```

`.env` needs `PII_SHIELD_TOKEN`; everything else has a default. **The proxy stays off until
`PII_SHIELD_UPSTREAM` is set** — without it you get `/v1/anonymize` and `/v1/deanonymize`
only, and `/v1/chat/completions` returns 404.

Language pipelines are baked into the image at build time (`PII_SHIELD_LANGUAGES="ru en ja"`)
rather than fetched at startup — Presidio reacts to a missing model by trying to install it
at runtime, which blocks for minutes and writes into whatever environment it resolves.

The container runs unprivileged with a read-only filesystem, and compose publishes the port
to `127.0.0.1` only: this service holds raw PII in memory, so exposing it on all interfaces
would make it an anonymization oracle for the whole network.

Measured on a server: **1.1 GB resident with one language loaded, 1.6 GB with two**. Pipelines load on demand, so a request naming a language that is in the image but not yet loaded will raise this further — the compose default of 3 GB covers two comfortably.

### Two tiers

What a policy costs depends on what it asks for, and the entities fall into three groups
with three very different footprints:

| Tier | Entities | Needs |
|---|---|---|
| Own patterns | national identifiers (INN, SNILS, OGRN, BIK, 身份证, マイナンバー, 주민등록번호), banking codes (IBAN, SWIFT/BIC, ABA, LEI), payment cards, credentials, **email and international phone numbers** | nothing — regex and checksums |
| Presidio recognizers | IP, URL, dates, the national IDs Presidio ships for en/es/it/pl, and email and phone when it is installed — it checks numbers against real numbering plans and a regex cannot | `pii-shield[ner]`, but **no language model**: a blank pipeline is enough |
| Language model | PERSON, ORGANIZATION, LOCATION, NRP | a spaCy pipeline, ~1 GB resident |

```bash
pii-shieldd --pattern-only --upstream https://api.openai.com/v1
# or PII_SHIELD_PATTERN_ONLY=1, or pattern_only = true in the config file
```
```python
Policy.pattern_only("ru")     # tier 1 only: ~40 MB resident, starts instantly
Policy.for_language("ru")     # everything, including names

policy.requires_ner()         # True only if PERSON/ORGANIZATION/LOCATION/NRP are active
policy.requires_presidio()    # True if any Presidio recognizer is needed
```

A pattern-only deployment is a real configuration, not a degraded one — it fits in a
container sized for the agent rather than for a language model. What it cannot do is find
names, and it says so: every result carries `names_analyzed`, which is `False` when no
model ran. A caller showing this to end users should repeat that distinction rather than
implying the text was fully examined.

A policy that *does* ask for names and has no pipeline still refuses to construct. That
guarantee is the point of the whole design and is not weakened by the tiers.

**A per-request language stays in the deployment's tier.** Naming a language in a
request selects the preset for *that* language at the tier the daemon was started in —
a pattern-only daemon does not quietly become one that needs a language model. It does
change the catalogue, though: a Russian INN is not in the English one, so asking for
English means it is not looked for. `/healthz` lists what a deployment actually covers:

```json
{"status": "ok", "ner_ready": false, "language": "ru", "pattern_only": true,
 "entities": ["ABA_ROUTING", "CREDIT_CARD", "IBAN_CODE", "RU_INN", "..."]}
```

`ner_ready` answers "are names detected". It does not answer "is a US SSN detected",
which on a pattern-only deployment is also no — that entity belongs to Presidio's layer.

**The tiers refuse the same things.** A tier decides what can be *detected*, not what is
too dangerous to send. Cards and credentials are found by the dependency-free layer, so
the cheapest deployment blocks exactly what the full one blocks.

## Use

```python
from pii_shield import Shield, Policy

shield = Shield(Policy.for_language("ja"))

safe = shield.anonymize("田中太郎さんに090-1234-5678で連絡してください")
answer = call_your_model(safe.text)
real = shield.deanonymize(answer, safe.session_id, consume=True)
```

Pass the same `session_id` across calls in one conversation and a given person keeps the
same stand-in throughout, instead of becoming a new character in every message.

### Languages

```python
from pii_shield import SUPPORTED_LANGUAGES, get_profile

get_profile("ko").model_name()    # 'ko_core_news_lg'
get_profile("zh").faker_locale    # 'zh_CN'
```

ca, da, de, el, en, es, fi, fr, hr, it, ja, ko, lt, mk, nb, nl, pl, pt, ro, ru, sl, sv,
uk, zh — every language with a spaCy core pipeline. An unknown code raises rather than
falling back to English, which would scan with the wrong model and report the text clean.

Four things vary by language, and getting any of them wrong fails quietly:

| | Why it is not uniform |
|---|---|
| Pipeline name | `en` and `zh` use `_core_web_`; the other 22 use `_core_news_` |
| Entity labels | Korean pipelines emit the KLUE tagset (`PS`, `OG`, `LC`), which Presidio's built-in mapping does not contain — without an override Korean detects **nothing** while appearing to work |
| Surrogate locale | a fake name has to be a name in the right language |
| Minimum span | 3 characters is right for space-delimited text and wrong for CJK, where a full personal name is two |

**Direction of pseudonymization.** Detection language and surrogate language are separate
axes. Read Russian, hand the model English names:

```python
policy = Policy.for_language("de")
policy.surrogate_language = "en"      # or surrogate_locale="zh_TW" for an exact variant

# Thomas Müller anrufen  ->  Christopher Weaver anrufen
```

Useful when the downstream model is much stronger in one language than another, or when
whoever reviews the outbound traffic does not read the source language. Detection is
unaffected — the Russian name is still found, only the stand-in changes. An unknown locale
or a language with no Faker locale is rejected rather than quietly degrading to tokens.

**National identifiers.** Presidio ships recognizers for `en` (SSN, ITIN, passport,
driver license, bank, UK NHS), `es` (NIF, NIE), `it` (fiscal code, VAT, ID card,
passport, driver license) and `pl` (PESEL) — and nothing else. This project adds, with
checksum validation:

| Language | Identifiers |
|---|---|
| ru | INN, SNILS, OGRN, internal passport, bank account, phone |
| zh | 居民身份证 (ISO 7064 MOD 11-2), mobile numbers |
| ja | マイナンバー (mod-11 check digit), phone numbers |
| ko | 주민등록번호, mobile numbers |

**Banking codes**, available in every language and pseudonymized by default:

| Entity | Validated by |
|---|---|
| `SWIFT_BIC` | ISO 3166 country code in positions 5–6, plus a context word |
| `IBAN_CODE` | Presidio's IBAN recognizer |
| `ABA_ROUTING` | Federal Reserve prefix range, plus the 3-7-1 weighted mod-10 check |
| `LEI` | ISO 17442 (ISO 7064 MOD 97-10) checksum |
| `CREDIT_CARD` | Luhn plus a real issuer prefix — Presidio recognizes cards in four languages only, so a card used to pass straight through Russian text |

Cards are the one exception to the valid-stand-in rule. They are **blocked** by default at every tier, and when a policy does replace one the result is card-shaped but deliberately fails Luhn: a valid replacement with a real issuer prefix is by construction a number that could belong to somebody, and passing validation is the hazard rather than the feature.
| `IBAN_CODE` | ISO 7064 MOD 97-10 plus the registry's per-country length |

**The stand-ins are themselves valid.** A real BIC becomes another well-formed BIC with the
right country code for the locale, an IBAN becomes an IBAN that passes its check digits, an
an INN becomes an INN whose checksum holds. This is not decoration: a payment instruction
carrying `<IBAN_CODE_1>`, or a BIC replaced by an invented company name, is *corrupted* data
rather than private data — the recipient's own validation rejects it and the failure reads as
a bug instead of as policy.

**Detection is context-gated.** Nine digits are cheap and the ABA checksum passes for roughly
one random number in ten; `SOMEUSER` is a structurally valid BIC because `US` sits in
positions 5–6. So a match without a nearby word like `SWIFT`, `routing` or `Bank` scores 0.4
and the default 0.5 threshold leaves it alone. Field labels (`SWIFT`, `BIC`, `IBAN`, `ABA`,
`INN`, `BIK`) are allowlisted so they are never themselves replaced.

Korea randomized digits 7–13 of the RRN in October 2020, so the classic checksum no
longer holds for numbers issued since. It is used to *raise* confidence (0.95 when it
passes, 0.6 on structure alone) rather than as a precondition, because rejecting every
recent number would be worse than a few false positives.

### Policy

Policy travels with the request, not with the process, so two consumers can share one
shield without the stricter one imposing its rules on the other.

```python
from pii_shield import Policy, EntityRule, Action

policy = Policy(
    language="ja",
    default_action=Action.SURROGATE,        # allow | mask | surrogate | hash | block
    rules=[
        EntityRule(entity="CREDIT_CARD", action=Action.BLOCK),
        EntityRule(entity="DATE_TIME", action=Action.ALLOW),
    ],
    allowlist=["GmbH", "REST", "API"],      # never treated as entities
)
```

Presets: `Policy.for_language(code)` (pseudonymize identity, block credentials and
permanent national IDs), `Policy.strict(code)` (block anything identifying),
`Policy.off(code)` (an auditable opt-out for a genuinely local model). `Policy.ru_default()`
remains as an alias for `for_language("ru")`.

Anything language-dependent — the entity catalogue, the allowlist, the minimum span, the
surrogate locale — defaults to `None` and resolves from the language profile. Set it to
override; leave it to get the right answer for whichever language the request is in.

### Sidecar

For consumers that cannot import Python:

```bash
pii-shieldd --port 8099          # binds 127.0.0.1 by default
```

```
POST /v1/anonymize      {text, session_id?, language?, surrogate_language?, policy?}
                        → {text, session_id, findings[], names_analyzed}
POST /v1/deanonymize    {text, session_id, consume?}  → {text}
DELETE /v1/session/{id}
GET  /healthz
```

`language` selects the language for that request; a language with no pipeline installed
**fails closed** rather than answering 200 with the text unexamined. Unknown fields are
rejected rather than ignored, so a misspelled one is a 422 and not a silent default.

`policy` carries overrides only: anything omitted keeps the language preset's value, so
`{"language": "en"}` means "the usual policy, in English" rather than "a policy with no
rules". Pass `rules` explicitly — including `[]` — to replace them.

A blocked payload returns **422** with the entity *kinds* only. A detection failure returns
**503**. Both mean: do not send the original text. See [`examples/client.mjs`](examples/client.mjs).

Python consumers should import `Shield` directly — an in-process call has no serialization
cost and no window in which the raw payload exists on a socket.

Settings come from defaults, then a TOML file, then flags — later wins:

```bash
pii-shieldd --config pii-shield.toml --port 9000
```

```toml
[pii-shield]
port = 8099
language = "ru"
upstream = "https://api.openai.com/v1"
```

An unknown key in that file is an error, not a silent no-op.

### OpenAI-compatible proxy

The endpoints above require the consumer to call them, and several agent frameworks
cannot. OpenClaw's plugin hooks can *observe* the model input (`llm_input`) or *block*
the run (`before_agent_run` — "only pass and block outcomes are supported"), but none
of them can rewrite a prompt. What those frameworks do support is pointing a provider
at a custom `baseUrl`.

So point it here instead:

```bash
pii-shieldd --port 8099 --upstream https://api.openai.com/v1
```

```
POST /v1/chat/completions       # OpenAI, streaming included
POST /v1/messages               # Anthropic Messages, streaming included
POST /v1/messages/count_tokens  # cleaned, not passed through — see below
GET  /v1/models                 # passthrough
```

Both wire formats are handled, so an unmodified client of either SDK works:

```python
anthropic.Anthropic(base_url="http://127.0.0.1:8099", api_key=...)   # x-api-key
openai.OpenAI(base_url="http://127.0.0.1:8099/v1", api_key=...)      # bearer
```

On the Anthropic route three places carry identifying data and all three are cleaned:
the top-level **`system`** prompt (where an agent's operator details live), **message
content** blocks, and **`tool_result.content`** — whatever the agent's tool returned,
which in practice is the densest personal data in the request. Arguments in
**`tool_use.input`** are cleaned on the way up and restored on the way back, because a
tool called with a stand-in name executes against a person who does not exist. Errors
use Anthropic's envelope on that route and OpenAI's on the other; the SDKs parse
responses against their own schemas and raise on the wrong one.

`/v1/messages/count_tokens` takes a **full prompt**. Proxying it untouched "for
compatibility" would hand the provider exactly the text the shield exists to withhold,
and it would look like the endpoint worked — so it is cleaned by the same path as
`/v1/messages` before the count is taken.

The client sends an ordinary chat completion and gets an ordinary one back; only the
provider sees stand-ins. Streaming works too — a surrogate split across SSE chunks is
reassembled before it is restored, so the output never depends on how the stream was
chunked.

`Authorization` — and `x-api-key`, `anthropic-version`, `anthropic-beta` — is
forwarded upstream untouched, because the caller owns that credential and this proxy
should never need to hold it. The sidecar's own gate is a
separate header, `X-Pii-Shield-Token`.

A blocked request returns **400** in OpenAI's error envelope and is never forwarded. The
sidecar's `/v1/anonymize` uses the same status and the same envelope, so one handler
covers both:

```json
{"error": {"message": "pii-shield blocked this request: SECRET_API_KEY",
           "type": "pii_shield_blocked", "code": "blocked",
           "entities": ["SECRET_API_KEY"]}}
```

Only entity *kinds* cross the wire, never the offending value.

**Serving several languages from one deployment.** The proxy reads the language its
deployment was configured with, which is wrong for a platform whose users write in
different ones. Two optional headers override it per request:

```
X-Pii-Shield-Language: de             # language of the text being read
X-Pii-Shield-Surrogate-Language: en   # language the stand-ins come from
```

The language must have a pipeline installed in that deployment; an unknown one is a
clean `400`, never a silent pass-through. The OpenAI request body has no field for this,
which is why it travels in headers — the body stays a standard chat completion.

Two limits worth knowing before building a multi-tenant service on this:

- **Each loaded language costs about a gigabyte of RAM**, and pipelines load on demand,
  so a deployment that accepts ten languages needs to be sized for ten.
- **A session is not bound to a caller.** Session ids are unguessable, but anything
  holding the shield's token can deanonymize any live session. With one shared token
  that is a single trust domain by design; isolating tenants from each other means one
  shield per tenant, or an authorizing layer in front.

OpenClaw config (its documented `models.providers` shape, JSON5):

```json5
{
  models: {
    providers: {
      shielded: {
        baseUrl: "http://127.0.0.1:8099/v1",
        apiKey: "${OPENAI_API_KEY}",   // forwarded upstream, not stored by the shield
        api: "openai-completions",
        timeoutSeconds: 300,
        models: [{ id: "gpt-5.5", name: "GPT-5.5 (shielded)", input: ["text"] }],
      },
    },
  },
}
```

## Integrations

Working adapters live in [`examples/`](examples/):

| Consumer | Adapter | How it attaches |
|---|---|---|
| spec-editor (Python) | [`spec_editor_provider.py`](examples/spec_editor_provider.py) | Decorates `LLMProvider.complete()`, so every backend is covered and no new agent can bypass it |
| Hermes (Python) | [`hermes_plugin/`](examples/hermes_plugin/) | `llm_request` middleware out, `transform_llm_output` hook back |
| OpenClaw / ClawBot (TypeScript) | none needed | Point a provider's `baseUrl` at the proxy |
| Anything else | [`client.mjs`](examples/client.mjs) | The sidecar's HTTP contract, or the proxy |

A pattern worth knowing before you start: **none of these frameworks lets a plugin
rewrite the outbound model input.** Hermes' `pre_llm_call` only injects context and its
`pre_api_request` is observational, so rewriting there needs the separate middleware
contract; OpenClaw has no rewrite-capable hook at all, which is why it goes through the
proxy. Only spec-editor is different, and only because we own its provider interface.

The Hermes adapter also redacts rather than raises on a blocked credential, because
Hermes' middleware chain fails open on an exception.

## Deploying on a public network

The shield has one shared token and no user management, which is right for a loopback
sidecar and not enough for anything reachable from the internet: no TLS, one static
credential, no per-client keys, no rate limiting, no budgets.

Rather than growing a second-rate copy of all that, put the shield behind something whose
job that is. [`deploy/litellm/`](deploy/litellm/) is a working stack:

```
client → nginx (TLS) → pii-shield → LiteLLM → provider
```

| | Responsibility |
|---|---|
| nginx | TLS, request size cap, and injecting the shield's own header so clients never handle it |
| pii-shield | pseudonymization; no user management of its own |
| LiteLLM | a virtual key per client, RPM/TPM limits, spend budgets, model allow-lists, provider routing |
| postgres | where LiteLLM keeps keys, budgets and spend |

```bash
cd deploy/litellm
cp .env.example .env        # three secrets to fill in
docker compose up -d
```

Then mint a key per client against LiteLLM's admin API:

```bash
curl -X POST http://127.0.0.1:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'content-type: application/json' \
  -d '{"models":["gpt-5.5"],"rpm_limit":60,"max_budget":50,"key_alias":"team-a"}'
```

**Why the shield goes in front of LiteLLM, not behind it.** LiteLLM keeps spend logs in
Postgres. It does not store prompt content by default, and the shipped config pins that
off — but a setting that is off can be turned on, and data that was never there cannot
leak. Putting the shield first means what reaches the database is already pseudonymized.
The shield forwards `Authorization` untouched, so the client's virtual key still reaches
LiteLLM for validation.

LiteLLM also holds the real provider credentials, so a client key is worthless outside
your gateway.

An nginx template is in [`deploy/litellm/nginx.conf.example`](deploy/litellm/nginx.conf.example):
install it as its own file in `sites-available` rather than appending to the default
server block, then `certbot --nginx -d <your-host>`. The DNS record must resolve to the
host first, or the HTTP-01 challenge fails.

## Inflected languages

A stand-in leaves in the nominative and comes back in whatever case the sentence
needed. Exact-match restoring misses that: hand a model "Иванна Олеговна Горбунова"
and it replies "с Иванной Олеговной Горбуновой", which an exact restore leaves alone —
so the caller reads an invented person and takes them for real. That is worse than no
restore at all, because it looks like it worked.

Restoring therefore matches on the stem for free-text names in the languages that
decline them (ru, uk, pl, hr, sl, lt, el, fi, mk). Whole names are matched before their
parts, and the loose pass runs only where the safe one found nothing; a single word maps
to the word in the same position, so a lone surname does not drag a three-part name into
the middle of a sentence. Identifiers are never matched loosely — a checksummed value
comes back verbatim or not at all. Streaming does the same, withholding a window wide
enough for the longest name plus its endings.

Two limits worth knowing. Stem matching cannot tell an inflection of the stand-in from a
different name that shares its stem — "Игнатов" looks like a form of "Игнатьев" to any
rule short of a morphological analyser — so an unrelated name may occasionally be
replaced when the full name is absent from the text. And the stand-in is inserted in the
nominative regardless of the case the sentence wanted, which leaves the text the model
reads slightly ungrammatical; agreeing it with the original would need a morphological
generator and is not done today.

## Memory

A language pipeline is around a gigabyte resident. Loading a second one inside a
container sized for one does not degrade: the kernel kills the process, and an agent
that was protected a moment ago has no shield at all. Being OOM-killed is not failing
closed — it is failing gone.

So the room is measured before a load, against the cgroup limit (or
`PII_SHIELD_MEMORY_LIMIT_BYTES`). The language the deployment was configured for is
**not** evicted by default: dropping it to serve one foreign request means reloading it
for the next, at about forty seconds each way. Other pipelines are dropped
least-recently-used, and when even that would not be enough the load is refused —
before anything is evicted, so a working pipeline is never destroyed for an attempt that
was going to fail regardless.

```python
Shield(policy, max_loaded_languages=2)   # cap regardless of memory
Shield(policy, evict_primary=True)       # trade reload cost for the extra language
```

## Design rules

**Fails closed.** If detection raises, `anonymize()` raises too; it never returns
half-filtered text as if it were clean. And if a policy asks for `PERSON` while no language
model is loadable, `Shield(...)` refuses to construct rather than silently passing every
name through while the caller believes they are being scrubbed.

**The map is not a second copy of your data.** Surrogate mappings live in memory only —
never written to disk, never logged, never rendered in a `repr()`. They are bounded per
session and globally, and expire on a TTL.

**Findings carry offsets, not values.** A `Finding` travels to logs and telemetry, so it
names the entity type and where it was, never what it said.

**Structure beats guesswork.** A checksum-verified identifier outranks an overlapping NER
guess regardless of the score the model assigned it. Likewise, the noise filters
(allowlist, minimum span length) apply only to NER output — they can never suppress a
verified identifier or a credential.

**Overlaps are resolved by consequence, not by score.** Where two NER spans cover the
same text, the one the policy *blocks* wins over the one it *allows*. Without that rule a
junk `DATE_TIME` scoring 0.85 swallows a real `US_SSN` scoring 0.4, which is exactly
backwards for a tool whose job is to stop the second one from leaving.

**Nothing is downloaded unless you ask.** Presidio reacts to a missing spaCy pipeline by
trying to install it — into an environment it picks itself, blocking startup for minutes
with no output. It has been observed installing into a *different* project's virtualenv.
Pipelines are therefore checked for importability before Presidio is asked to load them,
so an absent one is skipped instantly.

Fetching one deliberately is a different thing, and is available:

```bash
pii-shieldd --download-models          # or PII_SHIELD_DOWNLOAD_MODELS=1
```
```python
Shield(policy, download_models=True)
```

When on, a missing pipeline is installed from the publisher's own release host into
**this** interpreter — `sys.executable -m pip`, falling back to `uv pip install --python
sys.executable` because `uv venv` creates environments without pip. Both are told which
environment to use; letting the installer choose is the failure this replaces.

Leave it off in a container: an image should ship what it needs, and a first request that
blocks for minutes while it downloads half a gigabyte is not a request anyone wants to
serve. It earns its place on a workstation, in CI, and on a long-lived server that should
pick up a new language without a rebuild.

**A pipeline that cannot produce a PERSON is refused.** Korean and Swedish models emit
their own tagsets — `PS`, `PRS` — which Presidio's mapping does not contain, so names were
found and then discarded while the text came back looking clean. The loaded pipeline's own
labels are now checked against the mapping, which closes the class rather than chasing one
tagset at a time.

## Accuracy, honestly

Open-source PII detection is not solved. On a 2026 cross-domain benchmark the best average
F1 among five common approaches was **0.542** (Piiranha), with Presidio at **0.481**; on the
broader PIIBench corpus the best of eight systems scored **0.1385**. Structured types with a
checksum or a fixed shape — email, INN, SNILS, API keys — are near-perfect; free-text person
and organization names are not.

Treat this as **risk reduction, not a compliance guarantee**. That is why the default policy
*blocks* passports, cards, national identifiers and credentials rather than replacing them:
for the types where a miss is unacceptable, refusing the request beats hoping the detector
caught it.

Quality also varies by language and by pipeline size. The `sm` pipelines are noticeably
weaker at person names than `lg` — install `lg` for anything that matters. Chinese recall
in particular is the weakest of the five languages exercised by the live test suite
(ru, en, zh, ja, ko); its checksum-verified 身份证 and phone detection are not affected,
because those do not depend on the model at all.

## Development

```bash
uv venv && uv pip install -e ".[dev,ner,server,surrogates]"

# Live multilingual tests need pipelines; each language skips itself without one.
for m in ru_core_news_lg en_core_web_sm zh_core_web_sm ja_core_news_sm ko_core_news_sm; do
  uv pip install "https://github.com/explosion/spacy-models/releases/download/$m-3.8.0/$m-3.8.0-py3-none-any.whl"
done

pytest
ruff check .
```

Use `python -m spacy download` at your own risk here: it resolves the environment
itself and has been observed installing into the wrong virtualenv.

## License

MIT — see [LICENSE](LICENSE). Third-party attributions in [NOTICE](NOTICE): Presidio
(MIT, a runtime dependency) and the credential-pattern set derived from Hermes
(MIT, Nous Research).
