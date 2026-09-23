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

Not on PyPI yet, so the pip form installs from git, and it needs **Python 3.11 or newer** —
macOS ships 3.9 as `python3`, where pip backtracks for minutes instead of saying it cannot
resolve the package; [uv](https://docs.astral.sh/uv/) fetches a suitable interpreter itself
(see [Running it on your own machine](#running-it-on-your-own-machine)). Everything below
explains what this is doing and how to change it.

## Running it on your own machine

The shield runs happily as a local proxy on a developer's machine: point your tools at it
instead of at the provider, and nothing else changes. Your provider key passes through
untouched; the shield never stores it.

### Set it up

Python 3.11 or newer, which `uv` will fetch for you:

```bash
uv venv --python 3.12 ~/.pii-shield
VIRTUAL_ENV=~/.pii-shield uv pip install \
  "pii-shield[server,surrogates] @ git+https://github.com/radianceteam/pii-shield"
```

That is the pattern-only tier: **30 MB on disk, about 65 MB resident**, starting in a
second. It replaces tax and company identifiers, phone numbers, emails, IBAN, BIC and
account numbers, and refuses passports, insurance numbers, payment cards and credentials.
It does not detect names — see [the full tier](#the-full-tier) below.

### Start it

```bash
~/.pii-shield/bin/pii-shieldd --pattern-only --port 8199 \
  --upstream https://api.deepseek.com/v1
```

`--upstream` is your provider's base URL, and **one process serves one provider**: the
OpenAI route posts to `<upstream>/chat/completions` and the Anthropic route to
`<upstream>/messages`, so a daemon pointed at an OpenAI-shaped API serves OpenAI clients,
one pointed at `https://api.anthropic.com/v1` serves Anthropic clients, and needing both
means running two. The daemon binds `127.0.0.1`.

**If the port is taken, only the log says so.** The daemon prints `address already in use`
and exits with status 3, while whatever already held the port keeps answering — a bare
`{"detail":"Not Found"}`, typically, which reads like a shield bug rather than a port
clash. Check the log, or pick another port.

### Point your client at it

| Client | Setting |
|---|---|
| OpenAI SDKs, curl | `OPENAI_BASE_URL=http://127.0.0.1:8199/v1` |
| Anthropic SDK | `anthropic.Anthropic(base_url="http://127.0.0.1:8199", api_key=...)` |
| Claude Code | `ANTHROPIC_BASE_URL=http://127.0.0.1:8199`, with the daemon's `--upstream` set to `https://api.anthropic.com/v1` |
| Hermes | `model.base_url: http://127.0.0.1:8199/v1` in `~/.hermes/config.yaml` |
| Continue | `apiBase: http://127.0.0.1:8199/v1` on a `provider: openai` model — not yet verified |
| Cursor | does not work against a local shield — see below |

Claude Code sends its whole context through the shield — system prompt, `CLAUDE.md`, tool
results — and all of it is examined, not only what you typed. That is the point, and it
also means a credential shape anywhere in that context stops the request with a `400`
instead of reaching the provider.

Cursor is the exception, and not a fixable one: it builds prompts on its own servers and
calls the configured base URL from there, so `127.0.0.1` is unreachable and the request
never arrives. Publishing the shield to the internet would not buy much either — the
prompt reaches Cursor's backend before the shield ever sees it, which is the disclosure
the shield exists to prevent.

### The full tier

Names, organizations and places need the `ner` extra and one pipeline per language you
write in:

```bash
VIRTUAL_ENV=~/.pii-shield uv pip install \
  "pii-shield[ner,server,surrogates] @ git+https://github.com/radianceteam/pii-shield" \
  "https://github.com/explosion/spacy-models/releases/download/ru_core_news_lg-3.8.0/ru_core_news_lg-3.8.0-py3-none-any.whl"
~/.pii-shield/bin/pii-shieldd --language ru --port 8199 --upstream https://api.deepseek.com/v1
```

That is **800 MB on disk and about 1.1 GB resident per language**. Reading the pipeline off
a cold disk takes half a minute; once the file is in the page cache a restart takes three
seconds. `--download-models` fetches a missing pipeline instead of refusing to start, which
is worth having on a workstation and not in a container.

### Check what the provider actually receives

From your side a working round trip and a silent pass-through look the same: either way you
read your own real data in the reply. To see the difference, put a receiver where the
provider would be and read what arrives:

```python
# receiver.py — logs what arrives, answers in the OpenAI shape
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        print("provider got:", body["messages"][0]["content"], flush=True)
        out = json.dumps({"id": "r", "object": "chat.completion", "model": "r",
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}}]}).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out))); self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", 9997), H).serve_forever()
```

Start the shield with `--upstream http://127.0.0.1:9997/v1`, send it a prompt carrying a
phone number and an email, and the receiver prints the stand-ins rather than the originals:

```
provider got: тел +7 748 143-61-92, почта zhdanovasinklitikija@example.net
```

### Docker on a workstation

The image is meant for servers, where baking the pipelines into the build is the point. On
a workstation the `uv` install above takes seconds, while building the image pulls the same
pipelines through Docker Desktop's VM — whose network was measured here at 60 KB/s against
the host's 6.9 MB/s, slow enough to time the build out after an hour until Docker Desktop
was restarted.

Everything in this section was checked on macOS (arm64) on 2026-09-22, against a daemon
installed from git exactly as above; the one row marked otherwise was not.

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
| Own patterns | national identifiers (INN, SNILS, OGRN, BIK, 身份证, マイナンバー, 주민등록번호), banking codes (IBAN, SWIFT/BIC, ABA, LEI), payment cards, credentials, **email and international phone numbers**, and **Russian full names carrying a patronymic** | nothing — regex and checksums |
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
 "redact_credentials": false,
 "entities": ["ABA_ROUTING", "CREDIT_CARD", "IBAN_CODE", "RU_INN", "..."]}
```

`ner_ready` answers "are names detected". It does not answer "is a US SSN detected",
which on a pattern-only deployment is also no — that entity belongs to Presidio's layer.

**The tiers refuse the same things.** A tier decides what can be *detected*, not what is
too dangerous to send. Cards and credentials are found by the dependency-free layer, so
the cheapest deployment blocks exactly what the full one blocks.

### What a turn costs, and why it stops growing

An agent resends its whole conversation on every turn, so the naive cost of shielding it
climbs with the conversation: by the fifth turn the language model has read the first
message five times. Detection is deterministic — the same text, policy and pipeline give
the same spans — so the answer can simply be remembered:

```bash
pii-shieldd --cache-analysis --upstream https://api.anthropic.com/v1
# or PII_SHIELD_CACHE_ANALYSIS=1, or cache_analysis = true in the config file
```

Measured on a growing session, full tier, Russian `lg` pipeline, one turn adding about
20 KB:

| | turn 1 (21 KB) | turn 3 (63 KB) | turn 5 (105 KB) |
|---|---|---|---|
| without the cache | 0.68 s | 1.59 s | 2.69 s |
| with it | 0.54 s | 0.59 s | **0.61 s** |

What is remembered is a hash of the text and the offsets and kinds found in it — never
the text itself, and never a value. The entry is keyed by the policy and by the pipeline
actually loaded, so a result computed without a language model can never be served to a
request that asked for names. It is off by default because it is only worth the memory
where the same text arrives repeatedly, which is what an agent does and what a one-shot
library call does not.

The tier still decides the floor: the pattern-only tier reads 100 KB in 0.06 s and the
full one in 3.6 s, of which about 2 s is the language model's own arithmetic. What the
cache changes is that this price is paid once per piece of text rather than once per
turn.

**Most of that arithmetic was going through the wrong library.** thinc ships its own
build of BLIS and routes every matrix multiplication through it, and that build uses a
generic kernel: on an AMD EPYC 9454P, which has AVX-512, it reached **36 GFLOPS** while
numpy — same process, same matrices — reached **2378** through OpenBLAS. Sixty-five
times. The shield now points the loaded pipeline at the platform's own BLAS, which is
worth **1.34×** end to end with identical findings; `PII_SHIELD_BLAS=blis` puts it back.

Two more costs came from asking for work nobody wanted. Presidio is asked only for the
entities the policy *acts on* — asking for URL and DATE_TIME as well, which it is
configured to leave alone, cost another **1.34×** in patterns over the whole text and a
few hundred more results through a quadratic deduplication. And the Russian lemmatizer
walks its dictionary in pure Python unless `DAWG2` is installed, which is now part of the
`ner` extra: same dictionary, same answers, **1.15×**.

Together, on 199 KB of Russian text: **7.19 s to 2.90 s**.

A fourth cost was a pure function rebuilding its own lookup table. spaCy's Russian
lemmatizer converts every morphological analysis from OpenCorpora notation to Universal
Dependencies, and the table for that conversion is built inside the function — on 199 KB
it was called 374 000 times with **22 distinct tags** between them. Remembering the
answer is **1.41×**, and `PII_SHIELD_NO_PATCHES=1` switches it and anything like it off.

Chasing the arithmetic further is not worth it: after all of this, the matrix
multiplications are **0.10 s of 3.74 s**, so an infinitely fast BLAS — AMD's AOCL, MKL,
anything — would buy 1.03×. What remains is a quadratic deduplication inside Presidio
and the glue around the Russian lemmatizer.

The obvious next candidate is not taken. Dropping the lemmatizer is worth another 1.88×,
and the test suite says no: a US social security number written without dashes is found
only because the words around it raise its score, and that scoring is lemma-based. The
measurement was real and so was the miss.

**And the arithmetic can be spread across cores.** It is not a large language model —
`ru_core_news_lg` is a small convolutional network over word vectors — and it runs inside
numpy, which releases the interpreter lock while it multiplies. So blocks of one payload
genuinely overlap in threads, sharing the single loaded pipeline rather than paying for
another gigabyte each:

```bash
pii-shieldd --parallel 4 --cache-analysis --upstream https://api.anthropic.com/v1
```

| 392 KB through the Russian pipeline | |
|---|---|
| one pass | 6.85 s |
| `nlp.pipe` over four blocks | 6.72 s — batching alone buys nothing |
| two threads | 3.90 s |
| four threads | 2.58 s |
| eight threads | 2.04 s |

Blocks are split on blank lines, because an entity never spans one and because what this
shield reads is already written that way — an agent's turn is a list of messages, a
document is paragraphs. Measured against the same text read in one pass: 781 findings of
782 identical, the odd one out a false positive that the split happened to drop. Payloads
under 32 KB are read in one pass, where threads would cost more than they save.

**Threads are for one request; processes are for several callers.** One process analyzes
one payload at a time, so four developers behind one deployment wait in line — measured,
four different payloads took 15.75 s one after another and 15.10 s all at once, which is
no gain at all. `--workers 4` serves them from four processes instead. Each holds its own
language pipeline, about a gigabyte, so this is memory traded for throughput. One caveat:
a session belongs to the process that made it, so a caller using `/v1/anonymize` and
`/v1/deanonymize` across two requests needs to reach the same worker. The proxy is
unaffected — its sessions never outlive a request.

### Refusing a credential, or redacting it

A credential is refused by default, and for a person pasting a key by hand that is the
useful answer: the request stops, and the key never went anywhere.

A coding agent is the case it does not fit. The agent resends its whole conversation on
every turn, so one connection string anywhere in that history refuses every following
request — and the history only grows, so the session never recovers. That is not a shield
protecting a workflow; it is a shield ending it.

```bash
pii-shieldd --redact-credentials --upstream https://api.openai.com/v1
# or PII_SHIELD_REDACT_CREDENTIALS=1, or redact_credentials = true in the config file
```
```python
Policy.for_language("ru").model_copy(update={"redact_credentials": True})
```

The credential is then replaced by `<SECRET_CONNECTION_STRING>` and the request proceeds.
The replacement is **one-way**: nothing is written to the session map, so unlike a name
the real value cannot come back in an answer — which is the point, since a model has no
business repeating a key it was never given. `/v1/anonymize` reports it next to
`names_analyzed`:

```json
{"text": "подключение не поднимается: <SECRET_CONNECTION_STRING>",
 "names_analyzed": true, "credentials_redacted": true}
```

Only credentials move. Payment cards and permanent national identifiers are still
refused, because a plausible fake card or passport number is worse than a refusal: the
model may reason about it, or write it into its answer as if it were real. And an
explicit `ALLOW` rule stays allowed — the switch relaxes a refusal, it does not overrule
a decision the caller made on purpose.

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

Three jobs the proxy cannot do, and they are what this is for:

- **your code calls the model itself** — from a runtime that cannot import this library,
  or through an SDK feature the proxy does not cover (batch, embeddings, files);
- **the text is not going to a model at all** — logs, tickets, exports, user content on
  its way into a database. `Scope.DISPLAY` exists for exactly this;
- **you need the findings**, not just the cleaned text: what was found, where, of what
  kind. The proxy returns a chat completion and nothing else.

```bash
pii-shieldd --port 8099          # binds 127.0.0.1 by default
```

```
POST /v1/anonymize      {text, session_id?, language?, surrogate_language?, policy?}
                        → {text, session_id, findings[], names_analyzed,
                           credentials_redacted}
POST /v1/deanonymize    {text, session_id, consume?}  → {text}
DELETE /v1/session/{id}
GET  /healthz
```

**The session id comes from the shield.** Pass back the one `/v1/anonymize` returned,
and one real value keeps one stand-in for the whole conversation. An id this process
never issued — invented, expired (an hour by default) or lost to a restart — is a
**404**, on both endpoints. It used to be a 200 with the text unchanged, which is
indistinguishable from a successful round trip over a text that mentioned nobody: the
caller was handed stand-ins and told everything went fine.

`language` selects the language for that request; a language with no pipeline installed
**fails closed** rather than answering 200 with the text unexamined. Unknown fields are
rejected rather than ignored, so a misspelled one is a 422 and not a silent default.

`policy` carries overrides only: anything omitted keeps the language preset's value, so
`{"language": "en"}` means "the usual policy, in English" rather than "a policy with no
rules". Pass `rules` explicitly — including `[]` — to replace them.

| | Meaning |
|---|---|
| `400` with `"type": "pii_shield_blocked"` | The policy refuses to send this at all. Entity *kinds* only, never the value |
| `404` with `"code": "unknown_session"` | That session is not here. Nothing can be restored from it |
| `422` | Malformed request — an unknown field, typically |
| `503` | Detection failed, so nothing was cleaned and nothing may be sent |

See [`examples/client.mjs`](examples/client.mjs) for the shape of all four.

Python consumers should import `Shield` directly — an in-process call has no serialization
cost and no window in which the raw payload exists on a socket.

**One limit if you put this endpoint on a network for several tenants.** Sessions live in
the shield, and the token that reaches the endpoint reaches all of them: a caller holding
another caller's session id can restore that session. Ids are unguessable, so this is
about a leaked id rather than a guessed one — but it means one shield per tenant, or an
authorizing layer in front, if tenants must not be able to read each other's mappings.
The proxy has no such surface, because its sessions never leave the process that made
them. Should the multi-tenant sidecar become a real deployment, the fix is to stop
holding sessions at all — return the mapping to the caller and take it back on the way
in, so a caller can only ever restore what it was given.

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
parts; a single word maps to the word in the same position, so a lone surname does not
drag a three-part name into the middle of a sentence. Identifiers are never matched
loosely — a checksummed value comes back verbatim or not at all. Streaming does the
same, withholding a window wide enough for the longest name plus its endings.

**A loose match may never write over a real value, and never guesses.** Three rules,
each of them the answer to output this produced:

- **What has been written is never matched again.** The exact pass restored *"Петру
  Николаевичу Васильеву"*, and the stem of the stand-in *"Василиса"* then matched the
  real surname *"Васильеву"* that had just been written and replaced it with
  *"Николаевичу"*. Replacements are protected; later passes see only untouched text.
- **A lone word is matched whole, not by stem.** The stand-in *"Игнатьев"* matched
  *"Игнатов"* — a different person standing next to the name — because three characters
  had been trimmed off the stem. A single word now matches only itself, or itself with
  its last letter replaced by an ending.
- **No correspondence, no replacement.** Word *i* maps to word *i*, which means nothing
  when the two names are written in different orders: *"Гуляев Архип Юлианович"* against
  *"Анна Сергеевна Кузнецова"* turned *"Архип Юлианович"* into *"Сергеевна Кузнецова"*.
  Where the patronymic sits at different indices, nothing is replaced and the stand-in
  is left standing. A reader who sees an invented name can ask about it; a reader who
  sees the wrong real name cannot.

The stand-in is chosen so that it can come back: same word order as the original, the
gender its patronymic announces, no part shorter than five letters (*"Лука"* returns as
*"Луку"*, and four letters cannot be told from the start of another name), and no titles
or qualifications — Faker offers *"тов. Некрасова Майя"* and *"Wendy King MD"*, and
neither is a name.

One limit remains: the stand-in is inserted in the nominative regardless of the case the
sentence wanted, which leaves the text the model reads slightly ungrammatical. Agreeing
it with the original would need a morphological generator and is not done today.

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

**The model reads sentences; an agent sends forms.** Measured on `ru_core_news_lg`,
`"ФИО: Пётр Николаевич Васильев"` produced no finding at all, and
`"| Пётр Николаевич Васильев | менеджер |"` produced only *"Николаевич Васильев"* — the
first name went to the provider while the response looked anonymized. Tool output, table
rows and CRM dumps are most of what an agent sends, so this is not an edge case. A
Russian patronymic is a near-unique anchor (`-ович/-евич`, `-овна/-евна`, through every
case ending), so those names are now found by pattern as well, in both word orders and as
initials — `RU_FULL_NAME`, no model involved. A name written without a patronymic
(*"позвони Ивану Петрову"*) still needs the pipeline, which is why `names_analyzed` stays
`false` on the pattern-only tier: finding some names is not analyzing names.

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
