# Changelog

This project is installed from git — nothing here is published to PyPI, where the name
belongs to an unrelated project — so a version is a commit worth coming back for rather
than an upload. [Update](README.md#update) says how to come back for one.

## 0.2.0 — 2026-09-24

The first tagged version. Until now an installed copy was whatever `main` held on the
day it went in, and `pip` kept it silently on a re-run because the version string never
moved. That is the immediate reason this tag exists.

### Fixed

- **Tool calls through the Anthropic route arrived unparseable, and the end of a
  streamed answer could be dropped.** A Russian stand-in cannot be recognised until its
  ending arrives, so the restorer withholds a window and releases it when the block
  closes — and that tail was written after the upstream's `event: content_block_stop`
  line, which made the tail a stop and left the real stop unnamed. Clients dispatch on
  that line. Each event now carries its own name (`4b46306`).
- **A restore could put a real name in the wrong place** — including a name the sentence
  never mentioned. Restoration now protects what it has already written and refuses a
  correspondence it cannot justify (`1a99c7f`, `b3edc73`, `5dca1bf`).
- **A stand-in carrying a title never came back**: "Ing. Marlis Hofmann B.Eng." and
  "Herr Amir Jockel" are not names, and are no longer drawn as them (`f3db391`,
  `e8edae6`).
- **An unknown or expired session answered 200 with the text unchanged**, which reads
  like a restore that found nothing. It is a 404 (`a3468cf`).
- **Whether a stand-in can be declined is decided by the stand-in's own language**, not
  by the language of the text it replaced (`2c84b02`).
- **A Japanese or Korean stand-in could carry the real surname out with it.** The guard
  that rejects "Margaret Smith" for "John Smith" compares whitespace-separated words and
  four-character stems, and a name written without spaces has neither: 田中太郎 drew
  田中 あすか and every check passed. A shared pair of characters, or a shared leading
  character, now disqualifies the candidate (`3ef8e04`).
- Under a read-only root filesystem the suffix-list cache had nowhere to go, so every
  worker warned and re-fetched the public suffix list over HTTP (`f87980e`).

### Added

- **Anthropic Messages API**: `/v1/messages`, streaming, `count_tokens` (`0f50352`).
- **Russian full names by pattern**, where the language model does not find them — a
  name in a form or a table, without a sentence around it (`b0b8117`).
- **`--redact-credentials`**: replace a credential with an unreadable placeholder
  instead of refusing the request. Off by default. A coding agent carries keys in its
  own history, which only grows, so refusing once stops every turn after it (`e4da3fc`).
- **The shield says what it could not put back** — `unrestored` in the sidecar response,
  `X-Pii-Shield-Unrestored` on a proxied one (`9fb7a3c`).

### Faster

The full tier costs what it costs because of arithmetic, not because of the model, and
most of what was measured turned out to be work nobody wanted:

- 199 KB of Russian: **7.19 s → 2.90 s** — the platform's BLAS instead of the generic
  build thinc ships, Presidio asked only for the entities the policy acts on, and the
  Russian lemmatizer's dictionary read from a compiled DAWG (`9da0dc0`).
- Presidio's context lookup walked the token list once per finding: **4.94 s → 3.59 s**
  on 100 KB, identical findings (`9cdc314`).
- `--cache-analysis` for an agent that resends its conversation every turn: **2.69 s at
  turn five → 0.61 s, flat**. What is kept is a hash and the offsets found, never the
  text (`9cdc314`).
- A large payload is read in parallel blocks: 392 KB, **6.85 s → 2.58 s** across four
  (`f64cc44`).
- Presidio's own rewrite of `remove_duplicates`, taken before its release: 5.5 million
  comparisons that removed nothing, 0.60 s of a 3 s detection (`b179d59`, `4ef8435`).
- The Russian lemmatizer rebuilt a hundred-entry table for every token (`4842552`).

### Documentation

- The install block told the reader to `pip install pii-shield`, which installs an
  unrelated project of the same name. It installs from git, and [Update](README.md#update)
  says how to move to a newer commit — `pip` needs `--force-reinstall`, `uv` does not
  (`ee97478`).
