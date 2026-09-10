# bucketcode

An agentic coding harness that edits **one bucket at a time** instead of
rewriting files.

A conventional coding agent pastes a file into the model and asks for a
rewrite. bucketcode gives the model a map of the program — every function's
address, contract, and one-line English description, with no bodies — and lets
it pull in only what it decides it needs.

It is built on [bucketlang](https://github.com/danb0nd/bucketlang), whose functions
("buckets") separate three things a normal language fuses into one name:

| | who reads it | stable? |
|---|---|---|
| **address** `#b00000001` | the machine — call edges point here | fixed |
| **label** `double` | a human reading code | sugar |
| **description** `"multiply by two"` | **the model** | mutable |

The description is the interface. It is how a bucket is *found*, and it is what
gets sent *instead of* a bucket's body when something nearby is edited. Both
rest on one bet: English is enough to know what a function does.

## What it does

```sh
bucketcode find app.bkt "round prices to cents"     # English -> bucket address
bucketcode context app.bkt --bucket cents           # what would be sent, and what it costs
bucketcode edit app.bkt "round the tax result too"  # the agent, with tools
```

## The agent

The model drives, using tools. Bucket-native tools come first and generic ones
exist as fallbacks whose descriptions say what they cost:

| tool | |
|---|---|
| `list_buckets` | the whole map — addresses, contracts, descriptions, no bodies |
| `find_buckets` | English query → ranked shortlist |
| `read_bucket` | one body, plus its callers and callees |
| `bucket_graph` | call structure, no bodies |
| `edit_bucket` | change one body + description — **gated** |
| `run_tests` | run every `@test` |
| `read_file` / `write_file` / `run_command` | fallbacks, deliberately described as expensive or unchecked |

### The gates are inside the edit tool

`edit_bucket` does not just write. It splices, then checks:

1. **compiles** — parses and typechecks
2. **atomic** — exactly the target bucket changed, nothing else
3. **behaviour** — that bucket's `@test` cases still pass

A failure comes back as the tool result, with the file, line, column, and the
offending sub-expression. The model fixes it on the next tool call — no outer
retry, no re-sent context.

The atomicity gate is the non-obvious one: a body that closes its own brace can
silently redefine its neighbours while still compiling and still passing the
target's tests.

## Safety

The model gets a terminal, but a small one:

- **no shell** — commands run as an argv array, so `;`, `&&`, `|`, backticks
  and `$()` are inert text rather than syntax
- **allowlist, not blocklist** — only named programs run; `git` and `cargo` are
  further limited to read-only and build subcommands
- **paths are confined** — resolved to canonical form and checked against the
  project root, so `..`, absolute paths, and symlinks out all fail the same way
- **timeouts, output caps, and a full log** of every command, allowed or refused
- `--commands never` disables execution entirely

## Spend limits

Every run is capped, and the ceiling is checked *before* a call is made rather
than discovered after it is paid for:

```sh
bucketcode edit app.bkt "..." --max-requests 4 --max-usd 0.50
```

Caps cover request count, input tokens, output tokens, and estimated USD. An
unrecognised model is priced at the most expensive tier so it cannot slip under
the ceiling by being priced at zero.

## Measurement

`--strategy` selects what gets sent, so the claim can be tested rather than
asserted:

- `whole_file` — the conventional baseline
- `subgraph` — the bucket neighbourhood
- `auto` — measure both, send the cheaper

Both arms share one prompt and one set of gates, so a difference is
attributable to the context and nothing else.

**The subgraph is not always smaller.** On a small file the pack's scaffolding
costs more than the file it replaces. The property that holds is that the cost
of editing one bucket stays flat as the rest of the program grows.

## Status

Early. The retrieval, gates, sandbox, and budget guard are tested; the
end-to-end agent has not yet been run against a large real program.

## Setup

```sh
export ANTHROPIC_API_KEY=...
cargo build --release
```

## Measured results

Live, against the API. Same task, same model (Sonnet 5), same gates on both arms.

| buckets | bucket path | whole file | ratio |
|--:|--:|--:|--:|
| 5 | $0.0102 | $0.0063 | 0.61x |
| 22 | $0.0107 | $0.0209 | 1.95x |
| 82 | $0.0105 | $0.0737 | **7.02x** |

Run it yourself on any edit with `--compare`.

**Break-even is around 10 buckets.** Below that, the fixed prompt and tool-schema
overhead (~1560 tokens, paid every turn) exceeds the file being replaced. Sending
the file is genuinely cheaper there and the tool says so.

**The saving is mostly output, not input.** The whole-file arm re-emits the entire
program — 4042 output tokens at 82 buckets versus 325 for one body. Output bills at
5x input, so that gap dominates.

**Subgraph cost tracks structure, not program size.** In `shop.bkt` it ranges 39–104
tokens depending on how many neighbours a bucket has. An earlier synthetic benchmark
showed it as flat; that was an artefact of generated helpers that called nothing.

**The uncounted difference is correctness.** Both arms passed. But a whole-file
rewrite cannot prove it changed only what was asked — atomicity is unavailable to
it. That guarantee is what a token count does not capture.
