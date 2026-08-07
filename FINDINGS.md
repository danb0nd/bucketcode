# Findings

Everything below was measured, not estimated. Where a number is an estimate it
says so. Where an earlier claim turned out to be wrong, the correction is kept
rather than the original quietly replaced.

Setup: `bucketcode` editing generated bucketlang programs, Claude Sonnet 5
unless stated, identical prompt and identical gates on both arms.

---

## 1. The headline

Editing one bucket versus rewriting the file, same task, same model:

| buckets in program | bucket path | whole file | ratio |
|--:|--:|--:|--:|
| 5 | $0.0102 | $0.0063 | 0.61× |
| 22 | $0.0107 | $0.0209 | 1.95× |
| 82 | $0.0105 | $0.0737 | **7.02×** |

Two things matter more than the 7×.

**Break-even is around 10 buckets.** Below it the fixed cost — system prompt
plus nine tool schemas, roughly 1,560 tokens on *every* turn — is larger than
the file being replaced. At that size sending the whole file is genuinely
cheaper, and the tool reports that rather than hiding it.

**The bucket path is flat.** $0.0102 → $0.0107 → $0.0105 across a 16× increase
in program size. The whole-file arm rose 12×. Nothing in the curve suggests the
gap stops widening.

## 2. The saving is output, not input

This was the surprise, and it reframes the whole thesis.

| | bucket path | whole file |
|---|--:|--:|
| input tokens | 1,082 | 4,362 |
| **output tokens** | **325** | **4,042** |

Output bills at 5× input, so the output gap dominates the cost difference.

The project was framed around *not reading* irrelevant code. The larger effect
is **not rewriting** it. A whole-file edit has to re-emit every line it did not
change, and pays premium rates to do so. Anything that lets a model emit only
the changed region captures most of this — the retrieval half is the smaller
win.

## 3. Subgraph cost tracks structure, not program size

**This corrects an earlier claim.** A first synthetic sweep showed subgraph cost
as a flat 75 tokens from 2 buckets to 322, and that was presented as the
headline property. It was an artefact: the generated helpers called nothing, so
the target's neighbourhood never grew no matter how many were added.

Measured on a real program:

| bucket | subgraph tokens | why |
|---|--:|---|
| `greet` | 39 | no edges |
| `with_tax` | 68 | 1 caller |
| `cents` | 75 | 1 caller |
| `main` | 76 | 2 callees |
| `subtotal` | 104 | recursive — its own neighbour |

Depth compounds it: `main` costs 76 tokens at depth 1, 87 at depth 2.

The real property is weaker than "flat" and worth stating precisely: **cost is
proportional to a bucket's neighbourhood, not to the size of the program.** A
densely connected bucket in a small program can cost more than an isolated one
in a large program.

## 4. The model used the efficient tools without being forced

Given both bucket-native tools and generic `read_file` / `write_file` /
`run_command` fallbacks, every observed run reached for the bucket tools first
and never called `read_file`.

The lever was tool *descriptions* — the fallbacks are labelled `EXPENSIVE` and
`UNCHECKED` with a sentence on what they cost. No system-prompt rule was needed.

A typical run:

```
turn 1  find_buckets("subtotal of items in cart")  +  list_buckets   (parallel)
turn 2  read_bucket(#b00000003)
turn 3  edit_bucket(...)   →   accepted. 2/2 related test(s) pass.
```

## 5. Descriptions were kept honest, unprompted

Asked to make `subtotal` sum prices rather than count them, the model rewrote
the body *and* the description:

> before: `"add up the prices of every item in a cart"`
> after: `"add up the prices of every item in a cart, returning the sum of the list (0 for an empty cart)"`

It documented the empty-cart case it had just written. Nothing asked it to
mention that specifically — the instruction was only that descriptions are how
a bucket is found and must match what the code does.

This matters because descriptions are load-bearing twice over: they are the
retrieval index, and they are what a neighbouring bucket receives *instead of*
a body. A drifted description doesn't degrade gracefully; it hands a future
reader a confident, wrong summary of code they cannot see.

## 6. Model choice moves cost more than context strategy does

Same task, same 22-bucket program:

| model | cost | turns |
|---|--:|--:|
| Haiku 4.5 | $0.0095 | 3 |
| Sonnet 5 | $0.0196 | 3 |
| Opus 5 | $0.0269 | 3 |

All three succeeded. On work this mechanical the cheaper model is the better
default, and picking it saves more than the 1.95× context win at that size.

Sonnet also explored more efficiently than Opus: at 82 buckets it called
`find_buckets` alone (1,082 input tokens), while Opus called `list_buckets`
too and paid 7,674. Exploration strategy is model-dependent, and
`list_buckets` is O(program) — see §8.

## 7. What a token count does not capture

Both arms produced correct, compiling, test-passing code. On cost alone at
5 buckets, the whole-file arm wins.

But the arms are not equivalent in what they can *prove*. The bucket path
checks that an edit changed exactly one bucket, by comparing every bucket's
content hash before and after. A whole-file rewrite cannot: the model returns a
new file, and there is no way to distinguish an intended change from an
unintended one.

This is not hypothetical. During development, a body whose braces closed early
silently redefined a neighbouring bucket — it compiled, and the target's tests
still passed, because the target was untouched. Only the atomicity check caught
it.

So the honest framing is: **below break-even you are trading a correctness
guarantee for a small cost saving**, not choosing between equivalent options.

## 8. Known problems

**`list_buckets` is O(program).** It returns every bucket — 1,643 tokens at 82
buckets — and stays in conversation history for every later turn. When a model
calls it (Opus does, Sonnet often doesn't), the bucket path stops being flat.
It should be paginated or capped.

**The benchmark generator is unrealistic.** Its helpers form no call graph,
which is what produced the false flat line in §3. A generator that builds
realistic call structure would give trustworthier synthetic numbers.

**Single file only.** No multi-file support, which is the largest gap versus a
conventional coding agent.

**Small sample.** Three program sizes, one task shape, a handful of runs. Enough
to show direction, not enough for confidence intervals.

**Untested at scale.** Nothing here has run against a program of hundreds of
buckets written by a person rather than a generator.

## 9. What I would do next

1. **Cap `list_buckets`** — it is the one measured thing breaking the core
   property.
2. **Route by size.** Below ~10 buckets, send the file; above it, use the
   subgraph. The measurement to make that call already exists; the policy
   doesn't.
3. **Regenerate descriptions from code on every edit.** They are the index, and
   nothing currently keeps them true beyond the model choosing to.
4. **Multi-file**, then re-measure — the ratios above are all single-file.

---

## Reproducing

```sh
bucketcode bench --csv bench/context_sweep.csv          # deterministic, free
bucketcode edit prog.bkt "..." --compare                # both arms, live
bucketcode serve prog.bkt                               # UI, Testing tab
```

Raw data in `bench/`. Every UI experiment appends to `<file>.runs.jsonl`.
