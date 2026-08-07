//! The agent loop: the model drives, using tools.
//!
//! This is the real harness. Instead of pre-computing a context pack and asking
//! for one body back, the model is given tools and decides what it needs — so
//! the compression stops being a fixed policy and becomes something the model
//! negotiates: it starts from a map of labels and descriptions and pulls in a
//! body only when it decides that description was not enough.
//!
//! The important design choice is that **the gates live inside the edit tool**.
//! `edit_bucket` splices, compiles, checks atomicity, and runs the bucket's
//! tests; if any of those fail it returns the diagnostic as an error tool
//! result. The model sees the exact line and column that broke, in the same
//! turn, and fixes it — no outer retry loop, no re-sending context. A rejected
//! edit costs one tool round trip instead of a whole new request.
//!
//! Nothing is written to disk here. The agent mutates an in-memory copy of the
//! source; the caller decides whether to persist the result.

use bucketlang::ast::BucketKind;
use bucketlang::compile::{compile_with_base, BuildProfile, CompileOptions};
use bucketlang::edit::{only_target_changed, run_subject_tests, splice_bucket_body, user_hashes};
use bucketlang::eval::eval_bucket;
use bucketlang::registry::Registry;
use serde::Serialize;
use serde_json::json;
use std::path::{Path, PathBuf};

use crate::budget::Guard;
use crate::index::search;
use crate::llm::{Client, LlmError, Message, Tool, ToolCall};

/// What the agent did, for display and for measurement.
#[derive(Debug, Clone, Serialize)]
pub struct AgentRun {
    /// The run completed without error. True even when no edit was needed.
    pub ok: bool,
    /// An edit was actually applied. Distinct from `ok`: a model that correctly
    /// reports the code already does what was asked has succeeded without
    /// editing, and scoring that as failure made the comparison meaningless.
    pub edited: bool,
    pub turns: usize,
    pub tool_calls: Vec<ToolRecord>,
    /// The final text the model wrote, after it stopped calling tools.
    pub summary: String,
    /// `Some` if the source changed. The caller decides whether to write it.
    pub source: Option<String>,
    pub spend: crate::budget::Spend,
    /// Tokens the whole file would have cost, for comparison.
    pub whole_file_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolRecord {
    pub turn: usize,
    pub name: String,
    pub input: serde_json::Value,
    pub ok: bool,
    /// Truncated for display; the model saw the whole thing.
    pub result: String,
}

/// The tools the model can call.
///
/// Deliberately small. Each one is a *bucketlang* operation, not a filesystem
/// operation — there is no raw read/write, so the model cannot corrupt the file
/// by writing text at an offset. Every mutation goes through `edit_bucket` and
/// therefore through the gates.
pub fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "list_buckets".into(),
            description: "List every bucket in the program: address, label, contract, and description. \
                          No bodies. Start here — the descriptions are usually enough to decide what to change, \
                          and this costs a fraction of reading the file."
                .into(),
            input_schema: json!({"type": "object", "properties": {}, "additionalProperties": false}),
        },
        Tool {
            name: "find_buckets".into(),
            description: "Search buckets by describing what you want in English. Matches against descriptions, \
                          labels, and parameter names, and returns a ranked shortlist. Use this when you know \
                          what behaviour you want but not which bucket has it."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What you are looking for, in plain English."}
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        },
        Tool {
            name: "read_bucket".into(),
            description: "Read one bucket's full body, plus its callers and callees. Call this only when a \
                          description was not enough — every body you read costs tokens that the description \
                          was meant to save."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Address like #b00000001, or the bucket's label."}
                },
                "required": ["address"],
                "additionalProperties": false
            }),
        },
        Tool {
            name: "edit_bucket".into(),
            description: "Replace one bucket's body and description. The edit is checked before it is accepted: \
                          it must compile, it must change only this bucket, and this bucket's @test cases must \
                          still pass. If any check fails you get the exact error back and can try again. \
                          Always update the description to match what the code now does — it is how this bucket \
                          is found and how it is summarised to other buckets."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Address or label of the bucket to change."},
                    "body": {"type": "string", "description": "The new body expression only — no label, contract, description, or braces."},
                    "description": {"type": "string", "description": "What the bucket does after this edit, in plain English, phrased the way someone would ask for it."}
                },
                "required": ["address", "body", "description"],
                "additionalProperties": false
            }),
        },
        Tool {
            name: "patch_bucket".into(),
            description: "Change part of a bucket's body by replacing an exact snippet, leaving the rest \
                          untouched. PREFER THIS over edit_bucket whenever the body is more than a line or \
                          two: you emit only the fragment that changes instead of retyping the whole body, \
                          and output tokens are the most expensive thing you spend. `find` must appear \
                          exactly once in the body. Same checks as edit_bucket."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Address or label of the bucket."},
                    "find": {"type": "string", "description": "Exact snippet to replace. Must occur exactly once in the body."},
                    "replace": {"type": "string", "description": "What to put in its place."},
                    "description": {"type": "string", "description": "Optional. Update it when the behaviour changed."},
                    "tests": {
                        "type": "array",
                        "description": "Optional. New @test set, applied in the SAME change as the body. Use this whenever an existing test pins the behaviour you are changing — updating the test separately cannot work, because either order fails.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "call": {"type": "string"},
                                "expects": {"type": "string"},
                                "expect_error": {"type": "boolean"}
                            },
                            "required": ["call"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["address", "find", "replace"],
                "additionalProperties": false
            }),
        },
        Tool {
            name: "delete_bucket".into(),
            description: "Remove a bucket. Refused if anything still calls it — fix the callers first. \
                          The program must still compile and every test must still pass."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {"address": {"type": "string"}},
                "required": ["address"],
                "additionalProperties": false
            }),
        },
        Tool {
            name: "set_signature".into(),
            description: "Rename a bucket and/or change its parameters and return type. Call sites are NOT \
                          rewritten for you, so change them in the same session or the program will not \
                          compile and the change will be refused."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "address": {"type": "string"},
                    "label": {"type": "string", "description": "New name. Omit to keep the current one."},
                    "params": {
                        "type": "array",
                        "description": "Full replacement parameter list. Omit to keep the current one.",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                            "required": ["name", "type"],
                            "additionalProperties": false
                        }
                    },
                    "returns": {"type": "string", "description": "New return type. Omit to keep the current one."}
                },
                "required": ["address"],
                "additionalProperties": false
            }),
        },
        Tool {
            name: "set_tests".into(),
            description: "Replace every @test on a bucket at once. This is how you change a test that pins \
                          old behaviour — give the full list you want the bucket to end up with, or an empty \
                          list to remove them all. Each entry is either an expected value or an expected \
                          error. The new tests must pass."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "address": {"type": "string"},
                    "tests": {
                        "type": "array",
                        "description": "The complete set of tests this bucket should have.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "call": {"type": "string", "description": "The call, e.g. double(3)"},
                                "expects": {"type": "string", "description": "Expected value, e.g. 6. Omit when expect_error is true."},
                                "expect_error": {"type": "boolean", "description": "True if the call should fail."}
                            },
                            "required": ["call"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["address", "tests"],
                "additionalProperties": false
            }),
        },
        Tool {
            name: "add_bucket".into(),
            description: "Create a new bucket. Give it a label, typed parameters, a return type, an English \
                          description, and a body. Checked before it is accepted: the program must still \
                          compile, no existing bucket may change, and every test must still pass. Use this \
                          when the task needs behaviour that does not exist yet — do not try to add one with \
                          write_file."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Name, e.g. quadruple. Must not already exist."},
                    "params": {
                        "type": "array",
                        "description": "Parameters in order. Empty for none.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string", "description": "Num, Bool, Str, List[Num], or a declared type"}
                            },
                            "required": ["name", "type"],
                            "additionalProperties": false
                        }
                    },
                    "returns": {"type": "string", "description": "Return type, e.g. Num."},
                    "description": {"type": "string", "description": "What it does, in plain English, phrased the way someone would ask for it."},
                    "body": {"type": "string", "description": "The body expression only — no label, contract, description, or braces."}
                },
                "required": ["label", "params", "returns", "description", "body"],
                "additionalProperties": false
            }),
        },
        Tool {
            name: "bucket_graph".into(),
            description: "Show the call graph: which buckets call which. Use it to see what a change would \
                          affect before making it, or to find your way around an unfamiliar program. \
                          Structure only, no bodies — cheap."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Optional. Limit to one bucket's callers and callees."}
                },
                "additionalProperties": false
            }),
        },
        Tool {
            name: "run_tests".into(),
            description: "Run every @test in the program and report which passed. Use it to confirm the program \
                          is healthy, or to see what is already broken before you start."
                .into(),
            input_schema: json!({"type": "object", "properties": {}, "additionalProperties": false}),
        },
        // ---- generic fallbacks -------------------------------------------
        // Present because some tasks genuinely need them. Their descriptions
        // say what they cost, because the bucket tools exist precisely to
        // avoid that cost and a model reaches for whatever looks simplest.
        Tool {
            name: "read_file".into(),
            description: "Read a whole file. EXPENSIVE — a file costs many times what list_buckets or \
                          read_bucket cost for the same job, and most of it will be irrelevant to your task. \
                          Use it only for files that are not bucketlang source, or when you have already \
                          tried the bucket tools and genuinely need the raw text."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the project root."}},
                "required": ["path"],
                "additionalProperties": false
            }),
        },
        Tool {
            name: "write_file".into(),
            description: "Write a whole file. UNCHECKED — unlike edit_bucket, this does not verify that the \
                          result compiles, that you changed only what you meant to, or that tests still pass. \
                          Never use it to change a bucket; use edit_bucket, which checks all three. Use it only \
                          to create a new file."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the project root."},
                    "contents": {"type": "string"}
                },
                "required": ["path", "contents"],
                "additionalProperties": false
            }),
        },
        Tool {
            name: "run_command".into(),
            description: "Run a command in the project directory. Pass the program and each argument as \
                          separate array entries — there is no shell, so pipes, redirects, and `&&` will not \
                          work and are rejected. Only a small allowlist runs (bkt, cargo, ls, cat, grep, rg, \
                          find, git status/diff/log, wc, head, tail, diff, echo). Prefer run_tests over \
                          `bkt run`; it is cheaper and reports failures better."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Program and arguments, e.g. [\"cargo\", \"test\"]."
                    }
                },
                "required": ["argv"],
                "additionalProperties": false
            }),
        },
    ]
}

pub const SYSTEM: &str = r#"You are a coding agent working on a bucketlang program.

# The model

A program is a graph of small typed functions called buckets. Each has a stable
address (`#b00000001`), a label, a typed contract, an English description, and a
one-expression body.

Identity, name, and meaning are separate on purpose: the address never changes,
so a description can be rewritten freely whenever behaviour changes.

# Descriptions are the interface, not documentation

1. A bucket is **found** by matching English against its description. One that
   does not say what it does cannot be located, however good its code is.
2. A description is what other buckets see **instead of** the body. When
   something else is edited, this bucket appears as its signature and
   description only.

So a stale description is worse than a missing one: it makes the bucket
unfindable, or findable for the wrong reason, and hands a future reader a
confident summary of code they cannot see. Update it on every edit, phrased as
someone would ask for the behaviour — "round a price down to whole cents", not
"helper" or "uses floor and mod".

# How to work

Start with `list_buckets` or `find_buckets`. The descriptions are usually enough
to decide what to change. Read a body only when a description genuinely is not
enough — each one you read spends the tokens the description exists to save.

Then call `edit_bucket`. It checks your edit before accepting it and hands back
the precise error if something is wrong. Fix and call it again.

# Syntax

- literals `1`, `true`, `"text"`; arithmetic `+ - * / **` (`+` joins two `Str`)
- comparison `== != < > <= >=`, logic `&& || !`
- `if cond then a else b`, `match x { None => …, Some(v) => … }`
- calls by label `double(21)`; pipe `x |> double`
- lists `[1, 2]`, records `{ x: 1 }`, field access `p.x`
- bindings `name = expr` on their own line; the last expression is the result
- cores: `print`, `error`, `pow`, `mod`, `floor`, `abs`, `to_json`, `from_json`,
  `list_len`, `list_nth`, `list_append`, `list_concat`, `list_remove`

Write only the body expression. Never close a brace you did not open — a body
that ends the bucket early is rejected, because it silently redefines its
neighbours.

# Limits

You can do all of this:

- `patch_bucket` — change part of a body. **Prefer this** whenever the body is
  more than a line or two: you emit only the fragment that changes, and output
  tokens cost five times what input does.
- `edit_bucket` — replace a whole body and description. For short bodies.
- `add_bucket` / `delete_bucket` — create or remove one.
- `set_signature` — rename, or change parameters and return type. Call sites are
  not rewritten for you; fix them in the same session.
- `set_tests` — replace a bucket's `@test` lines on their own.

When a change to behaviour is pinned by an existing test, change both **in one
call**: pass `tests` to `patch_bucket` alongside `find`/`replace`. Doing it in
two steps cannot work — new tests fail against the old body, and the old test
fails against the new body.

You cannot add or remove type aliases, modules, or imports. If a task needs one,
say so plainly and stop; do not reach for `write_file`, which bypasses every
check and will not help.

A task may turn out to need no change at all. If the code already does what was
asked, say so and stop. That is a correct outcome, not a failure.

When the task is done, stop calling tools and say briefly what you changed."#;

fn opts() -> CompileOptions {
    CompileOptions {
        strict: true,
        profile: BuildProfile::Dev,
    }
}

/// The program the agent is working on. Mutated in memory only.
struct Workspace {
    source: String,
    file: PathBuf,
    dirty: bool,
    sandbox: crate::sandbox::Sandbox,
}

impl Workspace {
    fn compile(&self) -> Result<bucketlang::CompileResult, String> {
        compile_with_base(&self.source, opts(), Path::new(&self.file))
            .map_err(|e| e.with_source(&self.source, None).render(Some(&self.source)))
    }
}

fn render_bucket_list(reg: &Registry) -> String {
    let mut s = String::from("address      label / contract                     description\n");
    for (addr, b) in &reg.buckets {
        if b.kind != BucketKind::User {
            continue;
        }
        let contract = format!(
            "{}({}) -> {}",
            b.label.clone().unwrap_or_default(),
            b.contract
                .params
                .iter()
                .map(|p| format!("{}: {}", p.name, p.ty.name()))
                .collect::<Vec<_>>()
                .join(", "),
            b.contract.ret.name()
        );
        s.push_str(&format!("{addr}  {contract:36}  \"{}\"\n", b.desc));
    }
    s
}

/// Execute one tool call against the workspace.
fn dispatch(ws: &mut Workspace, call: &ToolCall) -> (String, bool) {
    let compiled = match ws.compile() {
        Ok(c) => c,
        Err(e) => return (format!("the program does not currently compile:\n{e}"), true),
    };
    let reg = &compiled.registry;

    match call.name.as_str() {
        "list_buckets" => (render_bucket_list(reg), false),

        "find_buckets" => {
            let q = call.input.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let matches = search(reg, q, 6);
            if matches.is_empty() {
                return (
                    format!("nothing matched \"{q}\". Try list_buckets to see everything."),
                    false,
                );
            }
            (crate::index::render_shortlist(&matches), false)
        }

        "read_bucket" => {
            let addr = call.input.get("address").and_then(|v| v.as_str()).unwrap_or("");
            match crate::context::build_context(reg, addr, 1) {
                Ok(mut pack) => {
                    // Show the body as it is written in the file, not as the
                    // AST renders it. The renderer adds parentheses -- source
                    // `p * 1.08` prints as `(p * 1.08)` -- so a model patching
                    // against what it was shown would never match.
                    if let Some(span) = reg
                        .resolve_target(addr.trim())
                        .and_then(|a| reg.get(&a))
                        .and_then(|b| b.body_span)
                    {
                        if let Some(src) = ws.source.get(span.start..span.end) {
                            pack.target.body_labelled = src.trim_end().to_string();
                        }
                    }
                    (pack.render(), false)
                }
                Err(e) => (format!("{e}"), true),
            }
        }

        "run_tests" => {
            let mut out = Vec::new();
            let (mut run, mut passed) = (0usize, 0usize);
            let mut failures = Vec::new();
            for tid in &reg.test_ids {
                run += 1;
                let b = reg.get(tid).unwrap();
                let result = eval_bucket(reg, tid, &[], &mut out);
                let ok = if b.expect_error {
                    result.is_err()
                } else {
                    result.is_ok()
                };
                if ok {
                    passed += 1;
                } else {
                    let subject = b.subject.clone().unwrap_or_else(|| tid.clone());
                    let why = result
                        .err()
                        .map(|e| e.with_source(&ws.source, None).render(Some(&ws.source)))
                        .unwrap_or_else(|| "expected an error, got a value".into());
                    failures.push(format!("{subject}: {why}"));
                }
            }
            let mut s = format!("{passed}/{run} tests passed\n");
            for f in failures {
                s.push_str(&format!("\n{f}\n"));
            }
            (s, passed != run)
        }

        "edit_bucket" => {
            let addr = call.input.get("address").and_then(|v| v.as_str()).unwrap_or("");
            let body = call.input.get("body").and_then(|v| v.as_str()).unwrap_or("");
            let desc = call.input.get("description").and_then(|v| v.as_str());

            let before = user_hashes(reg);
            let target_addr = match reg.resolve_target(addr.trim()) {
                Some(a) => a,
                None => return (format!("unknown bucket {addr}"), true),
            };

            // Gate 0: the splice must land on the target.
            let spliced = match splice_bucket_body(&ws.source, reg, addr, body) {
                Ok(s) => s,
                Err(e) => return (e.to_string(), true),
            };
            // The description is part of the edit, so rewrite it in the same pass.
            let spliced = match desc {
                Some(d) => replace_description(&spliced, reg, addr, d).unwrap_or(spliced),
                None => spliced,
            };

            // Gate 1: it has to compile.
            let recompiled = match compile_with_base(&spliced, opts(), Path::new(&ws.file)) {
                Ok(c) => c,
                Err(e) => {
                    return (
                        e.with_source(&spliced, None).render(Some(&spliced)),
                        true,
                    )
                }
            };

            // Gate 2: exactly one bucket changed.
            if let Err(msg) =
                only_target_changed(&before, &user_hashes(&recompiled.registry), &target_addr)
            {
                return (
                    format!("{msg}\n\nhint: write only the expression; do not close the bucket's brace"),
                    true,
                );
            }

            // Gate 3: the bucket's own tests still pass.
            let mut sink = std::io::sink();
            match run_subject_tests(&recompiled.registry, addr, &mut sink) {
                Ok((run, passed)) => {
                    ws.source = spliced;
                    ws.dirty = true;
                    (
                        format!("accepted. {passed}/{run} related test(s) pass."),
                        false,
                    )
                }
                Err(e) => (
                    e.with_source(&spliced, None).render(Some(&spliced)),
                    true,
                ),
            }
        }

        // Surgical body edit: emit only the fragment that changes.
        "patch_bucket" => {
            let addr = call.input.get("address").and_then(|v| v.as_str()).unwrap_or("");
            let find = call.input.get("find").and_then(|v| v.as_str()).unwrap_or("");
            let replace = call.input.get("replace").and_then(|v| v.as_str()).unwrap_or("");
            let desc = call.input.get("description").and_then(|v| v.as_str());
            if find.is_empty() {
                return ("`find` cannot be empty".into(), true);
            }

            let Some(resolved) = reg.resolve_target(addr.trim()) else {
                return (format!("unknown bucket {addr}"), true);
            };
            let Some(span) = reg.get(&resolved).and_then(|b| b.body_span) else {
                return (
                    format!("{addr} has no editable body here (it may be a core or an import)"),
                    true,
                );
            };
            let body = &ws.source[span.start..span.end];

            // Ambiguity is an error, not a coin flip: replacing "the first one"
            // is how a patch silently changes the wrong thing.
            let hits = body.matches(find).count();
            if hits == 0 {
                return (
                    format!(
                        "`find` does not appear in the body of {addr}. The body, exactly as written:\n\
                         ---\n{}\n---",
                        body.trim()
                    ),
                    true,
                );
            }
            if hits > 1 {
                return (
                    format!(
                        "`find` appears {hits} times in {addr}; include enough surrounding text to make \
                         it unique. The body is:\n{}",
                        body.trim()
                    ),
                    true,
                );
            }

            let new_body = body.replacen(find, replace, 1);
            let candidate = format!(
                "{}{}{}",
                &ws.source[..span.start],
                new_body,
                &ws.source[span.end..]
            );
            let candidate = match desc {
                Some(d) => replace_description(&candidate, reg, addr, d).unwrap_or(candidate),
                None => candidate,
            };

            // If new tests were supplied, rewrite the declaration so body and
            // tests land in one change. Applying them separately deadlocks.
            let candidate = match call.input.get("tests") {
                Some(t) => {
                    let lines = match render_tests(t) {
                        Ok(l) => l,
                        Err(e) => return (e, true),
                    };
                    let b = reg.get(&resolved).unwrap();
                    let keep_entry = decl_annotations(&ws.source, reg, addr)
                        .iter()
                        .any(|a| a.trim().starts_with("@entry"));
                    let params = b
                        .contract
                        .params
                        .iter()
                        .map(|p| format!("{}: {}", p.name, p.ty.name()))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let mut decl = String::new();
                    for l in &lines {
                        decl.push_str(l);
                        decl.push('\n');
                    }
                    if keep_entry {
                        decl.push_str("@entry\n");
                    }
                    decl.push_str(&format!(
                        "{}({params}) -> {} \"{}\" {{{}\n}}",
                        b.label.clone().unwrap_or_else(|| resolved.clone()),
                        b.contract.ret.name(),
                        desc.unwrap_or(&b.desc).replace('"', "'"),
                        new_body.trim_end()
                    ));
                    match splice_decl(&candidate, reg, addr, Some(&decl)) {
                        Ok(c) => c,
                        Err(e) => return (e, true),
                    }
                }
                None => candidate,
            };

            let before = user_hashes(reg);
            match check_change(ws, &candidate, &before, 0, 0, &[resolved.clone()]) {
                Ok(c) => {
                    ws.source = candidate;
                    ws.dirty = true;
                    (
                        format!(
                            "patched {addr}. {} test(s) pass.",
                            c.registry.test_ids.len()
                        ),
                        false,
                    )
                }
                Err(e) => (e, true),
            }
        }

        "delete_bucket" => {
            let addr = call.input.get("address").and_then(|v| v.as_str()).unwrap_or("");
            let Some(resolved) = reg.resolve_target(addr.trim()) else {
                return (format!("unknown bucket {addr}"), true);
            };

            // Refuse while anything still calls it, so the failure names the
            // callers instead of surfacing as a confusing compile error.
            let g = bucketlang::graph::build_graph(reg);
            if let Some(node) = g.get(&resolved) {
                let callers: Vec<String> = node
                    .inn
                    .iter()
                    .filter(|c| *c != &resolved)
                    .filter_map(|c| reg.get(c).map(|b| b.label.clone().unwrap_or_else(|| c.clone())))
                    .collect();
                if !callers.is_empty() {
                    return (
                        format!(
                            "{addr} is still called by {}. Change those first.",
                            callers.join(", ")
                        ),
                        true,
                    );
                }
            }

            let candidate = match splice_decl(&ws.source, reg, addr, None) {
                Ok(c) => c,
                Err(e) => return (e, true),
            };
            let before = user_hashes(reg);
            match check_change(ws, &candidate, &before, 0, 1, &[]) {
                Ok(_) => {
                    ws.source = candidate;
                    ws.dirty = true;
                    (format!("deleted {addr}."), false)
                }
                Err(e) => (e, true),
            }
        }

        "set_signature" => {
            let addr = call.input.get("address").and_then(|v| v.as_str()).unwrap_or("");
            let Some(resolved) = reg.resolve_target(addr.trim()) else {
                return (format!("unknown bucket {addr}"), true);
            };
            let Some(b) = reg.get(&resolved) else {
                return (format!("unknown bucket {addr}"), true);
            };

            let label = call
                .input
                .get("label")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| b.label.clone().unwrap_or_else(|| resolved.clone()));
            let returns = call
                .input
                .get("returns")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| b.contract.ret.name());
            let params = match call.input.get("params").and_then(|v| v.as_array()) {
                Some(a) => a
                    .iter()
                    .filter_map(|p| {
                        Some(format!(
                            "{}: {}",
                            p.get("name")?.as_str()?,
                            p.get("type")?.as_str()?
                        ))
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
                None => b
                    .contract
                    .params
                    .iter()
                    .map(|p| format!("{}: {}", p.name, p.ty.name()))
                    .collect::<Vec<_>>()
                    .join(", "),
            };

            let Some(span) = b.body_span else {
                return (format!("{addr} has no editable declaration here"), true);
            };
            let body = ws.source[span.start..span.end].trim_end().to_string();
            let annotations = decl_annotations(&ws.source, reg, addr);
            let mut decl = String::new();
            for a in &annotations {
                decl.push_str(a);
                decl.push('\n');
            }
            decl.push_str(&format!(
                "{label}({params}) -> {returns} \"{}\" {{{body}\n}}",
                b.desc.replace('"', "'")
            ));

            let candidate = match splice_decl(&ws.source, reg, addr, Some(&decl)) {
                Ok(c) => c,
                Err(e) => return (e, true),
            };
            let before = user_hashes(reg);
            // Renaming changes the label, not the address, so the bucket's own
            // hash may move but nothing should appear or vanish.
            match check_change(ws, &candidate, &before, 0, 0, &[resolved.clone()]) {
                Ok(_) => {
                    ws.source = candidate;
                    ws.dirty = true;
                    (format!("{addr} is now {label}({params}) -> {returns}."), false)
                }
                Err(e) => (e, true),
            }
        }

        "set_tests" => {
            let addr = call.input.get("address").and_then(|v| v.as_str()).unwrap_or("");
            let Some(resolved) = reg.resolve_target(addr.trim()) else {
                return (format!("unknown bucket {addr}"), true);
            };

            let mut lines = Vec::new();
            for t in call
                .input
                .get("tests")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default()
            {
                let c = t.get("call").and_then(|v| v.as_str()).unwrap_or("").trim();
                if c.is_empty() {
                    continue;
                }
                if t.get("expect_error").and_then(|v| v.as_bool()) == Some(true) {
                    lines.push(format!("@test_error {c}"));
                } else {
                    let e = t.get("expects").and_then(|v| v.as_str()).unwrap_or("").trim();
                    if e.is_empty() {
                        return (
                            format!("test `{c}` needs `expects`, or `expect_error: true`"),
                            true,
                        );
                    }
                    lines.push(format!("@test {c} == {e}"));
                }
            }

            // Keep @entry if it was there; only the @test lines are replaced.
            let keep_entry = decl_annotations(&ws.source, reg, addr)
                .iter()
                .any(|a| a.trim().starts_with("@entry"));
            let Some(span) = reg.get(&resolved).and_then(|b| b.body_span) else {
                return (format!("{addr} has no editable declaration here"), true);
            };
            let b = reg.get(&resolved).unwrap();
            let body = ws.source[span.start..span.end].trim_end().to_string();
            let params = b
                .contract
                .params
                .iter()
                .map(|p| format!("{}: {}", p.name, p.ty.name()))
                .collect::<Vec<_>>()
                .join(", ");

            let mut decl = String::new();
            for l in &lines {
                decl.push_str(l);
                decl.push('\n');
            }
            if keep_entry {
                decl.push_str("@entry\n");
            }
            decl.push_str(&format!(
                "{}({params}) -> {} \"{}\" {{{body}\n}}",
                b.label.clone().unwrap_or_else(|| resolved.clone()),
                b.contract.ret.name(),
                b.desc.replace('"', "'")
            ));

            let candidate = match splice_decl(&ws.source, reg, addr, Some(&decl)) {
                Ok(c) => c,
                Err(e) => return (e, true),
            };
            let before = user_hashes(reg);
            match check_change(ws, &candidate, &before, 0, 0, &[resolved.clone()]) {
                Ok(c) => {
                    ws.source = candidate;
                    ws.dirty = true;
                    (
                        format!(
                            "{addr} now has {} test(s); all {} test(s) in the program pass.",
                            lines.len(),
                            c.registry.test_ids.len()
                        ),
                        false,
                    )
                }
                Err(e) => (e, true),
            }
        }

        "add_bucket" => {
            let label = call.input.get("label").and_then(|v| v.as_str()).unwrap_or("").trim();
            let returns = call.input.get("returns").and_then(|v| v.as_str()).unwrap_or("Num");
            let desc = call.input.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let body = call.input.get("body").and_then(|v| v.as_str()).unwrap_or("");

            if label.is_empty() {
                return ("label is required".into(), true);
            }
            if desc.trim().is_empty() {
                return (
                    "a description is required — it is how this bucket is found and how it is \
                     summarised to other buckets"
                        .into(),
                    true,
                );
            }
            if reg.resolve_target(label).is_some() {
                return (
                    format!("a bucket named {label} already exists; use edit_bucket to change it"),
                    true,
                );
            }

            let params = call
                .input
                .get("params")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|p| {
                            Some(format!(
                                "{}: {}",
                                p.get("name")?.as_str()?,
                                p.get("type")?.as_str()?
                            ))
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();

            // Append, indenting the body the way the rest of the file is written.
            let indented = body
                .trim()
                .lines()
                .map(|l| format!("  {}", l.trim_end()))
                .collect::<Vec<_>>()
                .join("\n");
            let added = format!(
                "\n{label}({params}) -> {returns} \"{}\" {{\n{indented}\n}}\n",
                desc.replace('"', "'")
            );
            let candidate = format!("{}{}", ws.source.trim_end(), format!("\n{added}"));

            let before = user_hashes(reg);
            let recompiled = match compile_with_base(&candidate, opts(), Path::new(&ws.file)) {
                Ok(c) => c,
                Err(e) => {
                    return (
                        e.with_source(&candidate, None).render(Some(&candidate)),
                        true,
                    )
                }
            };
            let after = user_hashes(&recompiled.registry);

            // Same atomicity idea as an edit, adjusted for the one legitimate
            // difference: exactly one bucket may appear, and nothing else may
            // change. Reusing the edit oracle unchanged would reject every add.
            let appeared: Vec<&String> = after.keys().filter(|k| !before.contains_key(*k)).collect();
            let changed: Vec<&String> = before
                .iter()
                .filter(|(k, v)| after.get(*k) != Some(v))
                .map(|(k, _)| k)
                .collect();
            if appeared.len() != 1 || !changed.is_empty() {
                return (
                    format!(
                        "adding {label} was not clean: {} bucket(s) appeared and {} existing bucket(s) \
                         changed. Write only the body expression and do not close a brace you did not open.",
                        appeared.len(),
                        changed.len()
                    ),
                    true,
                );
            }

            // Every test, not just this bucket's — a new bucket has none of its own.
            let mut out = Vec::new();
            for tid in &recompiled.registry.test_ids {
                let tb = recompiled.registry.get(tid).unwrap();
                let r = eval_bucket(&recompiled.registry, tid, &[], &mut out);
                let ok = if tb.expect_error { r.is_err() } else { r.is_ok() };
                if !ok {
                    return (
                        format!(
                            "adding {label} broke an existing test ({}）",
                            tb.subject.clone().unwrap_or_else(|| tid.clone())
                        ),
                        true,
                    );
                }
            }

            let addr = appeared[0].clone();
            ws.source = candidate;
            ws.dirty = true;
            (
                format!("added {label} as {addr}. All {} test(s) still pass.", recompiled.registry.test_ids.len()),
                false,
            )
        }

        "bucket_graph" => {
            let g = bucketlang::graph::build_graph(reg);
            let only = call.input.get("address").and_then(|v| v.as_str());
            let mut out = String::new();
            for (addr, node) in &g {
                if let Some(a) = only {
                    let want = reg.resolve_target(a.trim());
                    if want.as_deref() != Some(addr.as_str()) {
                        continue;
                    }
                }
                let Some(b) = reg.get(addr) else { continue };
                if b.kind != BucketKind::User {
                    continue;
                }
                let name = b.label.clone().unwrap_or_else(|| addr.clone());
                let calls: Vec<String> = node
                    .out
                    .iter()
                    .filter_map(|c| reg.get(c).map(|cb| cb.label.clone().unwrap_or_else(|| c.clone())))
                    .collect();
                let called_by: Vec<String> = node
                    .inn
                    .iter()
                    .filter_map(|c| reg.get(c).map(|cb| cb.label.clone().unwrap_or_else(|| c.clone())))
                    .collect();
                out.push_str(&format!(
                    "{name} ({addr})\n  calls    -> {}\n  called by <- {}\n",
                    if calls.is_empty() { "-".into() } else { calls.join(", ") },
                    if called_by.is_empty() { "-".into() } else { called_by.join(", ") }
                ));
            }
            if out.is_empty() {
                out.push_str("no matching buckets\n");
            }
            (out, false)
        }

        "read_file" => {
            let path = call.input.get("path").and_then(|v| v.as_str()).unwrap_or("");
            match ws.sandbox.read_file(path) {
                Ok(t) => (t, false),
                Err(e) => (e, true),
            }
        }

        "write_file" => {
            let path = call.input.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let contents = call.input.get("contents").and_then(|v| v.as_str()).unwrap_or("");
            match ws.sandbox.write_file(path, contents) {
                Ok(t) => (t, false),
                Err(e) => (e, true),
            }
        }

        "run_command" => {
            let argv: Vec<String> = call
                .input
                .get("argv")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            if argv.is_empty() {
                return ("argv must be a non-empty array".into(), true);
            }
            match ws.sandbox.run(&argv) {
                Ok(t) => (t, false),
                Err(e) => (e, true),
            }
        }

        other => (format!("no such tool: {other}"), true),
    }
}

/// Rewrite a bucket's `"description"` string in the source.
///
/// The description sits between the contract's `->` type and the body's `{`, so
/// it is found by scanning from the bucket's body span backwards to the string
/// literal that precedes it — never by searching for the old text, which could
/// match anywhere.
fn replace_description(
    source: &str,
    reg: &Registry,
    target: &str,
    new_desc: &str,
) -> Option<String> {
    let addr = reg.resolve_target(target.trim())?;
    let span = reg.get(&addr)?.body_span?;
    // Walk back from the opening brace to the closing quote of the desc string.
    let head = &source[..span.start.saturating_sub(1)];
    let close = head.rfind('"')?;
    let open = head[..close].rfind('"')?;
    let mut out = String::with_capacity(source.len());
    out.push_str(&source[..open + 1]);
    out.push_str(&new_desc.replace('"', "'"));
    out.push_str(&source[close..]);
    Some(out)
}

/// Run the agent until it stops calling tools, or a limit stops it.
#[allow(clippy::too_many_arguments)]
pub fn run(
    client: &Client,
    mut guard: Guard,
    file: &Path,
    source: &str,
    task: &str,
    max_turns: usize,
    approval: crate::sandbox::ApprovalMode,
) -> Result<AgentRun, String> {
    let root = file.parent().unwrap_or(Path::new("."));
    let mut ws = Workspace {
        source: source.to_string(),
        file: file.to_path_buf(),
        dirty: false,
        sandbox: crate::sandbox::Sandbox::new(root)?.with_approval(approval),
    };
    let whole_file_tokens = crate::metrics::estimate_tokens(source);
    let tool_defs = tools();
    // Name the file. Without it the model guessed, and repeatedly called
    // read_file(".") — which is a directory.
    let mut messages = vec![Message::user(format!(
        "{task}\n\n(The program is the file `{}`. Prefer the bucket tools; \
         read_file is a last resort.)",
        file.file_name().and_then(|f| f.to_str()).unwrap_or("program.bkt")
    ))];
    let mut records: Vec<ToolRecord> = Vec::new();
    let mut summary = String::new();
    let mut turns = 0usize;

    for turn in 1..=max_turns {
        turns = turn;

        let projected = crate::metrics::estimate_tokens(SYSTEM)
            + messages
                .iter()
                .map(|m| crate::metrics::estimate_tokens(&m.content.to_string()))
                .sum::<usize>();
        if let Err(limit) = guard.check_before(projected as u32, 4_000) {
            eprintln!(
                "stopping before turn {turn}: would exceed the {limit} limit ({})",
                guard.spend().summary()
            );
            break;
        }

        let completion = match client.complete(SYSTEM, &messages, Some(&tool_defs), None) {
            Ok(c) => c,
            Err(LlmError::Refused {
                category,
                explanation,
            }) => {
                return Err(format!(
                    "the model declined this request{}{}",
                    category.map(|c| format!(" ({c})")).unwrap_or_default(),
                    explanation.map(|e| format!(": {e}")).unwrap_or_default()
                ))
            }
            Err(e) => return Err(e.to_string()),
        };

        let over = guard.record(&completion.usage);
        eprintln!(
            "turn {turn}: {} in / {} out tok  |  {}",
            completion.usage.input_tokens,
            completion.usage.output_tokens,
            guard.spend().summary()
        );
        if !completion.text.trim().is_empty() {
            summary = completion.text.trim().to_string();
            eprintln!("  {}", summary.replace('\n', "\n  "));
        }

        if !completion.wants_tools() {
            break;
        }

        messages.push(Message::assistant(completion.raw_content.clone()));

        // Execute every call from this turn, and return all results together —
        // splitting them across messages trains the model out of parallel calls.
        let mut results = Vec::new();
        for call in &completion.tool_calls {
            let (out, is_error) = dispatch(&mut ws, call);
            eprintln!(
                "  {} {} -> {}",
                if is_error { "✗" } else { "✓" },
                call.name,
                out.lines().next().unwrap_or("").chars().take(88).collect::<String>()
            );
            records.push(ToolRecord {
                turn,
                name: call.name.clone(),
                input: call.input.clone(),
                ok: !is_error,
                result: out.chars().take(2_000).collect(),
            });
            results.push(call.result(out, is_error));
        }
        messages.push(Message::tool_results(results));

        if over.is_err() {
            eprintln!("stopping: spend limit reached ({})", guard.spend().summary());
            break;
        }
    }

    Ok(AgentRun {
        ok: true,
        edited: ws.dirty,
        turns,
        tool_calls: records,
        summary,
        source: if ws.dirty { Some(ws.source) } else { None },
        spend: guard.spend().clone(),
        whole_file_tokens,
    })
}

// ============================================================================
// Baseline arm: the conventional approach, for comparison.
// ============================================================================

/// What a normal coding agent does: paste the file, ask for the edited file back.
///
/// This is the control. It is put through the *same* gates as the bucket path —
/// it must compile and every test must still pass — so a difference in outcome
/// is attributable to the context strategy rather than to one arm being graded
/// more leniently.
///
/// It cannot use the atomicity gate, and that asymmetry is the point: when the
/// model returns a whole file there is no way to tell an intended edit from an
/// unintended one. Any bucket may have changed, and nothing can prove otherwise.
pub fn run_baseline(
    client: &Client,
    mut guard: Guard,
    file: &Path,
    source: &str,
    task: &str,
) -> Result<BaselineRun, String> {
    const SYSTEM_BASELINE: &str = r#"You are editing a bucketlang program.

A bucket is a small typed function: `label(x: Num) -> Num "description" { body }`.
`@test label(input) == expected` lines pin behaviour. `@entry` marks the entry point.

Syntax: arithmetic `+ - * / **` (`+` also joins two Str), comparison `== != < > <= >=`,
logic `&& || !`, `if c then a else b`, `match x { None => …, Some(v) => … }`,
calls by label, pipe `x |> f`, lists `[1,2]`, records `{x: 1}`, field access `p.x`,
bindings `name = expr`, cores: print, error, pow, mod, floor, abs, to_json, from_json,
list_len, list_nth, list_append, list_concat, list_remove.

You will be given a whole file. Return the COMPLETE edited file and nothing else —
no explanation, no markdown fences. Keep every bucket you were not asked to change
exactly as it was."#;

    let user = format!("{task}\n\n--- file ---\n{source}");
    let projected = crate::metrics::estimate_tokens(&user)
        + crate::metrics::estimate_tokens(SYSTEM_BASELINE);
    guard
        .check_before(projected as u32, 8_000)
        .map_err(|l| format!("baseline would exceed the {l} limit before running"))?;

    let completion = client
        .complete(SYSTEM_BASELINE, &[Message::user(user)], None, None)
        .map_err(|e| e.to_string())?;
    let _ = guard.record(&completion.usage);

    // Models fence code even when told not to; strip it rather than fail the arm
    // on a formatting habit.
    let mut text = completion.text.trim().to_string();
    if text.starts_with("```") {
        text = text
            .lines()
            .skip(1)
            .take_while(|l| !l.trim_start().starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n");
    }

    // Same gates the bucket path faces, minus atomicity, which is unavailable here.
    let mut compiles = false;
    let mut tests_pass = false;
    let mut error = None;
    match compile_with_base(&text, opts(), file) {
        Ok(c) => {
            compiles = true;
            let mut out = Vec::new();
            let mut all = true;
            for tid in &c.registry.test_ids {
                let b = c.registry.get(tid).unwrap();
                let r = eval_bucket(&c.registry, tid, &[], &mut out);
                let ok = if b.expect_error { r.is_err() } else { r.is_ok() };
                if !ok {
                    all = false;
                }
            }
            tests_pass = all;
            if !all {
                error = Some("a test failed after the edit".to_string());
            }
        }
        Err(e) => error = Some(e.with_source(&text, None).render(Some(&text))),
    }

    // How much of the file came back rewritten, whether or not it needed to.
    let changed_lines = text
        .lines()
        .zip(source.lines())
        .filter(|(a, b)| a != b)
        .count()
        + text.lines().count().abs_diff(source.lines().count());

    Ok(BaselineRun {
        ok: compiles && tests_pass,
        edited: compiles && tests_pass && text.trim() != source.trim(),
        compiles,
        tests_pass,
        error,
        source: if compiles && tests_pass { Some(text) } else { None },
        spend: guard.spend().clone(),
        changed_lines,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct BaselineRun {
    pub ok: bool,
    /// Whether the returned file actually differs from the original.
    pub edited: bool,
    pub compiles: bool,
    pub tests_pass: bool,
    pub error: Option<String>,
    pub source: Option<String>,
    pub spend: crate::budget::Spend,
    /// Lines that differ from the original — the blast radius of a whole-file rewrite.
    pub changed_lines: usize,
}

// ---------------------------------------------------------------------------
// Source-level surgery on a whole bucket declaration.
// ---------------------------------------------------------------------------

/// Byte range covering a bucket's entire declaration: any `@test` / `@entry`
/// lines above it, the signature line, and the body through its closing brace.
///
/// The language records a span for the *body* only, which is all a body edit
/// needs. Renaming, changing a contract, or deleting requires the whole
/// declaration, and it is derived here rather than by searching for the label —
/// searching text is exactly the bug the span work was done to remove.
fn decl_range(source: &str, reg: &Registry, target: &str) -> Option<(usize, usize)> {
    let addr = reg.resolve_target(target.trim())?;
    let span = reg.get(&addr)?.body_span?;

    // End: the closing brace after the body.
    let end = source[span.end..]
        .find('}')
        .map(|i| span.end + i + 1)
        .unwrap_or(span.end);

    // Start: walk back to the beginning of the signature line, then keep
    // walking while the lines above are annotations belonging to this bucket.
    let mut start = source[..span.start].rfind('\n').map(|i| i + 1).unwrap_or(0);
    loop {
        let prev_end = match start.checked_sub(1) {
            Some(0) | None => break,
            Some(i) => i,
        };
        let prev_start = source[..prev_end].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line = source[prev_start..prev_end].trim();
        if line.starts_with("@test") || line.starts_with("@entry") {
            start = prev_start;
        } else {
            break;
        }
    }
    Some((start, end))
}

/// The `@test` / `@entry` annotation lines attached to a bucket.
fn decl_annotations(source: &str, reg: &Registry, target: &str) -> Vec<String> {
    let Some((start, _)) = decl_range(source, reg, target) else {
        return Vec::new();
    };
    source[start..]
        .lines()
        .take_while(|l| {
            let t = l.trim();
            t.starts_with("@test") || t.starts_with("@entry")
        })
        .map(|l| l.to_string())
        .collect()
}

/// Render `@test` lines from the tool's test spec.
fn render_tests(tests: &serde_json::Value) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    for t in tests.as_array().cloned().unwrap_or_default() {
        let c = t.get("call").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if c.is_empty() {
            continue;
        }
        if t.get("expect_error").and_then(|v| v.as_bool()) == Some(true) {
            lines.push(format!("@test_error {c}"));
        } else {
            let e = t.get("expects").and_then(|v| v.as_str()).unwrap_or("").trim();
            if e.is_empty() {
                return Err(format!("test `{c}` needs `expects`, or `expect_error: true`"));
            }
            lines.push(format!("@test {c} == {e}"));
        }
    }
    Ok(lines)
}

/// Replace a whole declaration, or delete it when `replacement` is `None`.
fn splice_decl(
    source: &str,
    reg: &Registry,
    target: &str,
    replacement: Option<&str>,
) -> Result<String, String> {
    let (start, end) = decl_range(source, reg, target)
        .ok_or_else(|| format!("cannot locate the declaration of {target}"))?;
    let mut out = String::with_capacity(source.len());
    out.push_str(&source[..start]);
    match replacement {
        Some(text) => out.push_str(text),
        None => {
            // Also swallow the blank line the declaration left behind.
            let rest = source[end..].trim_start_matches('\n');
            out.push_str(rest);
            return Ok(out);
        }
    }
    out.push_str(&source[end..]);
    Ok(out)
}

/// Compile a candidate source and check what changed against `before`.
///
/// `allow_appear` / `allow_vanish` describe the change the caller intends; any
/// other movement is collateral and rejects. This generalises the single-bucket
/// atomicity oracle so add, delete, and rename each get a gate shaped to them
/// rather than sharing one that fits none of them.
fn check_change(
    ws: &Workspace,
    candidate: &str,
    before: &std::collections::BTreeMap<String, String>,
    allow_appear: usize,
    allow_vanish: usize,
    allow_change: &[String],
) -> Result<bucketlang::CompileResult, String> {
    let recompiled = compile_with_base(candidate, opts(), Path::new(&ws.file))
        .map_err(|e| e.with_source(candidate, None).render(Some(candidate)))?;
    let after = user_hashes(&recompiled.registry);

    let appeared = after.keys().filter(|k| !before.contains_key(*k)).count();
    let vanished = before.keys().filter(|k| !after.contains_key(*k)).count();
    let changed: Vec<&String> = before
        .iter()
        .filter(|(k, v)| after.get(*k).is_some_and(|n| n != *v))
        .map(|(k, _)| k)
        .filter(|k| !allow_change.contains(k))
        .collect();

    if appeared != allow_appear || vanished != allow_vanish || !changed.is_empty() {
        return Err(format!(
            "that was not a clean change: {appeared} bucket(s) appeared (expected {allow_appear}), \
             {vanished} vanished (expected {allow_vanish}), and {} other bucket(s) changed. \
             Write only what was asked and do not close a brace you did not open.",
            changed.len()
        ));
    }

    // Every test must still pass — a structural change can break a distant one.
    let mut out = Vec::new();
    for tid in &recompiled.registry.test_ids {
        let tb = recompiled.registry.get(tid).unwrap();
        let r = eval_bucket(&recompiled.registry, tid, &[], &mut out);
        let ok = if tb.expect_error { r.is_err() } else { r.is_ok() };
        if !ok {
            let subject = tb.subject.clone().unwrap_or_else(|| tid.clone());
            // Say what to do about it. A bare "a test broke" leaves the model
            // guessing, and in practice it gave up rather than reaching for the
            // tool that fixes exactly this.
            return Err(format!(
                "that change broke the test on {subject}: `{}`.\n\n\
                 If the behaviour was *meant* to change, make the body change and the test change \
                 together: pass a `tests` array to patch_bucket in the same call. Doing it in two \
                 steps cannot work -- new tests fail against the old body, and the old test fails \
                 against the new body. If the behaviour was not meant to change, fix the body instead.",
                tb.desc
            ));
        }
    }
    Ok(recompiled)
}
