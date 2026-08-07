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

You can change a bucket's body and description. You **cannot** add or remove
buckets, or change a `@test` annotation or a contract — there is no tool for
those. If a task needs one of them, say so plainly and stop; do not try to
reach it through `write_file`, which bypasses every check and will not help.

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
                Ok(pack) => (pack.render(), false),
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
