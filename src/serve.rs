//! A local web UI.
//!
//! Hand-rolled on `TcpListener` rather than pulling in a web framework: the
//! surface is five routes on localhost, and a framework would be a larger
//! dependency than the thing it serves.
//!
//! Everything except the agent works **without an API key** — the program map,
//! English search, the call graph, and the whole-file-versus-subgraph
//! comparison are all computed locally. That is deliberate: the claim this
//! project makes is about context size, and you should be able to see it
//! measured before spending anything to test it.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bucketlang::ast::BucketKind;
use bucketlang::compile::{compile_file, BuildProfile, CompileOptions};
use serde_json::json;

use crate::index::search;
use crate::metrics::estimate_tokens;
use crate::strategy::{self, Strategy};

fn opts() -> CompileOptions {
    CompileOptions {
        strict: true,
        profile: BuildProfile::Dev,
    }
}

/// Which file the UI is currently editing. Shared because uploading swaps it,
/// and every later request must see the swap.
type Current = Arc<Mutex<PathBuf>>;

pub fn serve(file: PathBuf, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("cannot bind port {port}: {e}"))?;
    println!("bucketcode ui  →  http://127.0.0.1:{port}");
    println!("editing        →  {}", file.display());
    println!("api key        →  {}", if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        "present (agent enabled)"
    } else {
        "not set (everything except the agent still works)"
    });

    let current: Current = Arc::new(Mutex::new(file.clone()));
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let f = Arc::clone(&current);
                std::thread::spawn(move || {
                    if let Err(e) = handle(s, f) {
                        eprintln!("request failed: {e}");
                    }
                });
            }
            Err(e) => eprintln!("connection failed: {e}"),
        }
    }
    Ok(())
}

fn handle(mut stream: TcpStream, current: Current) -> Result<(), String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();

    // Headers, so we can find the body length.
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        reader.read_line(&mut h).map_err(|e| e.to_string())?;
        if h.trim().is_empty() {
            break;
        }
        if let Some(v) = h.to_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    }
    let body = String::from_utf8_lossy(&body).to_string();

    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target.clone(), String::new()),
    };

    let file = current.lock().map_err(|_| "lock poisoned")?.clone();
    let (status, ctype, payload) = route(&method, &path, &query, &body, &file, &current);
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\n\r\n",
        payload.len()
    );
    stream.write_all(resp.as_bytes()).map_err(|e| e.to_string())?;
    stream.write_all(payload.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

fn param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        if k == key {
            Some(urldecode(v))
        } else {
            None
        }
    })
}

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("20");
                out.push(u8::from_str_radix(hex, 16).unwrap_or(b' '));
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn json_ok(v: serde_json::Value) -> (&'static str, &'static str, String) {
    ("200 OK", "application/json", v.to_string())
}

fn json_err(msg: String) -> (&'static str, &'static str, String) {
    (
        "200 OK",
        "application/json",
        json!({ "error": msg }).to_string(),
    )
}

fn route(
    method: &str,
    path: &str,
    query: &str,
    body: &str,
    file: &PathBuf,
    current: &Current,
) -> (&'static str, &'static str, String) {
    match (method, path) {
        ("GET", "/") => ("200 OK", "text/html; charset=utf-8", INDEX_HTML.to_string()),

        // The program map: what the model sees first, and what it costs.
        // Dedicated status route. The agent route cannot answer this: an empty
        // probe trips its goal check before it ever looks for a key, which
        // reported "key present" on a machine that had none.
        ("GET", "/api/status") => json_ok(json!({
            "has_key": std::env::var("ANTHROPIC_API_KEY").is_ok(),
            "model": crate::llm::DEFAULT_MODEL,
        })),

        ("GET", "/api/program") => {
            let src = match std::fs::read_to_string(file) {
                Ok(s) => s,
                Err(e) => return json_err(e.to_string()),
            };
            let compiled = match compile_file(file, opts()) {
                Ok(c) => c,
                Err(e) => {
                    return json_err(e.with_source(&src, None).render(Some(&src)));
                }
            };
            let reg = &compiled.registry;
            let graph = bucketlang::graph::build_graph(reg);

            let mut buckets = Vec::new();
            let mut map_text = String::new();
            for (addr, b) in &reg.buckets {
                if b.kind != BucketKind::User {
                    continue;
                }
                let contract = format!(
                    "({}) -> {}",
                    b.contract
                        .params
                        .iter()
                        .map(|p| format!("{}: {}", p.name, p.ty.name()))
                        .collect::<Vec<_>>()
                        .join(", "),
                    b.contract.ret.name()
                );
                let node = graph.get(addr);
                let calls: Vec<String> = node
                    .map(|n| {
                        n.out
                            .iter()
                            .filter_map(|c| {
                                reg.get(c)
                                    .filter(|cb| cb.kind == BucketKind::User)
                                    .map(|cb| cb.label.clone().unwrap_or_else(|| c.clone()))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                map_text.push_str(&format!(
                    "{addr} {}{contract} \"{}\"\n",
                    b.label.clone().unwrap_or_default(),
                    b.desc
                ));
                let called_by: Vec<String> = node
                    .map(|n| {
                        n.inn
                            .iter()
                            .filter_map(|c| {
                                reg.get(c)
                                    .filter(|cb| cb.kind == BucketKind::User)
                                    .map(|cb| cb.label.clone().unwrap_or_else(|| c.clone()))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let tests: Vec<String> = reg
                    .test_ids
                    .iter()
                    .filter(|t| {
                        reg.get(t).and_then(|tb| tb.subject.clone()).as_deref()
                            == b.label.as_deref()
                    })
                    .map(|t| {
                        reg.get(t)
                            .map(|tb| tb.desc.clone())
                            .unwrap_or_default()
                    })
                    .collect();
                buckets.push(json!({
                    "address": addr,
                    "label": b.label,
                    "contract": contract,
                    "desc": b.desc,
                    "calls": calls,
                    "called_by": called_by,
                    "tests": tests,
                    "complexity": b.complexity.nodes,
                    "body": bucketlang::render::render_labelled(&b.body, reg),
                    "desc_tokens": estimate_tokens(&b.desc),
                    "body_tokens": estimate_tokens(&bucketlang::render::render_labelled(&b.body, reg)),
                }));
            }

            json_ok(json!({
                "file": file.display().to_string(),
                "source": src,
                "buckets": buckets,
                "tests": reg.test_ids.len(),
                // The headline number: the map costs this, the file costs that.
                "map_tokens": estimate_tokens(&map_text),
                "file_tokens": estimate_tokens(&src),
            }))
        }

        // English in, ranked buckets out.
        ("GET", "/api/find") => {
            let q = param(query, "q").unwrap_or_default();
            let compiled = match compile_file(file, opts()) {
                Ok(c) => c,
                Err(e) => return json_err(e.to_string()),
            };
            let matches = search(&compiled.registry, &q, 8);
            json_ok(json!({ "query": q, "matches": matches }))
        }

        // The comparison that makes the claim testable.
        ("GET", "/api/context") => {
            let bucket = param(query, "bucket").unwrap_or_default();
            let src = match std::fs::read_to_string(file) {
                Ok(s) => s,
                Err(e) => return json_err(e.to_string()),
            };
            let compiled = match compile_file(file, opts()) {
                Ok(c) => c,
                Err(e) => return json_err(e.to_string()),
            };
            let mut arms = Vec::new();
            for s in [Strategy::WholeFile, Strategy::Subgraph] {
                match strategy::build(s, &compiled.registry, &src, &bucket, 1) {
                    Ok(r) => arms.push(json!({
                        "strategy": r.strategy.as_str(),
                        "tokens": r.tokens,
                        "text": r.text,
                    })),
                    Err(e) => arms.push(json!({
                        "strategy": s.as_str(),
                        "error": e,
                    })),
                }
            }
            json_ok(json!({ "bucket": bucket, "arms": arms }))
        }

        // Run every test in the program.
        ("GET", "/api/tests") => {
            let src = match std::fs::read_to_string(file) {
                Ok(s) => s,
                Err(e) => return json_err(e.to_string()),
            };
            let compiled = match compile_file(file, opts()) {
                Ok(c) => c,
                Err(e) => return json_err(e.with_source(&src, None).render(Some(&src))),
            };
            let reg = &compiled.registry;
            let mut results = Vec::new();
            let mut out = Vec::new();
            for tid in &reg.test_ids {
                let b = reg.get(tid).unwrap();
                let r = bucketlang::eval::eval_bucket(reg, tid, &[], &mut out);
                let passed = if b.expect_error { r.is_err() } else { r.is_ok() };
                results.push(json!({
                    "id": tid,
                    "subject": b.subject,
                    "desc": b.desc,
                    "passed": passed,
                    "error": r.err().map(|e| e.with_source(&src, None).render(Some(&src))),
                }));
            }
            let passed = results.iter().filter(|r| r["passed"] == true).count();
            json_ok(json!({ "total": results.len(), "passed": passed, "results": results }))
        }

        // Deterministic context sweep — no API calls, so it is free to re-run.
        ("GET", "/api/bench") => {
            match crate::bench::sweep(&[2, 5, 10, 20, 40, 80, 160, 320]) {
                Ok(rows) => json_ok(json!({ "rows": rows })),
                Err(e) => json_err(e),
            }
        }

        // Hold size constant, vary fan-out: the variable subgraph cost responds to.
        ("GET", "/api/fanout") => {
            // Capped at 6. bucketlang's complexity budget allows 12 calls per
            // bucket and counts core calls too, so an infix `a + b + c` chain
            // costs a call per operator: fan-out f needs f + (f-1) calls. That
            // budget is also a real ceiling on how large any subgraph can get.
            match crate::bench::fanout_sweep(82, &[0, 1, 2, 4, 6]) {
                Ok(rows) => json_ok(json!({ "rows": rows })),
                Err(e) => json_err(e),
            }
        }

        // Per-bucket subgraph cost: shows that it tracks structure, not size.
        ("GET", "/api/subgraph_costs") => {
            let src = match std::fs::read_to_string(file) {
                Ok(s) => s,
                Err(e) => return json_err(e.to_string()),
            };
            let compiled = match compile_file(file, opts()) {
                Ok(c) => c,
                Err(e) => return json_err(e.to_string()),
            };
            let reg = &compiled.registry;
            let mut rows = Vec::new();
            for (addr, b) in &reg.buckets {
                if b.kind != BucketKind::User {
                    continue;
                }
                let sub = strategy::build(Strategy::Subgraph, reg, &src, addr, 1)
                    .map(|r| r.tokens)
                    .unwrap_or(0);
                let file_tok = estimate_tokens(&src);
                rows.push(json!({
                    "label": b.label.clone().unwrap_or_else(|| addr.clone()),
                    "subgraph": sub,
                    "file": file_tok,
                    "ratio": file_tok as f64 / sub.max(1) as f64,
                }));
            }
            json_ok(json!({ "rows": rows }))
        }

        // The agent. The only route that needs a key.
        ("POST", "/api/edit") => {
            let req: serde_json::Value = match serde_json::from_str(body) {
                Ok(v) => v,
                Err(e) => return json_err(e.to_string()),
            };
            let goal = req.get("goal").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if goal.trim().is_empty() {
                return json_err("say what you want changed".into());
            }
            let max_usd = req.get("max_usd").and_then(|v| v.as_f64()).unwrap_or(0.25);
            let max_requests = req.get("max_requests").and_then(|v| v.as_u64()).unwrap_or(6) as u32;
            let write = req.get("write").and_then(|v| v.as_bool()).unwrap_or(false);

            let cfg = match crate::llm::Config::from_env() {
                Ok(c) => c,
                Err(e) => {
                    return json_err(format!(
                        "{e} — export it in the shell you launched this from, then restart"
                    ))
                }
            };
            let src = match std::fs::read_to_string(file) {
                Ok(s) => s,
                Err(e) => return json_err(e.to_string()),
            };
            let budget = crate::budget::Budget {
                max_requests,
                max_input_tokens: 200_000,
                max_output_tokens: 32_000,
                max_usd,
            };
            let guard = crate::budget::Guard::new(budget, &cfg.model);
            let client = crate::llm::Client::new(cfg);

            match crate::agent::run(
                &client,
                guard,
                file,
                &src,
                &goal,
                12,
                // The UI has no approval prompt, so commands stay off here.
                crate::sandbox::ApprovalMode::Never,
            ) {
                Ok(run) => {
                    if run.ok && write {
                        if let Some(s) = &run.source {
                            if let Err(e) = std::fs::write(file, s) {
                                return json_err(e.to_string());
                            }
                        }
                    }
                    json_ok(serde_json::to_value(&run).unwrap_or(json!({})))
                }
                Err(e) => json_err(e),
            }
        }

        // Run one prompt through both arms and record everything.
        //
        // Written to disk as JSONL rather than kept in memory: an experiment
        // you cannot go back and re-read is not an experiment.
        ("POST", "/api/experiment") => {
            let req: serde_json::Value = match serde_json::from_str(body) {
                Ok(v) => v,
                Err(e) => return json_err(e.to_string()),
            };
            let goal = req.get("goal").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
            if goal.is_empty() {
                return json_err("say what you want changed".into());
            }
            let model = req
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or(crate::llm::DEFAULT_MODEL)
                .to_string();
            let max_usd = req.get("max_usd").and_then(|v| v.as_f64()).unwrap_or(0.30);
            let arms = req.get("arms").and_then(|v| v.as_str()).unwrap_or("both").to_string();
            let write = req.get("write").and_then(|v| v.as_bool()).unwrap_or(false);

            let src = match std::fs::read_to_string(file) {
                Ok(s) => s,
                Err(e) => return json_err(e.to_string()),
            };
            let budget = crate::budget::Budget {
                max_requests: 8,
                max_input_tokens: 200_000,
                max_output_tokens: 32_000,
                max_usd,
            };

            let mut record = json!({
                "at": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs()).unwrap_or(0),
                "goal": goal,
                "model": model,
                "file": file.display().to_string(),
                "file_tokens": estimate_tokens(&src),
            });

            if arms == "both" || arms == "buckets" {
                let cfg = match crate::llm::Config::from_env() {
                    Ok(c) => c.with_model(model.clone()).normalized(),
                    Err(e) => return json_err(e),
                };
                let client = crate::llm::Client::new(cfg);
                let guard = crate::budget::Guard::new(budget, &model);
                match crate::agent::run(
                    &client, guard, file, &src, &goal, 12,
                    crate::sandbox::ApprovalMode::Never,
                ) {
                    Ok(run) => {
                        if run.ok && write {
                            if let Some(s) = &run.source {
                                let _ = std::fs::write(file, s);
                            }
                        }
                        record["buckets"] = serde_json::to_value(&run).unwrap_or(json!({}));
                    }
                    Err(e) => record["buckets"] = json!({ "error": e }),
                }
            }

            if arms == "both" || arms == "file" {
                let cfg = match crate::llm::Config::from_env() {
                    Ok(c) => c.with_model(model.clone()).normalized(),
                    Err(e) => return json_err(e),
                };
                let client = crate::llm::Client::new(cfg);
                let guard = crate::budget::Guard::new(budget, &model);
                match crate::agent::run_baseline(&client, guard, file, &src, &goal) {
                    Ok(b) => record["whole_file"] = serde_json::to_value(&b).unwrap_or(json!({})),
                    Err(e) => record["whole_file"] = json!({ "error": e }),
                }
            }

            // Append to the log next to the file being edited.
            let log = file.with_extension("runs.jsonl");
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&log) {
                use std::io::Write as _;
                let _ = writeln!(f, "{record}");
            }
            record["log"] = json!(log.display().to_string());
            json_ok(record)
        }

        // Everything recorded so far, newest first.
        ("GET", "/api/runs") => {
            let log = file.with_extension("runs.jsonl");
            let text = std::fs::read_to_string(&log).unwrap_or_default();
            let mut rows: Vec<serde_json::Value> = text
                .lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect();
            rows.reverse();
            json_ok(json!({ "runs": rows, "log": log.display().to_string() }))
        }

        // Load a different program into the UI.
        //
        // The source is validated before it is accepted: a file that does not
        // compile would leave every other route erroring with no way back.
        // Written into a workspace directory rather than edited in place, so an
        // upload never touches the user's original.
        ("POST", "/api/open") => {
            let req: serde_json::Value = match serde_json::from_str(body) {
                Ok(v) => v,
                Err(e) => return json_err(e.to_string()),
            };
            let name = req
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("uploaded.bkt");
            let source = req.get("source").and_then(|v| v.as_str()).unwrap_or("");
            if source.trim().is_empty() {
                return json_err("that file is empty".into());
            }

            // Keep only the file name — an uploaded path must not decide where
            // anything lands on disk.
            let stem = std::path::Path::new(name)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("uploaded.bkt");
            let stem = if stem.ends_with(".bkt") || stem.ends_with(".bucket") {
                stem.to_string()
            } else {
                format!("{stem}.bkt")
            };

            let dir = file
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));
            let dest = dir.join(&stem);

            if let Err(e) = std::fs::write(&dest, source) {
                return json_err(format!("could not save: {e}"));
            }
            // Validate by compiling from its final location, so imports resolve
            // the same way they will for every later request.
            match compile_file(&dest, opts()) {
                Ok(c) => {
                    let n = c
                        .registry
                        .buckets
                        .values()
                        .filter(|b| b.kind == BucketKind::User)
                        .count();
                    if let Ok(mut cur) = current.lock() {
                        *cur = dest.clone();
                    }
                    json_ok(json!({
                        "file": dest.display().to_string(),
                        "buckets": n,
                        "tests": c.registry.test_ids.len(),
                    }))
                }
                Err(e) => {
                    let _ = std::fs::remove_file(&dest);
                    json_err(format!(
                        "that file does not compile, so it was not loaded:\n\n{}",
                        e.with_source(source, Some(&stem)).render(Some(source))
                    ))
                }
            }
        }

        _ => ("404 Not Found", "text/plain", "not found".into()),
    }
}

const INDEX_HTML: &str = include_str!("ui.html");
