//! The verified edit loop.
//!
//! One turn is: retrieve a neighbourhood, ask for a new body for one bucket,
//! then put that body through three deterministic gates before anything is
//! written. If a gate rejects, the *diagnostic* goes back to the agent and it
//! tries again — which is why the diagnostics carry a bucket address, a span,
//! and expected/found rather than a sentence.
//!
//! The gates are what make an unattended loop safe:
//!
//! 1. **compile** — the spliced program parses and typechecks
//! 2. **atomicity** — exactly the target bucket changed, nothing else
//! 3. **behaviour** — the target's `@test` cases still pass
//!
//! Gate 2 is easy to overlook and is the one that catches a body whose braces
//! close early, quietly redefining a neighbour.

use bucketlang::compile::{compile, compile_with_base, BuildProfile, CompileOptions};
use bucketlang::edit::{only_target_changed, run_subject_tests, splice_bucket_body, user_hashes};
use bucketlang::error::Diagnostic;
use serde::Serialize;
use std::path::{Path, PathBuf};

use crate::context::{build_context, ContextPack};
use crate::metrics::{estimate_tokens, Metrics};

/// What the agent is given for one attempt.
#[derive(Debug, Clone, Serialize)]
pub struct EditRequest {
    /// What the user asked for, in their words.
    pub goal: String,
    /// Address of the bucket being edited.
    pub target: String,
    /// The neighbourhood: target body, neighbour signatures and descriptions.
    pub context: ContextPack,
    /// 1 for the first try.
    pub attempt: usize,
    /// Why the previous attempt was rejected. Empty on the first try.
    pub feedback: Vec<Diagnostic>,
}

impl EditRequest {
    /// The request as the text an agent would actually receive. Also what the
    /// token metrics are measured against, so the number reflects what was sent.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("goal: {}\n\n", self.goal));
        s.push_str(&self.context.render());
        if !self.feedback.is_empty() {
            s.push_str("\nprevious attempt was rejected:\n");
            for d in &self.feedback {
                s.push_str(&format!("  {}\n", d.render(None).replace('\n', "\n  ")));
            }
        }
        s
    }
}

/// Whatever proposes a new body: a language model, a script, a human.
///
/// Deliberately narrow. The agent sees a request and returns a body; it never
/// touches the file, so it cannot skip the gates.
pub trait Agent {
    fn name(&self) -> &str;
    fn propose(&mut self, req: &EditRequest) -> Result<String, String>;
}

/// An agent that replays a fixed list of bodies.
///
/// Lets the whole loop — including the retry path — be tested without a network
/// call, which is what keeps the gates themselves under test.
pub struct ScriptedAgent {
    bodies: Vec<String>,
    next: usize,
}

impl ScriptedAgent {
    pub fn new(bodies: impl IntoIterator<Item = impl Into<String>>) -> Self {
        ScriptedAgent {
            bodies: bodies.into_iter().map(Into::into).collect(),
            next: 0,
        }
    }
}

impl Agent for ScriptedAgent {
    fn name(&self) -> &str {
        "scripted"
    }

    fn propose(&mut self, _req: &EditRequest) -> Result<String, String> {
        let b = self
            .bodies
            .get(self.next)
            .ok_or_else(|| "scripted agent ran out of proposals".to_string())?
            .clone();
        self.next += 1;
        Ok(b)
    }
}

/// Which gate rejected an attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Gate {
    Splice,
    Compile,
    Atomicity,
    Behaviour,
}

#[derive(Debug, Clone, Serialize)]
pub struct Attempt {
    pub n: usize,
    pub body: String,
    /// `None` when every gate passed.
    pub rejected_by: Option<Gate>,
    pub diagnostics: Vec<Diagnostic>,
    pub tests_run: usize,
    pub tests_passed: usize,
    pub request_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EditOutcome {
    pub ok: bool,
    pub target: String,
    pub attempts: Vec<Attempt>,
    /// The accepted source. `None` if every attempt was rejected.
    pub source: Option<String>,
    pub metrics: Metrics,
}

#[derive(Debug, Clone)]
pub struct LoopOptions {
    /// How many proposals before giving up.
    pub max_attempts: usize,
    /// How far to walk the call graph when building context.
    pub depth: usize,
    pub strict: bool,
    /// Resolves imports relative to this file. `None` compiles as a bare string.
    pub base_file: Option<PathBuf>,
}

impl Default for LoopOptions {
    fn default() -> Self {
        LoopOptions {
            max_attempts: 3,
            depth: 1,
            strict: true,
            base_file: None,
        }
    }
}

fn opts(strict: bool) -> CompileOptions {
    CompileOptions {
        strict,
        profile: BuildProfile::Dev,
    }
}

fn build(source: &str, o: &LoopOptions) -> bucketlang::error::Result<bucketlang::CompileResult> {
    match &o.base_file {
        Some(p) => compile_with_base(source, opts(o.strict), Path::new(p)),
        None => compile(source, opts(o.strict)),
    }
}

/// Run the loop until the target bucket is edited and every gate passes, or the
/// attempt budget runs out. Nothing is written to disk — the accepted source is
/// returned, and the caller decides.
pub fn run_edit(
    source: &str,
    target: &str,
    goal: &str,
    agent: &mut dyn Agent,
    o: &LoopOptions,
) -> Result<EditOutcome, String> {
    let base = build(source, o).map_err(|e| {
        format!(
            "source does not compile before edit: {}",
            e.with_source(source, None).render(Some(source))
        )
    })?;
    let before = user_hashes(&base.registry);
    let target_addr = base
        .registry
        .resolve_target(target.trim())
        .ok_or_else(|| format!("unknown bucket {target}"))?;

    let context = build_context(&base.registry, target, o.depth).map_err(|e| e.to_string())?;

    let mut attempts: Vec<Attempt> = Vec::new();
    let mut feedback: Vec<Diagnostic> = Vec::new();
    let mut retry_tokens = 0usize;
    let mut first_request_text = String::new();

    for n in 1..=o.max_attempts {
        let req = EditRequest {
            goal: goal.to_string(),
            target: target_addr.clone(),
            context: context.clone(),
            attempt: n,
            feedback: feedback.clone(),
        };
        let request_text = req.render();
        let request_tokens = estimate_tokens(&request_text);
        if n == 1 {
            first_request_text = request_text.clone();
        } else {
            retry_tokens += request_tokens;
        }

        let body = agent.propose(&req)?;
        let mut attempt = Attempt {
            n,
            body: body.clone(),
            rejected_by: None,
            diagnostics: Vec::new(),
            tests_run: 0,
            tests_passed: 0,
            request_tokens,
        };

        // Gate 0: the splice itself must land on the target.
        let spliced = match splice_bucket_body(source, &base.registry, target, &body) {
            Ok(s) => s,
            Err(e) => {
                attempt.rejected_by = Some(Gate::Splice);
                attempt.diagnostics = vec![e.into_diagnostic()];
                feedback = attempt.diagnostics.clone();
                attempts.push(attempt);
                continue;
            }
        };

        // Gate 1: it has to compile.
        let compiled = match build(&spliced, o) {
            Ok(c) => c,
            Err(e) => {
                attempt.rejected_by = Some(Gate::Compile);
                attempt.diagnostics = vec![e.with_source(&spliced, None).into_diagnostic()];
                feedback = attempt.diagnostics.clone();
                attempts.push(attempt);
                continue;
            }
        };

        // Gate 2: exactly one bucket changed.
        if let Err(msg) = only_target_changed(&before, &user_hashes(&compiled.registry), &target_addr)
        {
            attempt.rejected_by = Some(Gate::Atomicity);
            attempt.diagnostics = vec![Diagnostic::new(
                bucketlang::error::Stage::Link,
                "not_atomic",
                msg,
            )
            .in_bucket(target_addr.clone())
            .hint("the body must not close its own brace; write only the expression")];
            feedback = attempt.diagnostics.clone();
            attempts.push(attempt);
            continue;
        }

        // Gate 3: the bucket's own tests still pass.
        let mut sink = std::io::sink();
        match run_subject_tests(&compiled.registry, target, &mut sink) {
            Ok((run, passed)) => {
                attempt.tests_run = run;
                attempt.tests_passed = passed;
                attempts.push(attempt);
                let metrics = Metrics::new(source, &first_request_text, n, retry_tokens);
                return Ok(EditOutcome {
                    ok: true,
                    target: target_addr,
                    attempts,
                    source: Some(spliced),
                    metrics,
                });
            }
            Err(e) => {
                attempt.rejected_by = Some(Gate::Behaviour);
                attempt.diagnostics = vec![e.with_source(&spliced, None).into_diagnostic()];
                feedback = attempt.diagnostics.clone();
                attempts.push(attempt);
            }
        }
    }

    let n = attempts.len();
    let metrics = Metrics::new(source, &first_request_text, n, retry_tokens);
    Ok(EditOutcome {
        ok: false,
        target: target_addr,
        attempts,
        source: None,
        metrics,
    })
}
