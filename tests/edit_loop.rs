//! The loop, including the paths that matter most: what happens when the agent
//! is wrong, and whether the failure that comes back is good enough to act on.

use bucketcode::{build_context, run_edit, LoopOptions, ScriptedAgent};
use bucketlang::compile::{compile, BuildProfile, CompileOptions};

const PROGRAM: &str = "\
@test double(3) == 6
double(x: Num) -> Num \"multiply by two\" {
  x * 2
}

add_one(x: Num) -> Num \"add one\" {
  x + 1
}

@entry
main() -> Num \"double then add one\" {
  add_one(double(5))
}
";

fn strict_dev() -> CompileOptions {
    CompileOptions {
        strict: true,
        profile: BuildProfile::Dev,
    }
}

#[test]
fn context_sends_the_target_body_but_only_neighbour_signatures() {
    let c = compile(PROGRAM, strict_dev()).unwrap();
    let pack = build_context(&c.registry, "main", 1).unwrap();
    let text = pack.render();

    // The target arrives whole.
    assert!(
        text.contains("add_one(double(5))"),
        "target body missing:\n{text}"
    );
    // Its callees arrive as description only — their bodies are the thing we are
    // deliberately not sending.
    assert!(text.contains("multiply by two"), "callee desc missing:\n{text}");
    assert!(
        !text.contains("x * 2"),
        "callee body leaked into context:\n{text}"
    );
}

/// Build a program with `n` independent helper buckets plus a target, so we can
/// watch the ratio move as the file grows around the bucket being edited.
fn program_with_helpers(n: usize) -> String {
    let mut s = String::new();
    s.push_str("@test double(3) == 6\ndouble(x: Num) -> Num \"multiply by two\" {\n  x * 2\n}\n\n");
    for i in 0..n {
        s.push_str(&format!(
            "helper_{i}(x: Num) -> Num \"helper number {i} that does some arithmetic\" {{\n  x + {i} * 2 - 1\n}}\n\n"
        ));
    }
    s.push_str("@entry\nmain() -> Num \"entry\" {\n  double(5)\n}\n");
    s
}

#[test]
fn context_does_not_beat_the_file_on_a_tiny_program() {
    // Recorded deliberately. The pack carries fixed scaffolding, so on a file
    // this small it is *larger* than just sending everything. Compression here is
    // amortised — claiming a win on an 11-line program would be measuring noise.
    let c = compile(PROGRAM, strict_dev()).unwrap();
    let pack = build_context(&c.registry, "double", 1).unwrap();
    let ctx = bucketcode::estimate_tokens(&pack.render());
    let whole = bucketcode::estimate_tokens(PROGRAM);
    assert!(
        ctx >= whole,
        "expected no win at this size (ctx {ctx}, whole {whole}) — \
         if this now passes, the pack got cheaper and the test should be updated"
    );
}

#[test]
fn context_wins_once_the_program_grows_around_the_target() {
    // The claim that actually matters: cost of editing one bucket is flat in the
    // size of the rest of the program, because the rest is never sent.
    let small = program_with_helpers(4);
    let large = program_with_helpers(40);

    let ctx_small = {
        let c = compile(&small, strict_dev()).unwrap();
        bucketcode::estimate_tokens(&build_context(&c.registry, "double", 1).unwrap().render())
    };
    let ctx_large = {
        let c = compile(&large, strict_dev()).unwrap();
        bucketcode::estimate_tokens(&build_context(&c.registry, "double", 1).unwrap().render())
    };
    let whole_large = bucketcode::estimate_tokens(&large);

    assert_eq!(
        ctx_small, ctx_large,
        "context for one bucket must not grow with unrelated buckets \
         ({ctx_small} -> {ctx_large})"
    );
    assert!(
        ctx_large * 4 < whole_large,
        "at 40 helpers the subgraph should be far cheaper \
         (ctx {ctx_large}, whole {whole_large})"
    );
}

#[test]
fn a_correct_edit_passes_every_gate_first_try() {
    let mut agent = ScriptedAgent::new(["x * 2"]);
    let out = run_edit(
        PROGRAM,
        "double",
        "keep doubling",
        &mut agent,
        &LoopOptions::default(),
    )
    .unwrap();

    assert!(out.ok, "should have succeeded: {:?}", out.attempts);
    assert_eq!(out.attempts.len(), 1);
    assert!(out.metrics.passed_first_try);
    assert!(out.source.is_some());
}

#[test]
fn a_type_error_is_rejected_and_fed_back_with_a_location() {
    // First proposal is a type error; second is correct. The loop must reject,
    // hand back something actionable, and then accept.
    let mut agent = ScriptedAgent::new(["\"not a number\"", "x * 2"]);
    let out = run_edit(
        PROGRAM,
        "double",
        "double the input",
        &mut agent,
        &LoopOptions::default(),
    )
    .unwrap();

    assert!(out.ok, "should recover on the second attempt");
    assert_eq!(out.attempts.len(), 2);

    let first = &out.attempts[0];
    assert_eq!(first.rejected_by, Some(bucketcode::loop_::Gate::Compile));
    let d = &first.diagnostics[0];
    assert!(d.line.is_some(), "feedback must carry a line");
    assert_eq!(
        d.bucket.as_deref(),
        Some("#b00000001"),
        "feedback must name the bucket to re-edit"
    );
    assert!(!out.metrics.passed_first_try);
    assert!(out.metrics.retry_tokens > 0, "retry cost should be counted");
}

#[test]
fn a_wrong_answer_is_caught_by_the_bucket_own_tests() {
    // Compiles fine, wrong behaviour. Only the @test case separates them, which
    // is the whole reason tests live next to the code.
    let mut agent = ScriptedAgent::new(["x * 3"]);
    let out = run_edit(
        PROGRAM,
        "double",
        "double the input",
        &mut agent,
        &LoopOptions {
            max_attempts: 1,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(!out.ok, "x * 3 must not pass double's tests");
    assert_eq!(
        out.attempts[0].rejected_by,
        Some(bucketcode::loop_::Gate::Behaviour)
    );
}

#[test]
fn brace_injection_is_rejected_by_the_atomicity_gate() {
    // A body that closes its own brace can define a neighbour. It compiles and
    // the target's tests still pass, so only the atomicity gate catches it.
    let mut agent = ScriptedAgent::new([
        "x * 2\n}\n\nsneaky(z: Num) -> Num \"injected\" {\n  z * 99",
    ]);
    let out = run_edit(
        PROGRAM,
        "double",
        "double the input",
        &mut agent,
        &LoopOptions {
            max_attempts: 1,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(!out.ok, "injection must be rejected");
    assert_eq!(
        out.attempts[0].rejected_by,
        Some(bucketcode::loop_::Gate::Atomicity)
    );
    assert!(out.source.is_none(), "nothing should be handed back to write");
}

#[test]
fn the_loop_gives_up_rather_than_writing_something_broken() {
    let mut agent = ScriptedAgent::new(["\"a\"", "\"b\"", "\"c\""]);
    let out = run_edit(
        PROGRAM,
        "double",
        "double the input",
        &mut agent,
        &LoopOptions {
            max_attempts: 3,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(!out.ok);
    assert_eq!(out.attempts.len(), 3);
    assert!(out.source.is_none());
}

#[test]
fn editing_an_unknown_bucket_fails_before_any_proposal() {
    let mut agent = ScriptedAgent::new(["x * 2"]);
    let err = run_edit(
        PROGRAM,
        "nonexistent",
        "whatever",
        &mut agent,
        &LoopOptions::default(),
    )
    .unwrap_err();
    assert!(err.contains("unknown bucket"), "got: {err}");
}
