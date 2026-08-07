//! Measuring the compression claim.
//!
//! Two costs are being compared and they scale differently, which is the whole
//! point:
//!
//! - **whole file** grows linearly with the program — every bucket you never
//!   touch is still paid for on every request
//! - **subgraph** tracks the edited bucket's *neighbourhood* — its body plus the
//!   signatures and descriptions of what it calls. Flat in program size, but
//!   **not** flat: it rises with fan-out, which is what `fanout_sweep` isolates
//!
//! There is also a third, easy to forget and large: the **agent's fixed
//! overhead**, the system prompt plus nine tool schemas, paid on every turn.
//! On a small program that overhead dwarfs both context strategies, which is
//! why a naive "look how few tokens the subgraph is" claim is misleading. The
//! sweep below reports all three so the crossover is a measurement rather than
//! an assertion.
//!
//! Nothing here calls the API — these are deterministic counts.

use bucketlang::compile::{compile, BuildProfile, CompileOptions};
use serde::Serialize;

use crate::metrics::estimate_tokens;
use crate::strategy::{self, Strategy};

#[derive(Debug, Clone, Serialize)]
pub struct Row {
    /// Number of user buckets in the generated program.
    pub buckets: usize,
    /// How many buckets the edited one calls — what drives subgraph size.
    pub fanout: usize,
    pub file_tokens: usize,
    pub subgraph_tokens: usize,
    /// System prompt + tool schemas, paid once per turn.
    pub agent_overhead: usize,
    /// `file / subgraph` — how much cheaper the neighbourhood is.
    pub context_ratio: f64,
    /// Whole file plus overhead vs subgraph plus overhead: what an agent
    /// actually pays per turn under each strategy.
    pub effective_ratio: f64,
}

/// Build a program of `n` helpers with a **real call graph**.
///
/// The first version of this generator emitted helpers that called nothing, so
/// the target's neighbourhood never grew and subgraph cost measured as flat at
/// every size. That was an artefact of the generator, not a property of the
/// design, and it made the headline claim look stronger than it is.
///
/// Here helpers form a chain and the target calls `fanout` of them, so the
/// subgraph carries real neighbours. Cost then tracks the neighbourhood — which
/// is the actual claim — while the file keeps growing with `n`.
fn program(n: usize, fanout: usize) -> String {
    let mut s = String::new();
    let fan = fanout.min(n);

    // The target calls `fan` helpers, so its neighbourhood is genuinely populated.
    let calls: Vec<String> = (0..fan).map(|i| format!("helper_{i}(p)")).collect();
    let body = if calls.is_empty() {
        "p * 2 * 1.08".to_string()
    } else {
        calls.join(" + ")
    };
    s.push_str(&format!(
        "price_total(p: Num) -> Num \"total price for two of an item including sales tax\" {{\n  {body}\n}}\n\n"
    ));

    // Helpers chain into each other so the graph has depth, not just width.
    for i in 0..n {
        let inner = if i + 1 < n && i < fan {
            format!("helper_{}(x) + {i}", i + 1)
        } else {
            format!("x + {i} * 2 - 1")
        };
        s.push_str(&format!(
            "helper_{i}(x: Num) -> Num \"apply adjustment number {i} to a running order total\" {{\n  {inner}\n}}\n\n"
        ));
    }
    s.push_str("@entry\nmain() -> Num \"checkout total\" {\n  price_total(19.99)\n}\n");
    s
}

/// Sweep program sizes at a fixed fan-out.
///
/// `fanout` is how many helpers the edited bucket calls — the thing subgraph
/// cost actually depends on. Holding it constant while `n` grows isolates the
/// claim: file cost rises with the program, neighbourhood cost does not.
pub fn sweep_with_fanout(sizes: &[usize], fanout: usize) -> Result<Vec<Row>, String> {
    let overhead = estimate_tokens(crate::agent::SYSTEM)
        + crate::agent::tools()
            .iter()
            .map(|t| estimate_tokens(&t.description) + estimate_tokens(&t.input_schema.to_string()))
            .sum::<usize>();

    let mut rows = Vec::new();
    for &n in sizes {
        let src = program(n, fanout);
        let compiled = compile(
            &src,
            CompileOptions {
                strict: true,
                profile: BuildProfile::Dev,
            },
        )
        .map_err(|e| e.to_string())?;

        let file = strategy::build(
            Strategy::WholeFile,
            &compiled.registry,
            &src,
            "price_total",
            1,
        )?;
        let sub = strategy::build(
            Strategy::Subgraph,
            &compiled.registry,
            &src,
            "price_total",
            1,
        )?;

        rows.push(Row {
            buckets: n + 2,
            fanout: fanout.min(n),
            file_tokens: file.tokens,
            subgraph_tokens: sub.tokens,
            agent_overhead: overhead,
            context_ratio: file.tokens as f64 / sub.tokens.max(1) as f64,
            effective_ratio: (file.tokens + overhead) as f64
                / (sub.tokens + overhead).max(1) as f64,
        });
    }
    Ok(rows)
}

/// Default sweep: a realistic fan-out of 3.
pub fn sweep(sizes: &[usize]) -> Result<Vec<Row>, String> {
    sweep_with_fanout(sizes, 3)
}

/// Hold the program size constant and vary fan-out, to show what subgraph cost
/// actually responds to.
pub fn fanout_sweep(buckets: usize, fanouts: &[usize]) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    for &f in fanouts {
        rows.extend(sweep_with_fanout(&[buckets], f)?);
    }
    Ok(rows)
}

pub fn to_csv(rows: &[Row]) -> String {
    let mut s = String::from(
        "buckets,fanout,file_tokens,subgraph_tokens,agent_overhead,context_ratio,effective_ratio\n",
    );
    for r in rows {
        s.push_str(&format!(
            "{},{},{},{},{},{:.3},{:.3}\n",
            r.buckets,
            r.fanout,
            r.file_tokens,
            r.subgraph_tokens,
            r.agent_overhead,
            r.context_ratio,
            r.effective_ratio
        ));
    }
    s
}
