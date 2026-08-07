//! How much of the program to send.
//!
//! The experiment needs a control. `WholeFile` is what a conventional coding
//! agent does — paste the file and ask for an edit; `Subgraph` is the bucket
//! neighbourhood. Both feed the *same* system prompt, the *same* schema, and
//! the *same* gates, so a difference in outcome is attributable to the context
//! and nothing else.
//!
//! `Auto` is the dynamic policy: measure both and send whichever is smaller.
//! It exists because the subgraph is *not* always smaller — on a small file the
//! pack's scaffolding costs more than the file it replaces, and pretending
//! otherwise would make the comparison dishonest.

use bucketlang::registry::Registry;
use serde::Serialize;

use crate::context::build_context;
use crate::metrics::estimate_tokens;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// Send the entire source file. The conventional baseline.
    WholeFile,
    /// Send the target bucket's body plus neighbour signatures and descriptions.
    Subgraph,
    /// Measure both, send the cheaper one.
    Auto,
}

impl Strategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Strategy::WholeFile => "whole_file",
            Strategy::Subgraph => "subgraph",
            Strategy::Auto => "auto",
        }
    }
}

impl std::str::FromStr for Strategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "whole_file" | "whole-file" | "file" | "classic" => Ok(Strategy::WholeFile),
            "subgraph" | "buckets" | "bucket" => Ok(Strategy::Subgraph),
            "auto" => Ok(Strategy::Auto),
            other => Err(format!(
                "unknown strategy '{other}' (expected whole_file, subgraph, or auto)"
            )),
        }
    }
}

/// The context to send, plus what it cost and what the alternative would have.
#[derive(Debug, Clone, Serialize)]
pub struct RenderedContext {
    /// Which strategy actually produced `text` — for `Auto`, the one it chose.
    pub strategy: Strategy,
    pub text: String,
    pub tokens: usize,
    /// Cost of the strategy not taken, so every run reports the comparison
    /// rather than only the arm that ran.
    pub alternative_tokens: usize,
}

impl RenderedContext {
    /// `alternative / chosen`. Above 1.0 means the chosen context was cheaper.
    pub fn savings_ratio(&self) -> f64 {
        self.alternative_tokens as f64 / self.tokens.max(1) as f64
    }
}

/// Render the file as a conventional agent would: the whole thing, with the
/// target named so the model knows what to change.
fn whole_file(source: &str, target: &str) -> String {
    format!("# edit the bucket `{target}` in this file\n\n{source}\n")
}

/// Build the context for one edit under `strategy`.
pub fn build(
    strategy: Strategy,
    reg: &Registry,
    source: &str,
    target: &str,
    depth: usize,
) -> Result<RenderedContext, String> {
    let file_text = whole_file(source, target);
    let file_tokens = estimate_tokens(&file_text);

    // The subgraph can legitimately fail where the file cannot — an imported
    // or unknown target. Fall back rather than failing the run, and say so.
    let sub = build_context(reg, target, depth)
        .ok()
        .map(|pack| pack.render());

    match strategy {
        Strategy::WholeFile => Ok(RenderedContext {
            strategy: Strategy::WholeFile,
            tokens: file_tokens,
            alternative_tokens: sub.as_deref().map(estimate_tokens).unwrap_or(0),
            text: file_text,
        }),
        Strategy::Subgraph => {
            let text = sub.ok_or_else(|| {
                format!("cannot build a subgraph for {target}; try --strategy whole_file")
            })?;
            Ok(RenderedContext {
                strategy: Strategy::Subgraph,
                tokens: estimate_tokens(&text),
                alternative_tokens: file_tokens,
                text,
            })
        }
        Strategy::Auto => match sub {
            Some(text) if estimate_tokens(&text) < file_tokens => Ok(RenderedContext {
                strategy: Strategy::Subgraph,
                tokens: estimate_tokens(&text),
                alternative_tokens: file_tokens,
                text,
            }),
            other => Ok(RenderedContext {
                strategy: Strategy::WholeFile,
                tokens: file_tokens,
                alternative_tokens: other.as_deref().map(estimate_tokens).unwrap_or(0),
                text: file_text,
            }),
        },
    }
}
