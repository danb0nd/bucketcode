//! What an edit cost.
//!
//! The headline claim is that sending one bucket's neighbourhood beats sending
//! the file. That is a measurement, so it lives here rather than being asserted
//! in a README.

use serde::Serialize;

/// Rough token count.
///
/// Deliberately a heuristic, not a tokenizer: it is used to compare *two*
/// representations of the same program, and any consistent measure ranks them
/// the same way. Treat the ratio as meaningful and the absolute number as not.
pub fn estimate_tokens(s: &str) -> usize {
    // ~4 characters per token is the usual rule of thumb for code.
    let chars = s.chars().count();
    chars.div_ceil(4).max(1)
}

#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    /// Tokens if the whole file had been sent, the thing we are trying to beat.
    pub whole_file_tokens: usize,
    /// Tokens actually sent: the target body plus neighbour signatures.
    pub context_tokens: usize,
    /// `whole_file_tokens / context_tokens`. Above 1.0 means the subgraph won.
    pub compression_ratio: f64,
    /// How many proposals it took. 1 means it passed every gate first try.
    pub attempts: usize,
    pub passed_first_try: bool,
    /// Tokens spent on retries — the tax for a representation that was too thin.
    pub retry_tokens: usize,
}

impl Metrics {
    pub fn new(whole_file: &str, context: &str, attempts: usize, retry_tokens: usize) -> Self {
        let whole = estimate_tokens(whole_file);
        let ctx = estimate_tokens(context);
        Metrics {
            whole_file_tokens: whole,
            context_tokens: ctx,
            compression_ratio: whole as f64 / ctx.max(1) as f64,
            attempts,
            passed_first_try: attempts <= 1,
            retry_tokens,
        }
    }

    /// One line for a terminal.
    pub fn summary(&self) -> String {
        format!(
            "context {} tok vs whole-file {} tok ({:.2}x), {} attempt(s), {} retry tok",
            self.context_tokens,
            self.whole_file_tokens,
            self.compression_ratio,
            self.attempts,
            self.retry_tokens
        )
    }
}
