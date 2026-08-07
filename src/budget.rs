//! Spend limits.
//!
//! The edit loop retries on failure, and a retry loop wired to a paid API is
//! exactly the shape that runs up a bill when something goes wrong. Every limit
//! here is enforced in two places: before a request (would this call take us
//! over?) and after it (did it?). The pre-check matters more — it refuses to
//! *start* a call that could exceed the ceiling, rather than discovering the
//! overspend once it has already been paid for.
//!
//! Defaults are deliberately small. A run that needs more should say so
//! explicitly rather than inheriting a generous default.

use serde::Serialize;

use crate::llm::Usage;

/// Per-million-token prices, used only to enforce the USD ceiling.
///
/// A stale price here makes the guard *more* conservative or less, so it is
/// checked alongside the token ceilings rather than trusted alone.
#[derive(Debug, Clone, Copy)]
pub struct Pricing {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
    pub cache_read_per_mtok: f64,
    pub cache_write_per_mtok: f64,
}

impl Pricing {
    /// Published rates for the given model, or a deliberately pessimistic
    /// fallback for anything unrecognized so an unknown model cannot slip
    /// under the USD ceiling by being priced at zero.
    pub fn for_model(model: &str) -> Self {
        // Cache reads bill at ~0.1x input, cache writes at ~1.25x.
        let (input, output) = if model.contains("haiku") {
            (1.00, 5.00)
        } else if model.contains("sonnet") {
            (3.00, 15.00)
        } else if model.contains("fable") || model.contains("mythos") {
            (10.00, 50.00)
        } else {
            // Opus tier, and the fallback for anything unknown.
            (5.00, 25.00)
        };
        Pricing {
            input_per_mtok: input,
            output_per_mtok: output,
            cache_read_per_mtok: input * 0.1,
            cache_write_per_mtok: input * 1.25,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub max_requests: u32,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    pub max_usd: f64,
}

impl Default for Budget {
    /// Enough for a handful of edits, not enough to matter if it misbehaves.
    fn default() -> Self {
        Budget {
            max_requests: 8,
            max_input_tokens: 200_000,
            max_output_tokens: 32_000,
            max_usd: 1.00,
        }
    }
}

impl Budget {
    /// A single call — for smoke tests against the live API.
    pub fn one_shot() -> Self {
        Budget {
            max_requests: 1,
            max_input_tokens: 20_000,
            max_output_tokens: 4_000,
            max_usd: 0.25,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Spend {
    pub requests: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: u32,
    pub cache_write_tokens: u32,
    pub usd: f64,
}

impl Spend {
    pub fn record(&mut self, usage: &Usage, pricing: Pricing) {
        self.requests += 1;
        self.input_tokens += usage.input_tokens;
        self.output_tokens += usage.output_tokens;
        self.cache_read_tokens += usage.cache_read_input_tokens;
        self.cache_write_tokens += usage.cache_creation_input_tokens;
        self.usd += cost(usage, pricing);
    }

    pub fn summary(&self) -> String {
        format!(
            "{} request(s), {} in / {} out tokens ({} cached), ${:.4}",
            self.requests,
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.usd
        )
    }
}

fn cost(usage: &Usage, p: Pricing) -> f64 {
    let m = 1_000_000.0;
    (usage.input_tokens as f64 / m) * p.input_per_mtok
        + (usage.output_tokens as f64 / m) * p.output_per_mtok
        + (usage.cache_read_input_tokens as f64 / m) * p.cache_read_per_mtok
        + (usage.cache_creation_input_tokens as f64 / m) * p.cache_write_per_mtok
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitHit {
    Requests,
    InputTokens,
    OutputTokens,
    Usd,
}

impl std::fmt::Display for LimitHit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            LimitHit::Requests => "request count",
            LimitHit::InputTokens => "input tokens",
            LimitHit::OutputTokens => "output tokens",
            LimitHit::Usd => "estimated spend",
        };
        write!(f, "{s}")
    }
}

/// Tracks spend against a budget and refuses to go over.
#[derive(Debug, Clone)]
pub struct Guard {
    budget: Budget,
    pricing: Pricing,
    spend: Spend,
}

impl Guard {
    pub fn new(budget: Budget, model: &str) -> Self {
        Guard {
            budget,
            pricing: Pricing::for_model(model),
            spend: Spend::default(),
        }
    }

    pub fn spend(&self) -> &Spend {
        &self.spend
    }

    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    /// Check before issuing a request.
    ///
    /// `projected_input` is the size of the prompt about to be sent; the
    /// projected output is the configured `max_tokens`, since that is the most
    /// the call can produce. Both are counted at full price — a pre-check that
    /// assumed a cache hit would let a cold call cross the ceiling.
    pub fn check_before(
        &self,
        projected_input: u32,
        projected_output: u32,
    ) -> Result<(), LimitHit> {
        if self.spend.requests + 1 > self.budget.max_requests {
            return Err(LimitHit::Requests);
        }
        if self.spend.input_tokens + projected_input > self.budget.max_input_tokens {
            return Err(LimitHit::InputTokens);
        }
        if self.spend.output_tokens + projected_output > self.budget.max_output_tokens {
            return Err(LimitHit::OutputTokens);
        }
        let projected_usd = self.spend.usd
            + (projected_input as f64 / 1_000_000.0) * self.pricing.input_per_mtok
            + (projected_output as f64 / 1_000_000.0) * self.pricing.output_per_mtok;
        if projected_usd > self.budget.max_usd {
            return Err(LimitHit::Usd);
        }
        Ok(())
    }

    /// Record actual usage after a response, and report whether that call put
    /// us over. The call already happened, so this stops the *next* one.
    pub fn record(&mut self, usage: &Usage) -> Result<(), LimitHit> {
        self.spend.record(usage, self.pricing);
        if self.spend.requests > self.budget.max_requests {
            return Err(LimitHit::Requests);
        }
        if self.spend.input_tokens > self.budget.max_input_tokens {
            return Err(LimitHit::InputTokens);
        }
        if self.spend.output_tokens > self.budget.max_output_tokens {
            return Err(LimitHit::OutputTokens);
        }
        if self.spend.usd > self.budget.max_usd {
            return Err(LimitHit::Usd);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u32, output: u32) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            ..Default::default()
        }
    }

    #[test]
    fn refuses_a_request_that_would_exceed_the_count() {
        let mut g = Guard::new(
            Budget {
                max_requests: 1,
                ..Budget::default()
            },
            "claude-opus-5",
        );
        assert!(g.check_before(100, 100).is_ok());
        g.record(&usage(100, 100)).unwrap();
        assert_eq!(g.check_before(100, 100), Err(LimitHit::Requests));
    }

    #[test]
    fn pre_check_stops_the_call_before_it_is_paid_for() {
        // The whole point of check_before: the ceiling is enforced without
        // first spending the money that would breach it.
        let g = Guard::new(
            Budget {
                max_input_tokens: 1_000,
                ..Budget::default()
            },
            "claude-opus-5",
        );
        assert_eq!(g.check_before(5_000, 100), Err(LimitHit::InputTokens));
        assert_eq!(g.spend().requests, 0, "nothing should have been spent");
    }

    #[test]
    fn usd_ceiling_binds_even_when_token_ceilings_do_not() {
        let g = Guard::new(
            Budget {
                max_requests: 100,
                max_input_tokens: 10_000_000,
                max_output_tokens: 10_000_000,
                max_usd: 0.01,
            },
            "claude-opus-5",
        );
        // 100k output at $25/MTok is $2.50 — far past a one-cent ceiling.
        assert_eq!(g.check_before(1_000, 100_000), Err(LimitHit::Usd));
    }

    #[test]
    fn unknown_models_price_at_the_pessimistic_tier() {
        let p = Pricing::for_model("some-unreleased-model");
        assert_eq!(p.input_per_mtok, 5.00);
        assert_eq!(p.output_per_mtok, 25.00);
    }

    #[test]
    fn spend_accumulates_across_calls() {
        let mut g = Guard::new(Budget::default(), "claude-opus-5");
        g.record(&usage(1_000, 500)).unwrap();
        g.record(&usage(1_000, 500)).unwrap();
        assert_eq!(g.spend().requests, 2);
        assert_eq!(g.spend().input_tokens, 2_000);
        assert!(g.spend().usd > 0.0);
    }
}
