//! bucketcode — an agentic coding harness for bucketlang.
//!
//! Conventional coding agents send a file and ask for a rewrite. This one sends
//! a *neighbourhood* — one bucket's body plus the signatures and English
//! descriptions of what it calls — and asks for one bucket's body back. Then it
//! puts that answer through deterministic gates before anything is written.
//!
//! The pieces:
//!
//! - [`index`] — English in, bucket addresses out. The step that decides
//!   *what* to edit, matched against descriptions.
//! - [`strategy`] — how much to send: the whole file (the conventional
//!   baseline), the subgraph, or whichever is cheaper. Both arms share one
//!   prompt and one set of gates, so the comparison is honest.
//! - [`prompt`] — what the model is told, including why descriptions are the
//!   index and must be rewritten with every edit.
//! - [`llm`] — the Anthropic Messages API client.
//! - [`budget`] — hard spend ceilings, checked before a call is made.
//! - [`loop_`] — propose, gate, feed the failure back, retry.
//! - [`metrics`] — what it cost, versus sending the file.
//!
//! The language lives in a separate crate on purpose. `bucketlang` has to be
//! correct; this crate is an experiment whose whole point is finding where
//! compression breaks correctness.

pub mod bench;
pub mod budget;
pub mod context;
pub mod index;
pub mod llm;
pub mod loop_;
pub mod metrics;
pub mod prompt;
pub mod result;
pub mod strategy;

pub use budget::{Budget, Guard, LimitHit, Pricing, Spend};
pub use context::{build_context, ContextBucket, ContextPack, ContextRef, ContextTest};
pub use index::{search, Match};
pub use llm::{Client, Completion, Config, LlmError, Usage, DEFAULT_MODEL};
pub use loop_::{run_edit, Agent, EditOutcome, EditRequest, LoopOptions, ScriptedAgent};
pub use metrics::{estimate_tokens, Metrics};
pub use prompt::Reply;
pub use result::EditResult;
pub use strategy::{RenderedContext, Strategy};
pub mod agent;
pub mod sandbox;
pub mod serve;
