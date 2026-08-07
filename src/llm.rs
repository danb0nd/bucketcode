//! Anthropic Messages API client.
//!
//! Raw HTTP rather than an SDK — there is no official Anthropic SDK for Rust.
//! The request and response shapes here follow the documented REST contract.
//!
//! Three details of the current API are load-bearing and easy to get wrong:
//!
//! - **`stop_reason` must be read before `content`.** Safety classifiers can
//!   decline a request and still return HTTP 200 with an empty or partial
//!   `content` array. Indexing `content[0]` first turns a refusal into a panic
//!   or, worse, a silently empty edit.
//! - **No sampling parameters.** `temperature`, `top_p`, and `top_k` are
//!   rejected on current models. Steer with the prompt; use `effort` for depth.
//! - **Usage is authoritative.** The response reports real input/output token
//!   counts, so the harness measures compression against those rather than
//!   trusting its own character heuristic.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default model. Overridable per client — see `Client::with_model`.
pub const DEFAULT_MODEL: &str = "claude-opus-5";

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone)]
pub struct Config {
    pub api_key: String,
    pub model: String,
    /// Hard ceiling on output tokens. Also caps thinking, which is on by
    /// default on current models — too small a value truncates mid-answer.
    pub max_tokens: u32,
    /// `low` | `medium` | `high` | `xhigh` | `max`. Controls reasoning depth
    /// and overall spend. `None` uses the API default.
    pub effort: Option<String>,
    pub base_url: String,
    pub timeout: Duration,
}

impl Config {
    /// Read the key from `ANTHROPIC_API_KEY`.
    pub fn from_env() -> Result<Self, String> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY is not set".to_string())?;
        Ok(Config {
            api_key,
            model: DEFAULT_MODEL.to_string(),
            max_tokens: 16_000,
            effort: Some("high".into()),
            base_url: API_URL.to_string(),
            // Generous: a single request at high effort can think for minutes.
            timeout: Duration::from_secs(600),
        })
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_effort(mut self, effort: Option<String>) -> Self {
        self.effort = effort;
        self
    }

    /// Drop parameters the target model does not accept.
    ///
    /// `effort` is rejected outright by Haiku 4.5, so passing it through
    /// unconditionally turns a valid request into a 400 that reads as if the
    /// model itself were unavailable.
    pub fn normalized(mut self) -> Self {
        if self.model.contains("haiku") {
            self.effort = None;
        }
        self
    }
}

// ---------------------------------------------------------------- request

#[derive(Debug, Serialize)]
struct Request<'a> {
    model: &'a str,
    max_tokens: u32,
    /// System blocks rather than a bare string, so the stable prefix can carry
    /// `cache_control` and be billed at cache-read rates on later turns.
    system: Vec<SystemBlock<'a>>,
    messages: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Tool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig>,
}

#[derive(Debug, Serialize)]
struct SystemBlock<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Debug, Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
}

/// A conversation turn. Content is a block list rather than a string because
/// tool use requires it: an assistant turn carries `tool_use` blocks, and the
/// user turn answering it carries `tool_result` blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: serde_json::Value,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Message { role: "user".into(), content: serde_json::Value::String(text.into()) }
    }

    /// Echo an assistant turn back verbatim. The full content must be preserved
    /// -- dropping the `tool_use` blocks breaks the pairing the API requires.
    pub fn assistant(content: serde_json::Value) -> Self {
        Message { role: "assistant".into(), content }
    }

    /// Results for one assistant turn. Every `tool_use` needs a matching
    /// `tool_result` in a *single* user message -- splitting them across
    /// messages teaches the model to stop calling tools in parallel.
    pub fn tool_results(results: Vec<serde_json::Value>) -> Self {
        Message { role: "user".into(), content: serde_json::Value::Array(results) }
    }
}

/// A tool the model may call.
#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// One tool invocation from the model.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

impl ToolCall {
    /// Build the result block that answers this call.
    pub fn result(&self, content: impl Into<String>, is_error: bool) -> serde_json::Value {
        serde_json::json!({
            "type": "tool_result",
            "tool_use_id": self.id,
            "content": content.into(),
            "is_error": is_error,
        })
    }
}

#[derive(Debug, Serialize)]
struct OutputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<String>,
    /// Constrains the reply to a JSON schema. Used instead of asking for JSON
    /// in prose, which is unreliable, and instead of prefilling the assistant
    /// turn, which current models reject.
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<serde_json::Value>,
}

// --------------------------------------------------------------- response

#[derive(Debug, Deserialize)]
struct ApiResponse {
    #[serde(default)]
    content: Vec<ContentBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_details: Option<StopDetails>,
    #[serde(default)]
    usage: Usage,
    #[serde(default)]
    model: String,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct StopDetails {
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    explanation: Option<String>,
}

/// Token accounting straight from the API — what the harness measures with.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
}

impl Usage {
    /// Everything the request cost on the way in, cached or not.
    pub fn total_input(&self) -> u32 {
        self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens
    }
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub text: String,
    pub usage: Usage,
    pub model: String,
    pub stop_reason: Option<String>,
    /// Tool calls the model made this turn. Non-empty iff `stop_reason` is
    /// `tool_use`.
    pub tool_calls: Vec<ToolCall>,
    /// The assistant turn exactly as received, for echoing back into history.
    pub raw_content: serde_json::Value,
}

impl Completion {
    pub fn wants_tools(&self) -> bool {
        self.stop_reason.as_deref() == Some("tool_use")
    }
}

#[derive(Debug)]
pub enum LlmError {
    /// Safety classifiers declined. Retrying the same prompt will not help.
    Refused {
        category: Option<String>,
        explanation: Option<String>,
    },
    /// Output hit `max_tokens` — the reply is truncated, not wrong.
    Truncated,
    Http {
        status: u16,
        body: String,
    },
    Transport(String),
    Decode(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::Refused {
                category,
                explanation,
            } => {
                write!(f, "request was declined")?;
                if let Some(c) = category {
                    write!(f, " ({c})")?;
                }
                if let Some(e) = explanation {
                    write!(f, ": {e}")?;
                }
                Ok(())
            }
            LlmError::Truncated => write!(
                f,
                "response hit max_tokens and is truncated; raise max_tokens or lower effort"
            ),
            LlmError::Http { status, body } => write!(f, "api returned {status}: {body}"),
            LlmError::Transport(e) => write!(f, "transport error: {e}"),
            LlmError::Decode(e) => write!(f, "could not decode response: {e}"),
        }
    }
}

impl std::error::Error for LlmError {}

pub struct Client {
    cfg: Config,
}

impl Client {
    pub fn new(cfg: Config) -> Self {
        Client { cfg }
    }

    pub fn model(&self) -> &str {
        &self.cfg.model
    }

    /// One request. Returns the assistant turn, including any tool calls.
    ///
    /// `system` is sent as a cacheable block: it is identical across every turn
    /// of a session, so after the first call it bills at cache-read rates.
    pub fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: Option<&[Tool]>,
        schema: Option<serde_json::Value>,
    ) -> Result<Completion, LlmError> {
        let format = schema.map(|s| serde_json::json!({ "type": "json_schema", "schema": s }));

        let req = Request {
            model: &self.cfg.model,
            max_tokens: self.cfg.max_tokens,
            system: vec![SystemBlock {
                kind: "text",
                text: system,
                cache_control: Some(CacheControl { kind: "ephemeral" }),
            }],
            messages,
            tools,
            output_config: if self.cfg.effort.is_some() || format.is_some() {
                Some(OutputConfig { effort: self.cfg.effort.clone(), format })
            } else {
                None
            },
        };

        let resp = ureq::post(&self.cfg.base_url)
            .config()
            .timeout_global(Some(self.cfg.timeout))
            .build()
            .header("content-type", "application/json")
            .header("x-api-key", &self.cfg.api_key)
            .header("anthropic-version", API_VERSION)
            .send_json(&req);

        let mut resp = match resp {
            Ok(r) => r,
            Err(ureq::Error::StatusCode(code)) => {
                return Err(LlmError::Http {
                    status: code,
                    // The API explains its 4xx responses; discarding that body
                    // turns a self-describing error into a guessing game.
                    body: "check the model id and that every parameter is supported \
                           on it (effort is not accepted on Haiku 4.5)"
                        .to_string(),
                })
            }
            Err(e) => return Err(LlmError::Transport(e.to_string())),
        };

        let raw: serde_json::Value = resp
            .body_mut()
            .read_json()
            .map_err(|e| LlmError::Decode(e.to_string()))?;
        let parsed: ApiResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::Decode(e.to_string()))?;

        // Order matters: a refusal can carry an empty content array, so this
        // has to happen before anything reads content.
        match parsed.stop_reason.as_deref() {
            Some("refusal") => {
                let (category, explanation) = parsed
                    .stop_details
                    .map(|d| (d.category, d.explanation))
                    .unwrap_or((None, None));
                return Err(LlmError::Refused { category, explanation });
            }
            Some("max_tokens") => return Err(LlmError::Truncated),
            _ => {}
        }

        // Thinking blocks share the content array with text; keep only text.
        let text = parsed
            .content
            .iter()
            .filter(|b| b.kind == "text")
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("");

        let tool_calls = parsed
            .content
            .iter()
            .filter(|b| b.kind == "tool_use")
            .filter_map(|b| {
                Some(ToolCall {
                    id: b.id.clone()?,
                    name: b.name.clone()?,
                    input: b.input.clone().unwrap_or(serde_json::Value::Null),
                })
            })
            .collect();

        Ok(Completion {
            text,
            usage: parsed.usage,
            model: parsed.model,
            stop_reason: parsed.stop_reason,
            tool_calls,
            raw_content: raw.get("content").cloned().unwrap_or(serde_json::Value::Null),
        })
    }
}
