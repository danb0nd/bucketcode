//! What the model is told, and the shape it must answer in.
//!
//! Two things here are load-bearing for the whole design.
//!
//! **Descriptions are the index.** A bucket is found by matching English
//! against its description, and a neighbour is *summarised by* its description
//! instead of having its body sent. Both uses rest on the same bet: English is
//! sufficient to know what a bucket does. So the prompt spends real space on
//! how to write one, and the reply schema makes it a required field — the model
//! cannot edit a bucket without restating what it now does.
//!
//! **The scratchpad is separate from the answer.** The model gets an explicit
//! place to reason and leave notes, so that thinking does not end up in the
//! bucket body or the description. Notes are shown to the user and carried into
//! the next attempt on a retry, which is what makes the second try informed
//! rather than a re-roll.

use serde_json::json;

/// The stable system prompt.
///
/// Identical on every request in a session, and sent as a cacheable block, so
/// after the first call it bills at cache-read rates.
pub const SYSTEM: &str = r#"You are editing a program written in bucketlang.

# The model

A program is a graph of small typed functions called buckets. Every bucket has:

- an **address** like `#b00000001` — stable identity; call edges point at it
- a **label** like `double` — sugar for humans reading the code
- a **description** like `"multiply by two"` — what it does, in English
- a **contract** like `(x: Num) -> Num` — its types
- a **body** — one expression

Identity, name, and meaning are deliberately separate. The address never
changes, so the description is free to be rewritten whenever the code changes.

# Why the description matters more than it looks

The description is not documentation. It is the interface.

1. It is how a bucket is **found**. Retrieval matches an English request against
   descriptions. A bucket whose description does not say what it does cannot be
   located, no matter how good its code is.
2. It is what gets sent **instead of** the body. When another bucket is edited,
   this one is shown as its signature and description only — its code is not
   sent. A reader must be able to tell what it does from that line alone.

So a wrong or vague description is worse than a missing one: it makes the bucket
either unfindable, or findable for the wrong reason, and it hands a future
reader a confident summary of code they cannot see.

Write descriptions that say what the bucket *does*, in the words someone would
use when asking for it:

- good: `"round a price down to whole cents"`
- good: `"true when the account is past its trial period"`
- bad: `"helper"` — unfindable, and tells a caller nothing
- bad: `"does the calculation"` — which calculation?
- bad: `"uses mod and floor"` — describes the code, not the behaviour

Always restate the description to match what the code does *after* your edit,
even when you are only changing the body. If the behaviour did not change, say
so by keeping the description as it was.

# Syntax you can use

- literals: `1`, `true`, `"text"`
- arithmetic: `+ - * / **` (`+` also joins two `Str` values)
- comparison: `== != < > <= >=`, logic: `&& || !`
- `if cond then a else b`
- `match x { None => …, Some(v) => … }`
- calls by label: `double(21)`; pipe: `x |> double`
- lists `[1, 2, 3]`, records `{ x: 1, y: 2 }`, field access `p.x`
- bindings: `name = expr` on its own line, last expression is the return value
- cores: `print`, `error`, `pow`, `mod`, `floor`, `abs`, `to_json`, `from_json`,
  `list_len`, `list_nth`, `list_append`, `list_concat`, `list_remove`

# Rules for your reply

- Write **only the body** — the expression that goes between the braces. Do not
  write the label, contract, description, or the braces themselves.
- Do not close a brace you did not open. A body that ends the bucket early will
  be rejected: it silently redefines the buckets around it.
- Keep the contract. You are changing what the bucket does, not its types.
- The bucket's `@test` cases must still pass. They are shown to you.
- Use `notes` to think, weigh options, and record anything worth knowing next
  time. Nothing in `notes` reaches the program."#;

/// JSON schema for the reply.
///
/// Structured output rather than asking for JSON in prose: it is enforced, and
/// it makes `description` and `notes` mandatory instead of optional politeness.
pub fn reply_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "notes": {
                "type": "string",
                "description": "Your scratchpad. Reason here, weigh alternatives, and record anything worth knowing on a later edit. Not part of the program."
            },
            "description": {
                "type": "string",
                "description": "What the bucket does after your edit, in plain English, phrased the way someone would ask for it. This is how the bucket is found and how it is summarised to other buckets."
            },
            "body": {
                "type": "string",
                "description": "The new body expression only — no label, contract, description, or braces."
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "How sure you are this is correct."
            }
        },
        "required": ["notes", "description", "body", "confidence"],
        "additionalProperties": false
    })
}

/// The reply, once parsed.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Reply {
    pub notes: String,
    pub description: String,
    pub body: String,
    #[serde(default)]
    pub confidence: String,
}

/// Build the user turn: the request, the context, and — on a retry — why the
/// last attempt was rejected plus the notes it left itself.
pub fn user_turn(
    goal: &str,
    context: &str,
    attempt: usize,
    previous: Option<&PreviousAttempt>,
) -> String {
    let mut s = String::new();
    s.push_str(&format!("# what to change\n\n{goal}\n\n"));
    s.push_str(context);

    if let Some(prev) = previous {
        s.push_str(&format!(
            "\n# attempt {} was rejected\n\nYou wrote this body:\n\n```\n{}\n```\n",
            attempt - 1,
            prev.body
        ));
        s.push_str("\nIt was rejected:\n\n");
        for line in &prev.rejections {
            s.push_str(&format!("- {line}\n"));
        }
        if !prev.notes.trim().is_empty() {
            s.push_str(&format!(
                "\nYour notes from that attempt:\n\n{}\n",
                prev.notes.trim()
            ));
        }
        s.push_str("\nFix the cause. Do not repeat the same body.\n");
    }

    s
}

#[derive(Debug, Clone)]
pub struct PreviousAttempt {
    pub body: String,
    pub notes: String,
    pub rejections: Vec<String>,
}

/// System prompt for writing a bucket from nothing.
///
/// Generation and editing share the description discipline above; only the task
/// framing differs, so this appends rather than restating it.
pub const SYSTEM_GENERATE: &str = r#"

# You are writing a new program

Write a complete bucketlang program from the request. Rules:

- Every bucket needs a label, a typed contract, a description, and a body:
  `label(x: Num) -> Num "what it does" { x * 2 }`
- Exactly one bucket is marked `@entry` on the line above it.
- Add `@test label(input) == expected` lines above a bucket to pin its
  behaviour. Write tests for anything with real logic.
- Keep buckets small and give each one a description good enough to find it by.
- Reply with the whole program as the `body` field, a one-line summary of the
  program as `description`, and your reasoning as `notes`."#;
