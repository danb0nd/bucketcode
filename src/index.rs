//! English in, bucket addresses out.
//!
//! This is the step the whole design starts from and the one that was missing:
//! an agent asked to "round prices down to cents" has to turn that sentence
//! into an address before any of the editing machinery applies.
//!
//! Matching is over the **description** first, then the label and parameter
//! names. That ordering is the design, not a heuristic — the description is the
//! field written for this purpose, and it is the only one that can carry the
//! words a person would actually use.
//!
//! Deliberately keyword scoring rather than embeddings: no API call, no index
//! to keep warm, no cold-start. It is also tuned for **recall** — a target the
//! retriever misses is one the model cannot fix, whereas an extra candidate is
//! cheap for the model to discard. So this returns a ranked shortlist and lets
//! the caller (or the model) choose, instead of guessing at a single answer.

use bucketlang::ast::BucketKind;
use bucketlang::registry::Registry;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize)]
pub struct Match {
    pub address: String,
    pub label: Option<String>,
    pub desc: String,
    pub contract: String,
    pub score: f64,
    /// Which query words hit, so a caller can see *why* this ranked where it did.
    pub matched: Vec<String>,
}

/// Words too common to carry signal. Kept short on purpose — an aggressive stop
/// list costs recall, and recall is the thing being optimised for.
const STOP: &[&str] = &[
    "a", "an", "and", "the", "to", "of", "in", "on", "for", "is", "it", "that", "this", "with",
    "make", "makes", "change", "changes", "fix", "update", "should", "can", "i", "want", "please",
];

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .filter(|w| !STOP.contains(&w.as_str()))
        .collect()
}

/// Crude but effective stemming: fold common English suffixes so "rounding",
/// "rounds", and "rounded" all reach "round". A real stemmer would be better;
/// this covers the cases that actually come up in a request phrased as a task.
fn stem(w: &str) -> String {
    for suffix in ["ing", "ed", "es", "s"] {
        if w.len() > suffix.len() + 2 && w.ends_with(suffix) {
            return w[..w.len() - suffix.len()].to_string();
        }
    }
    w.to_string()
}

fn field_score(query: &[String], field: &str, weight: f64, hits: &mut Vec<String>) -> f64 {
    let words: Vec<String> = tokenize(field).iter().map(|w| stem(w)).collect();
    if words.is_empty() {
        return 0.0;
    }
    let mut score = 0.0;
    for q in query {
        let qs = stem(q);
        if words.iter().any(|w| *w == qs) {
            score += weight;
            hits.push(q.clone());
        } else if words.iter().any(|w| w.starts_with(&qs) || qs.starts_with(w)) {
            // Partial credit for a prefix overlap — recall over precision.
            score += weight * 0.4;
            hits.push(q.clone());
        }
    }
    score
}

/// Rank user buckets against an English query.
///
/// Returns at most `limit` matches, best first. Buckets with no hit at all are
/// omitted; an empty result means the request named nothing in this program.
pub fn search(reg: &Registry, query: &str, limit: usize) -> Vec<Match> {
    let q = tokenize(query);
    if q.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<Match> = Vec::new();
    for (addr, b) in &reg.buckets {
        // Only user buckets are editable, so only user buckets are findable.
        if b.kind != BucketKind::User {
            continue;
        }

        let mut hits = Vec::new();
        // Description carries the most weight: it is the field written to be
        // searched, and the one a caller sees instead of the body.
        let mut score = field_score(&q, &b.desc, 1.0, &mut hits);
        if let Some(label) = &b.label {
            score += field_score(&q, label, 0.7, &mut hits);
        }
        let params: Vec<&str> = b.contract.params.iter().map(|p| p.name.as_str()).collect();
        score += field_score(&q, &params.join(" "), 0.3, &mut hits);

        if score <= 0.0 {
            continue;
        }

        // Normalise by query length so a long request does not simply outscore
        // a short one, and dedupe the hit list for display.
        score /= q.len() as f64;
        let mut seen = BTreeMap::new();
        for h in hits {
            seen.insert(h, ());
        }

        out.push(Match {
            address: addr.clone(),
            label: b.label.clone(),
            desc: b.desc.clone(),
            contract: format!(
                "({}) -> {}",
                b.contract
                    .params
                    .iter()
                    .map(|p| format!("{}: {}", p.name, p.ty.name()))
                    .collect::<Vec<_>>()
                    .join(", "),
                b.contract.ret.name()
            ),
            score,
            matched: seen.into_keys().collect(),
        });
    }

    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Stable tiebreak so results don't shuffle between runs.
            .then_with(|| a.address.cmp(&b.address))
    });
    out.truncate(limit);
    out
}

/// Render a shortlist for a model to choose from.
///
/// Description only — no bodies. Choosing what to edit should cost a handful of
/// tokens, not a copy of the program.
pub fn render_shortlist(matches: &[Match]) -> String {
    if matches.is_empty() {
        return "no buckets matched that request\n".to_string();
    }
    let mut s = String::from("# candidate buckets\n");
    for m in matches {
        s.push_str(&format!(
            "{}  {}{}  \"{}\"\n",
            m.address,
            m.label.clone().unwrap_or_default(),
            m.contract,
            m.desc
        ));
    }
    s
}
