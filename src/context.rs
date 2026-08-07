//! Retrieval: request -> smallest sufficient subgraph.
//!
//! This is policy, not language. The rule it implements is the one the whole
//! design rests on: send the *target bucket's body*, but only the *signature and
//! description* of its neighbours. That asymmetry is the compression — a
//! caller's description is meant to be sufficient to reason about it without
//! reading its code, which is exactly why a bucket may not exist without one.
//!
//! Depth, what counts as a neighbour, and whether to include tests are all knobs
//! that belong here so they can be tuned and measured without touching the
//! compiler.

use bucketlang::ast::BucketKind;
use bucketlang::error::{Error, Result};
use bucketlang::graph::build_graph;
use bucketlang::registry::Registry;
use bucketlang::render::render_labelled;
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize)]
pub struct ContextPack {
    pub file_hint: String,
    pub target: ContextBucket,
    pub callees: Vec<ContextBucket>,
    pub callers: Vec<ContextRef>,
    pub tests: Vec<ContextTest>,
    pub cores_used: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextBucket {
    pub address: String,
    pub label: Option<String>,
    pub desc: String,
    pub contract: String,
    pub body_labelled: String,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextRef {
    pub address: String,
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextTest {
    pub address: String,
    pub desc: String,
    pub subject: Option<String>,
    pub body_labelled: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EditResult {
    pub ok: bool,
    pub bucket: String,
    pub tests_run: usize,
    pub tests_passed: usize,
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prints: Option<Vec<String>>,
    /// Present on successful dry-run (no `--write`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

fn contract_str(b: &bucketlang::ast::Bucket) -> String {
    let params: Vec<String> = b
        .contract
        .params
        .iter()
        .map(|p| format!("{}: {}", p.name, p.ty.name()))
        .collect();
    format!("({}) -> {}", params.join(", "), b.contract.ret.name())
}

fn bucket_pack(reg: &Registry, addr: &str) -> Option<ContextBucket> {
    let b = reg.get(addr)?;
    Some(ContextBucket {
        address: b.address.clone(),
        label: b.label.clone(),
        desc: b.desc.clone(),
        contract: contract_str(b),
        body_labelled: render_labelled(&b.body, reg),
        kind: match b.kind {
            BucketKind::User => "user".into(),
            BucketKind::Test => "test".into(),
            BucketKind::Core => "core".into(),
        },
    })
}

pub fn build_context(reg: &Registry, target: &str, depth: usize) -> Result<ContextPack> {
    let addr = reg
        .resolve_target(target)
        .ok_or_else(|| Error::msg(format!("unknown bucket {target}")))?;
    let g = build_graph(reg);
    let node = g
        .get(&addr)
        .ok_or_else(|| Error::msg(format!("no graph node for {addr}")))?;

    let mut callees = Vec::new();
    let mut cores_used = BTreeSet::new();
    let mut frontier: Vec<(String, usize)> = node.out.iter().cloned().map(|a| (a, 1)).collect();
    let mut seen = BTreeSet::from([addr.clone()]);

    while let Some((id, d)) = frontier.pop() {
        if d > depth || !seen.insert(id.clone()) {
            continue;
        }
        if let Some(b) = reg.get(&id) {
            match b.kind {
                BucketKind::Core => {
                    cores_used.insert(id);
                }
                BucketKind::User => {
                    callees.push(bucket_pack(reg, &id).unwrap());
                    if d < depth {
                        if let Some(n) = g.get(&id) {
                            for o in &n.out {
                                frontier.push((o.clone(), d + 1));
                            }
                        }
                    }
                }
                BucketKind::Test => {}
            }
        }
    }
    callees.sort_by(|a, b| a.address.cmp(&b.address));

    let mut callers = Vec::new();
    for c in &node.inn {
        if let Some(b) = reg.get(c) {
            if b.kind == BucketKind::User {
                callers.push(ContextRef {
                    address: b.address.clone(),
                    label: b.label.clone(),
                });
            }
        }
    }

    let subject_label = reg.get(&addr).and_then(|b| b.label.clone());
    let mut tests = Vec::new();
    for tid in &reg.test_ids {
        if let Some(b) = reg.get(tid) {
            let related = b.subject.as_ref().is_some_and(|s| {
                s == &addr || subject_label.as_ref().is_some_and(|l| l == s)
            });
            if related {
                tests.push(ContextTest {
                    address: b.address.clone(),
                    desc: b.desc.clone(),
                    subject: b.subject.clone(),
                    body_labelled: render_labelled(&b.body, reg),
                });
            }
        }
    }

    Ok(ContextPack {
        file_hint: String::new(),
        target: bucket_pack(reg, &addr).unwrap(),
        callees,
        callers,
        tests,
        cores_used: cores_used.into_iter().collect(),
    })
}
impl ContextPack {
    /// The pack as the text an agent receives.
    ///
    /// The shape is the whole argument: the target arrives as a full body, its
    /// neighbours as one line each. A callee's description stands in for its
    /// code, which is the substitution the compression depends on — and the
    /// reason a bucket without a description is not merely undocumented but
    /// unusable here.
    pub fn render(&self) -> String {
        let mut s = String::new();

        s.push_str(&format!(
            "# edit this bucket\n{} {}{}\n  \"{}\"\n  {{\n{}\n  }}\n",
            self.target.address,
            self.target.label.clone().unwrap_or_default(),
            self.target.contract,
            self.target.desc,
            indent(&self.target.body_labelled, 4),
        ));

        if !self.callees.is_empty() {
            s.push_str("\n# it calls these (signature and description only)\n");
            for c in &self.callees {
                s.push_str(&format!(
                    "{} {}{} \"{}\"\n",
                    c.address,
                    c.label.clone().unwrap_or_default(),
                    c.contract,
                    c.desc
                ));
            }
        }

        if !self.callers.is_empty() {
            s.push_str("\n# these call it — changing its contract breaks them\n");
            for c in &self.callers {
                s.push_str(&format!(
                    "{} {}\n",
                    c.address,
                    c.label.clone().unwrap_or_default()
                ));
            }
        }

        if !self.cores_used.is_empty() {
            s.push_str(&format!("\n# cores in scope: {}\n", self.cores_used.join(", ")));
        }

        if !self.tests.is_empty() {
            s.push_str("\n# these must still pass\n");
            for t in &self.tests {
                s.push_str(&format!("{}\n", t.body_labelled));
            }
        }

        s
    }
}

fn indent(s: &str, n: usize) -> String {
    let pad = " ".repeat(n);
    s.lines()
        .map(|l| format!("{pad}{l}"))
        .collect::<Vec<_>>()
        .join("\n")
}
