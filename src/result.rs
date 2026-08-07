use serde::Serialize;

#[derive(Debug, Serialize)]
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