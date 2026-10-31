//! Hugging Face addresses. Port of the part of `fle_studio/hf.py` that does not need the network.
pub const HF: &str = "https://huggingface.co";

/// Python's `urllib.parse.quote(text, safe=...)`.
fn quote(text: &str, safe: &str) -> String {
    text.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"_.-~".contains(&b) || safe.as_bytes().contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

pub fn resolve_url(repo: &str, path: &str, revision: &str) -> String {
    format!(
        "{HF}/{repo}/resolve/{}/{}",
        quote(revision, ""),
        quote(path, "/")
    )
}
