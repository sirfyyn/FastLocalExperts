//! Conversations, one JSON file each in `<data>/chats`, and the message tree. Port of `fle_studio/chats.py` and the core
//! of `chat_tree.py` (normalize, active path, text). Export, import, search and pictures come with the chat routes of stage 1.
use crate::settings::{py_str, py_truthy};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

pub const MAX_BYTES: usize = 32 * 1024 * 1024;
pub const VERSION: i64 = 2;

fn is_chat_id(s: &str) -> bool {
    s.len() == 16
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn random_hex(n: usize) -> String {
    let mut b = vec![0u8; n];
    getrandom::getrandom(&mut b).expect("random bytes");
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn node_id_ok(s: &str) -> bool {
    let n = s.chars().count();
    (1..=40).contains(&n)
        && s.chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

/// The text of a message body that may be a string or a list of OpenAI-style parts.
fn flatten(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .filter_map(|p| match p {
                Value::String(s) => Some(s.clone()),
                Value::Object(o) => o.get("text").and_then(Value::as_str).map(String::from),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

pub fn message_text(m: &Map<String, Value>) -> String {
    m.get("content").map(flatten).unwrap_or_default()
}

/// Messages with unique ids and valid parents, and a valid leaf. A plain list becomes one chain.
pub fn normalize(messages: &Value, leaf: &Value) -> (Vec<Map<String, Value>>, Option<String>) {
    let mut nodes: Vec<Map<String, Value>> = match messages {
        Value::Array(a) => a.iter().filter_map(|m| m.as_object().cloned()).collect(),
        _ => vec![],
    };
    let legacy = !nodes.iter().any(|m| m.contains_key("parent"));
    let mut seen: HashSet<String> = HashSet::new();
    for m in nodes.iter_mut() {
        let cur = m
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| node_id_ok(s) && !seen.contains(*s))
            .map(String::from);
        let id = cur.unwrap_or_else(|| loop {
            let c = random_hex(6);
            if !seen.contains(&c) {
                break c;
            }
        });
        m.insert("id".into(), Value::String(id.clone()));
        seen.insert(id);
    }
    let mut prev: Option<String> = None;
    for m in nodes.iter_mut() {
        let id = m["id"].as_str().unwrap().to_string();
        let parent = if legacy {
            prev.clone()
        } else {
            m.get("parent")
                .and_then(Value::as_str)
                .filter(|p| seen.contains(*p) && *p != id)
                .map(String::from)
        };
        m.insert(
            "parent".into(),
            parent.map(Value::String).unwrap_or(Value::Null),
        );
        prev = Some(id);
    }
    // A cycle would make the path endless: cut any node that is its own ancestor.
    let index: HashMap<String, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, m)| (m["id"].as_str().unwrap().to_string(), i))
        .collect();
    let n = nodes.len();
    for i in 0..n {
        let (mut walk, mut hops) = (i, 0usize);
        while hops <= n {
            let Some(p) = nodes[walk].get("parent").and_then(Value::as_str) else {
                break;
            };
            walk = index[p];
            hops += 1;
        }
        if hops > n {
            nodes[i].insert("parent".into(), Value::Null);
        }
    }
    let leaf = match leaf.as_str() {
        Some(l) if index.contains_key(l) => Some(l.to_string()),
        _ => nodes.last().map(|m| m["id"].as_str().unwrap().to_string()),
    };
    (nodes, leaf)
}

/// The conversation shown: from the root down to the leaf.
pub fn active_path<'a>(
    messages: &'a [Map<String, Value>],
    leaf: &Option<String>,
) -> Vec<&'a Map<String, Value>> {
    let by_id: HashMap<&str, &Map<String, Value>> = messages
        .iter()
        .filter_map(|m| m.get("id").and_then(Value::as_str).map(|i| (i, m)))
        .collect();
    let mut out = vec![];
    let mut node = leaf.as_deref().and_then(|l| by_id.get(l).copied());
    while let Some(m) = node {
        if out.len() > messages.len() {
            break;
        }
        out.push(m);
        node = m
            .get("parent")
            .and_then(Value::as_str)
            .and_then(|p| by_id.get(p).copied());
    }
    out.reverse();
    out
}

pub struct Chats {
    pub dir: PathBuf,
    pub files: PathBuf,
}

pub enum ChatError {
    /// `KeyError`: shown as 404
    NotFound,
    /// `ValueError`: shown as 400
    Invalid(String),
}

impl Chats {
    pub fn new(data_dir: &Path) -> Chats {
        let dir = data_dir.join("chats");
        let files = dir.join("files");
        let _ = std::fs::create_dir_all(&files);
        Chats { dir, files }
    }

    fn path(&self, id: &str) -> Result<PathBuf, ChatError> {
        if !is_chat_id(id) {
            return Err(ChatError::NotFound);
        }
        Ok(self.dir.join(format!("{id}.json")))
    }

    fn read(path: &Path) -> Option<Map<String, Value>> {
        match serde_json::from_str::<Value>(&std::fs::read_to_string(path).ok()?) {
            Ok(Value::Object(m)) => Some(m),
            _ => None,
        }
    }

    pub fn list(&self) -> Value {
        let mut out: Vec<(f64, Value)> = vec![];
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return json!([]);
        };
        let mut paths: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        paths.sort();
        for p in paths {
            let Some(d) = Self::read(&p) else { continue };
            let (nodes, leaf) = normalize(
                d.get("messages").unwrap_or(&Value::Null),
                d.get("leaf").unwrap_or(&Value::Null),
            );
            let updated = d.get("updated").cloned().unwrap_or(json!(0));
            let title = match d.get("title") {
                Some(t) if py_truthy(t) => t.clone(),
                _ => json!("New chat"),
            };
            let stem = p
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            out.push((
                updated.as_f64().unwrap_or(0.0),
                json!({"id": d.get("id").cloned().unwrap_or(json!(stem)), "title": title, "updated": updated,
                       "model": d.get("model").cloned().unwrap_or(json!("")), "messages": active_path(&nodes, &leaf).len()}),
            ));
        }
        // newest first; equal times keep their order (Python's sort with reverse=True is stable)
        out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Value::Array(out.into_iter().map(|x| x.1).collect())
    }

    pub fn get(&self, id: &str) -> Result<Value, ChatError> {
        let mut data = Self::read(&self.path(id)?).ok_or(ChatError::NotFound)?;
        let (nodes, leaf) = normalize(
            data.get("messages").unwrap_or(&Value::Null),
            data.get("leaf").unwrap_or(&Value::Null),
        );
        data.insert(
            "messages".into(),
            Value::Array(nodes.into_iter().map(Value::Object).collect()),
        );
        data.insert(
            "leaf".into(),
            leaf.map(Value::String).unwrap_or(Value::Null),
        );
        Ok(Value::Object(data))
    }

    pub fn save(&self, id: Option<&str>, data: &Map<String, Value>) -> Result<Value, ChatError> {
        let ident = id.map(String::from).unwrap_or_else(|| random_hex(8));
        let path = self.path(&ident)?;
        let messages = data
            .get("messages")
            .filter(|m| m.is_array())
            .ok_or_else(|| ChatError::Invalid("messages must be a list".into()))?;
        let (nodes, leaf) = normalize(messages, data.get("leaf").unwrap_or(&Value::Null));
        let shown = active_path(&nodes, &leaf);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let s = |k: &str| {
            data.get(k)
                .filter(|v| py_truthy(v))
                .map(py_str)
                .unwrap_or_default()
        };
        let mut title = s("title").trim().to_string();
        if title.is_empty() {
            let first = shown
                .iter()
                .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
                .map(|m| message_text(m))
                .unwrap_or_default();
            let t: String = first
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .chars()
                .take(60)
                .collect();
            title = if t.is_empty() { "New chat".into() } else { t };
        }
        let created = data
            .get("created")
            .filter(|v| py_truthy(v))
            .cloned()
            .unwrap_or(json!(now));
        let chat = json!({
            "id": ident, "version": VERSION, "title": title.chars().take(120).collect::<String>(), "created": created, "updated": now,
            "model": s("model"), "system": s("system"), "leaf": leaf, "messages": nodes,
        });
        let text = serde_json::to_string(&chat).unwrap();
        if text.len() > MAX_BYTES {
            return Err(ChatError::Invalid("chat too large".into()));
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, text)
            .and_then(|_| std::fs::rename(&tmp, &path))
            .map_err(|e| ChatError::Invalid(e.to_string()))?;
        Ok(chat)
    }

    pub fn delete(&self, id: &str) -> Result<(), ChatError> {
        let _ = std::fs::remove_file(self.path(id)?);
        Ok(())
    }
}
