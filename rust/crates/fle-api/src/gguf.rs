//! Read GGUF headers the way FLE Studio does: metadata and tensor sizes, without touching the weights. Port of
//! `fle_studio/gguf.py`. It differs from `fle_core::gguf` (the one `fle` uses) in small, visible ways: its own error texts,
//! no check that the header ends inside the file, and its own rule for which tensors are input tables. The page was built on
//! these rules, so they stay until the Python Studio is retired.
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const MAX_STRING: u64 = 16 * 1024 * 1024;
const STRING: u32 = 8;
const ARRAY: u32 = 9;

#[derive(Debug, Clone, PartialEq)]
pub struct GgufError(pub String);
impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for GgufError {}
type R<T> = Result<T, GgufError>;

fn err<T>(m: &str) -> R<T> {
    Err(GgufError(m.to_string()))
}

struct Reader<'a> {
    f: &'a mut std::fs::File,
}

impl Reader<'_> {
    fn read(&mut self, n: u64) -> R<Vec<u8>> {
        let mut b = Vec::new();
        let got = (&mut *self.f)
            .take(n)
            .read_to_end(&mut b)
            .map_err(|e| GgufError(e.to_string()))?;
        if got as u64 != n {
            return err("the file ends inside the header (incomplete download?)");
        }
        Ok(b)
    }
    fn u32(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.read(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> R<u64> {
        Ok(u64::from_le_bytes(self.read(8)?.try_into().unwrap()))
    }
    fn skip(&mut self, n: u64) -> R<()> {
        self.f
            .seek(SeekFrom::Current(n as i64))
            .map_err(|e| GgufError(e.to_string()))?;
        Ok(())
    }
    fn string(&mut self) -> R<String> {
        let n = self.u64()?;
        if n > MAX_STRING {
            return err("implausible string length in header");
        }
        Ok(String::from_utf8_lossy(&self.read(n)?).to_string())
    }
    fn scalar(&mut self, t: u32) -> R<Option<Value>> {
        let sz = match t {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            _ => return Ok(None),
        };
        let b = self.read(sz)?;
        Ok(Some(match t {
            0 => json!(b[0]),
            1 => json!(b[0] as i8),
            2 => json!(u16::from_le_bytes(b.try_into().unwrap())),
            3 => json!(i16::from_le_bytes(b.try_into().unwrap())),
            4 => json!(u32::from_le_bytes(b.try_into().unwrap())),
            5 => json!(i32::from_le_bytes(b.try_into().unwrap())),
            6 => json!(f32::from_le_bytes(b.try_into().unwrap()) as f64),
            7 => json!(b[0] != 0),
            10 => json!(u64::from_le_bytes(b.try_into().unwrap())),
            11 => json!(i64::from_le_bytes(b.try_into().unwrap())),
            _ => json!(f64::from_le_bytes(b.try_into().unwrap())),
        }))
    }
    fn value(&mut self, vtype: u32, keep: u64) -> R<Value> {
        if let Some(v) = self.scalar(vtype)? {
            return Ok(v);
        }
        if vtype == STRING {
            return Ok(json!(self.string()?));
        }
        if vtype == ARRAY {
            let (etype, count) = (self.u32()?, self.u64()?);
            let scalar_size = |t: u32| match t {
                0 | 1 | 7 => Some(1u64),
                2 | 3 => Some(2),
                4..=6 => Some(4),
                10..=12 => Some(8),
                _ => None,
            };
            if let Some(size) = scalar_size(etype) {
                if count <= keep {
                    let mut v = vec![];
                    for _ in 0..count {
                        v.push(self.scalar(etype)?.unwrap());
                    }
                    return Ok(Value::Array(v));
                }
                self.skip(size * count)?;
                return Ok(json!({"array_of": etype, "count": count}));
            }
            if etype == STRING {
                if count <= keep {
                    let mut v = vec![];
                    for _ in 0..count {
                        v.push(json!(self.string()?));
                    }
                    return Ok(Value::Array(v));
                }
                for _ in 0..count {
                    let n = self.u64()?;
                    if n > MAX_STRING {
                        return err("implausible string length in header");
                    }
                    self.skip(n)?;
                }
                return Ok(json!({"array_of": etype, "count": count}));
            }
            if etype == ARRAY {
                let mut v = vec![];
                for _ in 0..count {
                    v.push(self.value(ARRAY, keep)?);
                }
                return Ok(Value::Array(v));
            }
        }
        Err(GgufError(format!("unknown value type {vtype} in header")))
    }
}

pub struct Tensor {
    pub name: String,
    pub size: u64,
}

pub struct Header {
    pub meta: Map<String, Value>,
    pub tensors: Vec<Tensor>,
    pub file_size: u64,
}

/// Parse one GGUF file's header. Arrays longer than `keep_array` are summarised as `{array_of, count}`.
pub fn read_header(path: &Path, keep_array: u64) -> R<Header> {
    let mut f = std::fs::File::open(path).map_err(|e| GgufError(e.to_string()))?;
    let mut r = Reader { f: &mut f };
    if r.read(4)? != b"GGUF" {
        return err("not a GGUF file");
    }
    let version = r.u32()?;
    if version != 2 && version != 3 {
        return Err(GgufError(format!(
            "GGUF version {version} is not supported"
        )));
    }
    let (n_tensors, n_kv) = (r.u64()?, r.u64()?);
    if n_tensors > 2_000_000 || n_kv > 1_000_000 {
        return err("implausible header (not a GGUF file?)");
    }
    let mut meta = Map::new();
    for _ in 0..n_kv {
        let key = r.string()?;
        let t = r.u32()?;
        meta.insert(key, r.value(t, keep_array)?);
    }
    let mut raw: Vec<(String, u64)> = vec![];
    for _ in 0..n_tensors {
        let name = r.string()?;
        let nd = r.u32()?;
        if nd > 8 {
            return err("implausible tensor rank in header");
        }
        r.read(8 * nd as u64)?;
        let _ty = r.u32()?;
        raw.push((name, r.u64()?));
    }
    let align = match meta.get("general.alignment").and_then(Value::as_u64) {
        Some(a) if a > 0 => a,
        _ => 32,
    };
    let pos = f.stream_position().map_err(|e| GgufError(e.to_string()))?;
    let data_start = pos.div_ceil(align) * align;
    let file_size = std::fs::metadata(path)
        .map_err(|e| GgufError(e.to_string()))?
        .len();
    let mut order: Vec<usize> = (0..raw.len()).collect();
    order.sort_by_key(|&i| raw[i].1);
    let mut sizes = vec![0u64; raw.len()];
    for (k, &i) in order.iter().enumerate() {
        let end = if k + 1 < order.len() {
            raw[order[k + 1]].1 as i128
        } else {
            file_size as i128 - data_start as i128
        };
        sizes[i] = (end - raw[i].1 as i128).max(0) as u64;
    }
    let tensors = raw
        .into_iter()
        .enumerate()
        .map(|(i, (name, _))| Tensor {
            name,
            size: sizes[i],
        })
        .collect();
    Ok(Header {
        meta,
        tensors,
        file_size,
    })
}

/// `^(?P<base>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$`, ignoring case: (base, no, count).
pub fn split_match(name: &str) -> Option<(String, String, String)> {
    let low = name.to_lowercase();
    let stem = low.strip_suffix(".gguf")?;
    let n = stem.len();
    if n < 16 || !name.is_char_boundary(name.len() - 5 - 15) {
        return None;
    }
    let tail = &stem[n - 15..];
    let b = tail.as_bytes();
    let digits = |s: &[u8]| s.iter().all(u8::is_ascii_digit);
    if b[0] == b'-' && digits(&b[1..6]) && &tail[6..10] == "-of-" && digits(&b[10..15]) {
        let base_len = name.len() - 5 - 15;
        return Some((
            name[..base_len].to_string(),
            tail[1..6].to_string(),
            tail[10..15].to_string(),
        ));
    }
    None
}

/// All files of a split model, given any part. A single file is its own list.
pub fn split_parts(first: &Path) -> Vec<PathBuf> {
    let name = first
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    match split_match(&name) {
        None => vec![first.to_path_buf()],
        Some((base, _, count)) => {
            let c: u32 = count.parse().unwrap_or(0);
            (1..=c)
                .map(|i| first.with_file_name(format!("{base}-{i:05}-of-{c:05}.gguf")))
                .collect()
        }
    }
}

fn is_expert(n: &str) -> bool {
    ["gate_up", "gate", "up", "down"]
        .iter()
        .any(|k| n.contains(&format!(".ffn_{k}_exps.")))
}

fn kv_type_bytes(t: &str) -> f64 {
    match t {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 34.0 / 32.0,
        "q5_1" => 24.0 / 32.0,
        "q5_0" => 22.0 / 32.0,
        "q4_1" => 20.0 / 32.0,
        "q4_0" | "iq4_nl" => 18.0 / 32.0,
        _ => 2.0,
    }
}

#[derive(Clone)]
pub struct ModelInfo {
    pub arch: String,
    pub name: String,
    pub context_length: Option<i64>,
    pub layers: Option<i64>,
    pub experts: Option<i64>,
    pub experts_used: Option<i64>,
    pub expert_bytes: u64,
    pub input_bytes: u64,
    pub gpu_bytes: u64,
    pub has_chat_template: bool,
    meta: Map<String, Value>,
    pub missing: Vec<String>,
    /// sum of the file sizes of the parts that are present
    pub size: u64,
}

fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        _ => None,
    }
}

impl ModelInfo {
    pub fn is_moe(&self) -> bool {
        self.experts.unwrap_or(0) != 0 && self.expert_bytes != 0
    }

    /// Estimated KV cache bytes per context token. Hybrid models only count their full-attention layers. Like the
    /// Python original it sees scalar metadata only, so a per-layer list of kv heads falls back to the head count.
    pub fn kv_bytes_per_token(&self, type_k: &str, type_v: &str) -> f64 {
        let (m, a) = (&self.meta, &self.arch);
        let g = |k: &str| m.get(&format!("{a}.{k}"));
        let layers = g("block_count").and_then(py_int).unwrap_or(0).max(0) as usize;
        let heads = g("attention.head_count");
        let heads_kv = g("attention.head_count_kv").or(heads);
        let emb = g("embedding_length").and_then(Value::as_i64);
        let head_dim = match (emb, heads.and_then(Value::as_i64)) {
            (Some(e), Some(h)) if h != 0 => e.div_euclid(h),
            _ => 128,
        };
        let nz = |v: Option<&Value>| v.and_then(py_int).filter(|x| *x != 0);
        let k_len = nz(g("attention.key_length")).unwrap_or(head_dim);
        let v_len = nz(g("attention.value_length")).unwrap_or(k_len);
        let kvh = heads_kv.and_then(py_int).unwrap_or(0);
        let mut per_layer: Vec<i64> = vec![kvh; layers];
        if let Some(iv) = g("full_attention_interval")
            .and_then(Value::as_i64)
            .filter(|i| *i > 1)
        {
            per_layer = per_layer
                .into_iter()
                .enumerate()
                .filter(|(i, _)| (*i as i64 + 1) % iv == 0)
                .map(|(_, h)| h)
                .collect();
        }
        let (bk, bv) = (kv_type_bytes(type_k), kv_type_bytes(type_v));
        per_layer
            .iter()
            .map(|h| *h as f64 * (k_len as f64 * bk + v_len as f64 * bv))
            .sum()
    }

    pub fn public(&self) -> Value {
        json!({
            "arch": self.arch, "name": self.name, "context_length": self.context_length, "layers": self.layers, "experts": self.experts,
            "experts_used": self.experts_used, "expert_bytes": self.expert_bytes, "input_bytes": self.input_bytes, "gpu_bytes": self.gpu_bytes,
            "moe": self.is_moe(), "chat_template": self.has_chat_template, "kv_bytes_per_token_q8": self.kv_bytes_per_token("q8_0", "q8_0").round_ties_even() as i64,
        })
    }
}

/// Read every part of a (possibly split) model and add up what goes where.
pub fn inspect(first: &Path) -> R<ModelInfo> {
    let parts = split_parts(first);
    let present: Vec<&PathBuf> = parts.iter().filter(|p| p.exists()).collect();
    let missing: Vec<String> = parts
        .iter()
        .filter(|p| !p.exists())
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    if present.is_empty() || *present[0] != parts[0] {
        return err("the first part of the model is missing");
    }
    let head = read_header(&parts[0], 64)?;
    let meta: Map<String, Value> = head
        .meta
        .iter()
        .filter(|(k, v)| !v.is_array() && !v.is_object() && *k != "tokenizer.chat_template")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let arch = match head.meta.get("general.architecture") {
        Some(Value::String(s)) => s.clone(),
        Some(v) if !v.is_null() => v.to_string(),
        _ => "unknown".into(),
    };
    let (mut expert, mut inputs, mut other) = (0u64, 0u64, 0u64);
    let mut size = 0u64;
    for (i, p) in present.iter().enumerate() {
        let extra;
        let h = if i == 0 {
            &head
        } else {
            extra = read_header(p, 0)?;
            &extra
        };
        size += h.file_size;
        for t in &h.tensors {
            if is_expert(&t.name) {
                expert += t.size;
            } else if t.name.starts_with("token_embd.")
                || t.name.starts_with("per_layer_token_embd.")
            {
                inputs += t.size;
            } else {
                other += t.size;
            }
        }
    }
    let num = |k: &str| {
        head.meta
            .get(&format!("{arch}.{k}"))
            .filter(|v| v.is_number())
            .and_then(py_int)
    };
    let fname = first.file_name().unwrap().to_string_lossy().to_string();
    let name = match head.meta.get("general.name") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => {
            let base = split_match(&fname)
                .map(|(b, _, _)| b)
                .unwrap_or_else(|| fname.clone());
            base.strip_suffix(".gguf").unwrap_or(&base).to_string()
        }
    };
    Ok(ModelInfo {
        name,
        context_length: num("context_length"),
        layers: num("block_count"),
        experts: num("expert_count"),
        experts_used: num("expert_used_count"),
        expert_bytes: expert,
        input_bytes: inputs,
        gpu_bytes: other,
        has_chat_template: matches!(
            head.meta.get("tokenizer.chat_template"),
            Some(Value::String(_))
        ),
        arch,
        meta,
        missing,
        size,
    })
}

type CacheKey = (String, u64, i64);

/// `inspect` remembered per (path, size, mtime) of the first part; a model's header is read once per change.
pub fn inspect_cached(first: &Path) -> R<ModelInfo> {
    static CACHE: Mutex<Option<HashMap<CacheKey, ModelInfo>>> = Mutex::new(None);
    let md = std::fs::metadata(first).map_err(|e| GgufError(e.to_string()))?;
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let key = (first.to_string_lossy().to_string(), md.len(), mtime);
    if let Some(hit) = CACHE
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|c| c.get(&key))
        .filter(|h| h.missing.is_empty())
    {
        return Ok(hit.clone());
    }
    let info = inspect(first)?;
    CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(Default::default)
        .insert(key, info.clone());
    Ok(info)
}
