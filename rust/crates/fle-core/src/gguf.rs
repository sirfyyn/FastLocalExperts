//! Read GGUF headers without loading weights. Port of `tools/fle/gguf.py` (the Python helper is the specification;
//! `tests/gguf.rs` and the conformance check compare against it).
use anyhow::{bail, Result};
use serde_json::{json, Map, Value as Json};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAX_STRING: u64 = 32 * 1024 * 1024;
const MAX_ARRAY_KEEP: u64 = 32;

/// (name, block size, bytes per block, extra bytes at the start of every row)
type Info = (&'static str, u64, u64, u64);

fn base_type(id: u32) -> Option<Info> {
    Some(match id {
        0 => ("F32", 1, 4, 0),
        1 => ("F16", 1, 2, 0),
        2 => ("Q4_0", 32, 18, 0),
        3 => ("Q4_1", 32, 20, 0),
        6 => ("Q5_0", 32, 22, 0),
        7 => ("Q5_1", 32, 24, 0),
        8 => ("Q8_0", 32, 34, 0),
        9 => ("Q8_1", 32, 40, 0),
        10 => ("Q2_K", 256, 84, 0),
        11 => ("Q3_K", 256, 110, 0),
        12 => ("Q4_K", 256, 144, 0),
        13 => ("Q5_K", 256, 176, 0),
        14 => ("Q6_K", 256, 210, 0),
        15 => ("Q8_K", 256, 292, 0),
        16 => ("IQ2_XXS", 256, 66, 0),
        17 => ("IQ2_XS", 256, 74, 0),
        18 => ("IQ3_XXS", 256, 98, 0),
        19 => ("IQ1_S", 256, 50, 0),
        20 => ("IQ4_NL", 32, 18, 0),
        21 => ("IQ3_S", 256, 110, 0),
        22 => ("IQ2_S", 256, 82, 0),
        23 => ("IQ4_XS", 256, 136, 0),
        24 => ("I8", 1, 1, 0),
        25 => ("I16", 1, 2, 0),
        26 => ("I32", 1, 4, 0),
        27 => ("I64", 1, 8, 0),
        28 => ("F64", 1, 8, 0),
        29 => ("IQ1_M", 256, 56, 0),
        30 => ("BF16", 1, 2, 0),
        34 => ("TQ1_0", 256, 54, 0),
        35 => ("TQ2_0", 256, 66, 0),
        39 => ("MXFP4", 32, 17, 0),
        40 => ("NVFP4", 64, 36, 0),
        41 => ("Q1_0", 128, 18, 0),
        42 => ("Q2_0", 64, 18, 0),
        _ => return None,
    })
}

/// The FLE release and the DeepSeek41 port disagree on the last ids; the offsets decide which table is meant.
pub const LAYOUTS: [&str; 2] = ["release", "deepseek41"];

pub fn type_info(id: u32, layout: &str) -> Option<Info> {
    if let Some(i) = base_type(id) {
        return Some(i);
    }
    match (layout, id) {
        ("deepseek41", 43) => Some(("MXFP8", 32, 33, 0)),
        ("deepseek41", 44) => Some(("IQ2_KT", 256, 68, 4)),
        ("deepseek41", 45) => Some(("IQ3_KT", 256, 96, 4)),
        ("deepseek41", 46) => Some(("IQ4_KT", 256, 128, 0)),
        (_, 43) => Some(("IQ2_KT", 256, 68, 4)),
        (_, 44) => Some(("IQ3_KT", 256, 96, 4)),
        (_, 45) => Some(("IQ4_KT", 256, 128, 0)),
        _ => None,
    }
}

pub fn type_name(id: u32, layout: &str) -> String {
    type_info(id, layout)
        .map(|i| i.0.to_string())
        .unwrap_or_else(|| format!("type-{id}"))
}

#[derive(Debug, Clone)]
pub enum Val {
    Int(i128),
    Float(f64),
    Bool(bool),
    Str(String),
    List(Vec<Val>),
    Big { array_of: u32, count: u64 },
}

impl Val {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Val::Int(i) if *i >= 0 => Some(*i as u64),
            Val::Float(f) if *f >= 0.0 => Some(*f as u64),
            _ => None,
        }
    }
    fn json(&self) -> Option<Json> {
        match self {
            Val::Int(i) => Some(json!(*i as i64)),
            Val::Float(f) => Some(json!(f)),
            Val::Bool(b) => Some(json!(b)),
            Val::Str(s) => Some(json!(s)),
            _ => None, // lists and big arrays are not part of the summary's scalar meta
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub type_id: u32,
    pub offset: u64,
    pub nbytes: u64,
    pub type_name: String,
}

pub fn tensor_nbytes(shape: &[u64], type_id: u32, layout: &str) -> u64 {
    let Some((_, block, size, row_meta)) = type_info(type_id, layout) else {
        return 0;
    };
    if shape.is_empty() || shape.contains(&0) || block == 0 {
        return 0;
    }
    if block == 1 {
        return size * shape.iter().product::<u64>();
    }
    if !shape[0].is_multiple_of(block) {
        return 0;
    }
    let row = row_meta + size * (shape[0] / block);
    row * if shape.len() > 1 {
        shape[1..].iter().product::<u64>()
    } else {
        1
    }
}

#[derive(Debug, Clone)]
pub struct Shard {
    pub path: PathBuf,
    pub version: u32,
    pub file_size: u64,
    pub data_start: u64,
    pub tensors: Vec<Tensor>,
    pub meta: Vec<(String, Val)>,
    pub layout: String,
    pub expected_end: u64,
}

impl Shard {
    pub fn payload(&self) -> u64 {
        self.file_size.saturating_sub(self.data_start)
    }
    pub fn complete(&self) -> bool {
        self.expected_end <= self.payload()
    }
}

struct R<T: Read + Seek>(T);

impl<T: Read + Seek> R<T> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut b = vec![0u8; n];
        self.0.read_exact(&mut b).map_err(|_| {
            anyhow::anyhow!("the file ends inside the header (incomplete download?)")
        })?;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > MAX_STRING {
            bail!("implausible string length in header");
        }
        Ok(String::from_utf8_lossy(&self.bytes(n as usize)?).to_string())
    }
    fn skip(&mut self, n: u64) -> Result<()> {
        self.0.seek(SeekFrom::Current(n as i64))?;
        Ok(())
    }
    fn scalar_size(t: u32) -> Option<u64> {
        Some(match t {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            _ => return None,
        })
    }
    fn scalar(&mut self, t: u32) -> Result<Val> {
        let n = Self::scalar_size(t).unwrap() as usize;
        let b = self.bytes(n)?;
        Ok(match t {
            0 => Val::Int(b[0] as i128),
            1 => Val::Int(b[0] as i8 as i128),
            7 => Val::Bool(b[0] != 0),
            2 => Val::Int(u16::from_le_bytes(b.try_into().unwrap()) as i128),
            3 => Val::Int(i16::from_le_bytes(b.try_into().unwrap()) as i128),
            4 => Val::Int(u32::from_le_bytes(b.try_into().unwrap()) as i128),
            5 => Val::Int(i32::from_le_bytes(b.try_into().unwrap()) as i128),
            6 => Val::Float(f32::from_le_bytes(b.try_into().unwrap()) as f64),
            10 => Val::Int(u64::from_le_bytes(b.try_into().unwrap()) as i128),
            11 => Val::Int(i64::from_le_bytes(b.try_into().unwrap()) as i128),
            _ => Val::Float(f64::from_le_bytes(b.try_into().unwrap())),
        })
    }
    fn value(&mut self, t: u32) -> Result<Val> {
        if Self::scalar_size(t).is_some() {
            return self.scalar(t);
        }
        if t == 8 {
            return Ok(Val::Str(self.string()?));
        }
        if t != 9 {
            bail!("unknown value type {t} in header");
        }
        let (et, count) = (self.u32()?, self.u64()?);
        if count > 100_000_000 {
            bail!("implausible array length in header");
        }
        let keep = count <= MAX_ARRAY_KEEP;
        match et {
            8 => {
                if keep {
                    return Ok(Val::List(
                        (0..count)
                            .map(|_| self.string().map(Val::Str))
                            .collect::<Result<_>>()?,
                    ));
                }
                for _ in 0..count {
                    let n = self.u64()?;
                    if n > MAX_STRING {
                        bail!("implausible string length in header");
                    }
                    self.skip(n)?;
                }
                Ok(Val::Big {
                    array_of: et,
                    count,
                })
            }
            9 => {
                if keep {
                    return Ok(Val::List(
                        (0..count).map(|_| self.value(9)).collect::<Result<_>>()?,
                    ));
                }
                for _ in 0..count {
                    self.value(9)?;
                }
                Ok(Val::Big {
                    array_of: et,
                    count,
                })
            }
            t2 if Self::scalar_size(t2).is_some() => {
                if keep {
                    return Ok(Val::List(
                        (0..count).map(|_| self.scalar(t2)).collect::<Result<_>>()?,
                    ));
                }
                self.skip(Self::scalar_size(t2).unwrap() * count)?;
                Ok(Val::Big {
                    array_of: et,
                    count,
                })
            }
            _ => bail!("unknown value type {et} in header"),
        }
    }
}

pub fn read_shard(path: &Path) -> Result<Shard> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let mut r = R(BufReader::new(File::open(path)?));
    if r.bytes(4)? != b"GGUF" {
        bail!("{name}: not a GGUF file");
    }
    let version = r.u32()?;
    if version != 2 && version != 3 {
        bail!("{name}: GGUF version {version} is not supported");
    }
    let (n_tensors, n_kv) = (r.u64()?, r.u64()?);
    if n_tensors > 2_000_000 || n_kv > 1_000_000 {
        bail!("{name}: implausible header");
    }
    let mut meta = Vec::new();
    for _ in 0..n_kv {
        let k = r.string()?;
        let t = r.u32()?;
        meta.push((k, r.value(t)?));
    }
    let mut raw = Vec::new();
    for _ in 0..n_tensors {
        let tname = r.string()?;
        let nd = r.u32()?;
        if nd > 8 {
            bail!("{name}: implausible tensor rank");
        }
        let dims = (0..nd).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
        raw.push((tname, dims, r.u32()?, r.u64()?));
    }
    let pos = r.0.stream_position()?;
    let align = meta
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| v.as_u64())
        .filter(|a| *a > 0)
        .unwrap_or(32);
    let data_start = pos.div_ceil(align) * align;
    let file_size = std::fs::metadata(path)?.len();
    if data_start > file_size {
        bail!("{name}: header ends behind the end of the file");
    }
    let tensors = raw
        .into_iter()
        .map(|(name, shape, type_id, offset)| Tensor {
            name,
            shape,
            type_id,
            offset,
            nbytes: 0,
            type_name: String::new(),
        })
        .collect();
    let mut s = Shard {
        path: path.to_path_buf(),
        version,
        file_size,
        data_start,
        tensors,
        meta,
        layout: "release".into(),
        expected_end: 0,
    };
    s.layout = detect_layout(std::slice::from_ref(&s), &s.meta);
    apply_layout(&mut s);
    Ok(s)
}

fn apply_layout(s: &mut Shard) {
    for t in &mut s.tensors {
        t.type_name = type_name(t.type_id, &s.layout);
        t.nbytes = tensor_nbytes(&t.shape, t.type_id, &s.layout);
    }
    s.expected_end = s
        .tensors
        .iter()
        .map(|t| t.offset + t.nbytes)
        .max()
        .unwrap_or(0);
}

fn layout_score(shards: &[Shard], layout: &str) -> f64 {
    let mut score = 0.0;
    for s in shards {
        if s.tensors
            .iter()
            .any(|t| type_info(t.type_id, layout).is_none())
        {
            return f64::INFINITY;
        }
        let mut iv: Vec<(u64, u64)> = Vec::new();
        let mut end = 0;
        for t in &s.tensors {
            let n = tensor_nbytes(&t.shape, t.type_id, layout);
            if n == 0 || t.offset + n > s.payload() {
                return f64::INFINITY;
            }
            iv.push((t.offset, t.offset + n));
            end = end.max(t.offset + n);
        }
        iv.sort();
        if iv.windows(2).any(|w| w[1].0 < w[0].1) {
            return f64::INFINITY;
        }
        score += (s.payload() as f64 - end as f64).abs();
    }
    score
}

pub fn detect_layout(shards: &[Shard], meta: &[(String, Val)]) -> String {
    if !shards
        .iter()
        .any(|s| s.tensors.iter().any(|t| t.type_id >= 43))
    {
        let arch = meta
            .iter()
            .find(|(k, _)| k == "general.architecture")
            .map(|(_, v)| match v {
                Val::Str(s) => s.as_str(),
                _ => "",
            })
            .unwrap_or("");
        return if arch == "deepseek41" {
            "deepseek41"
        } else {
            "release"
        }
        .into();
    }
    LAYOUTS
        .iter()
        .map(|l| (*l, layout_score(shards, l)))
        .filter(|(_, s)| s.is_finite())
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(l, _)| l.to_string())
        .unwrap_or_else(|| "release".into())
}

/// `name-00002-of-00005.gguf` -> ("name", 2, 5)
fn split_name(n: &str) -> Option<(String, u32, u32)> {
    let s = n
        .strip_suffix(".gguf")
        .or_else(|| n.strip_suffix(".GGUF"))?;
    let (rest, count) = s.rsplit_once("-of-")?;
    let (base, no) = rest.rsplit_once('-')?;
    if no.len() != 5 || count.len() != 5 {
        return None;
    }
    Some((base.to_string(), no.parse().ok()?, count.parse().ok()?))
}

pub fn split_parts(first: &Path) -> Result<Vec<PathBuf>> {
    let n = first.file_name().unwrap().to_string_lossy().to_string();
    let Some((base, _, count)) = split_name(&n) else {
        return Ok(vec![first.to_path_buf()]);
    };
    if count > 1000 {
        bail!("{n}: implausible split count {count}");
    }
    Ok((1..=count)
        .map(|i| first.with_file_name(format!("{base}-{i:05}-of-{count:05}.gguf")))
        .collect())
}

fn is_sidecar(name: &str) -> bool {
    let l = name.to_lowercase();
    ["mmproj", "mtp", "draft"]
        .iter()
        .any(|k| l.split(|c: char| "-_.".contains(c)).any(|w| w == *k))
}

pub fn resolve_model(path: &Path) -> Result<PathBuf> {
    let mut p = if let Some(h) = path.to_string_lossy().strip_prefix("~/") {
        dirs::home_dir().unwrap_or_default().join(h)
    } else {
        path.to_path_buf()
    };
    if p.is_dir() {
        let mut c: Vec<PathBuf> = std::fs::read_dir(&p)?
            .flatten()
            .map(|e| e.path())
            .filter(|q| {
                q.extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                    && !is_sidecar(&q.file_name().unwrap().to_string_lossy())
            })
            .collect();
        let mut firsts: Vec<PathBuf> = c
            .iter()
            .filter(|q| {
                split_name(&q.file_name().unwrap().to_string_lossy())
                    .is_none_or(|(_, no, _)| no == 1)
            })
            .cloned()
            .collect();
        if firsts.is_empty() {
            firsts = std::mem::take(&mut c);
        }
        if firsts.is_empty() {
            bail!(
                "No GGUF model in {}. Pass the model file or a folder that contains it.",
                p.display()
            );
        }
        firsts
            .sort_by_key(|q| std::cmp::Reverse(std::fs::metadata(q).map(|m| m.len()).unwrap_or(0)));
        p = firsts.remove(0);
    }
    if let Some((base, no, count)) = split_name(&p.file_name().unwrap().to_string_lossy()) {
        if no != 1 {
            p = p.with_file_name(format!("{base}-00001-of-{count:05}.gguf"));
        }
    }
    if !p.exists() {
        bail!(
            "No model at {}. Check the path, or start the download again.",
            p.display()
        );
    }
    if !p
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    {
        bail!("{} is not a .gguf file", p.display());
    }
    Ok(p)
}

#[derive(Debug)]
pub struct Model {
    pub first: PathBuf,
    pub parts: Vec<PathBuf>,
    pub missing: Vec<PathBuf>,
    pub shards: Vec<Shard>,
    pub meta: Vec<(String, Val)>,
    pub layout: String,
    pub arch: String,
    pub name: String,
    pub size: u64,
    pub tensor_count: usize,
    pub expert_bytes: u64,
    pub input_bytes: u64,
    pub gpu_bytes: u64,
    pub type_histogram: Vec<(String, (u64, u64))>,
    pub unknown_types: Vec<u32>,
    pub incomplete: Vec<String>,
}

fn is_expert(n: &str) -> bool {
    n.contains(".ffn_")
        && ["gate_up", "gate", "up", "down"]
            .iter()
            .any(|k| n.contains(&format!(".ffn_{k}_exps.")))
}
fn is_input(n: &str) -> bool {
    n.split('.').any(|w| {
        [
            "token_embd",
            "per_layer_token_embd",
            "engram",
            "engram_embd",
            "ple",
        ]
        .contains(&w)
    })
}

pub fn read_model(path: &Path) -> Result<Model> {
    let first = resolve_model(path)?;
    let parts = split_parts(&first)?;
    let (present, missing): (Vec<_>, Vec<_>) = parts.iter().cloned().partition(|p| p.exists());
    if !present.contains(&first) {
        bail!(
            "The first part is missing: {}. Wait for the download to finish or pick another model.",
            first.display()
        );
    }
    let mut shards = present
        .iter()
        .map(|p| read_shard(p))
        .collect::<Result<Vec<_>>>()?;
    let mut meta: Vec<(String, Val)> = Vec::new();
    for s in &shards {
        for (k, v) in &s.meta {
            if !meta.iter().any(|(k2, _)| k2 == k) {
                meta.push((k.clone(), v.clone()));
            }
        }
    }
    let layout = detect_layout(&shards, &meta);
    for s in &mut shards {
        s.layout = layout.clone();
        apply_layout(s);
    }
    let get = |k: &str| meta.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.clone());
    let arch = match get("general.architecture") {
        Some(Val::Str(s)) => s,
        _ => "unknown".into(),
    };
    let (mut expert, mut inputs, mut other) = (0u64, 0u64, 0u64);
    let mut hist: Vec<(String, (u64, u64))> = Vec::new();
    let mut unknown = std::collections::BTreeSet::new();
    let mut seen = std::collections::HashSet::new();
    for s in &shards {
        for t in &s.tensors {
            if !seen.insert(t.name.clone()) {
                continue;
            }
            let have = t.nbytes.min(s.payload().saturating_sub(t.offset));
            if t.nbytes == 0 {
                unknown.insert(t.type_id);
            }
            if is_expert(&t.name) {
                expert += have
            } else if is_input(&t.name) {
                inputs += have
            } else {
                other += have
            }
            let at = match hist.iter().position(|(n, _)| *n == t.type_name) {
                Some(i) => i,
                None => {
                    hist.push((t.type_name.clone(), (0, 0)));
                    hist.len() - 1
                }
            };
            hist[at].1 .0 += 1;
            hist[at].1 .1 += have;
        }
    }
    let name = match get("general.name") {
        Some(Val::Str(s)) if !s.is_empty() => s,
        _ => {
            let n = first.file_name().unwrap().to_string_lossy().to_string();
            let n = n.trim_end_matches(".gguf").to_string();
            split_name(&format!("{n}.gguf")).map(|x| x.0).unwrap_or(n)
        }
    };
    Ok(Model {
        first,
        parts,
        missing,
        meta,
        layout,
        arch,
        name,
        size: shards.iter().map(|s| s.file_size).sum(),
        tensor_count: seen.len(),
        expert_bytes: expert,
        input_bytes: inputs,
        gpu_bytes: other,
        type_histogram: hist,
        unknown_types: unknown.into_iter().collect(),
        incomplete: shards
            .iter()
            .filter(|s| !s.complete())
            .map(|s| s.path.file_name().unwrap().to_string_lossy().to_string())
            .collect(),
        shards,
    })
}

impl Model {
    pub fn complete(&self) -> bool {
        self.missing.is_empty() && self.incomplete.is_empty()
    }
    fn num(&self, key: &str) -> Option<u64> {
        let full = format!("{}.{key}", self.arch);
        self.meta
            .iter()
            .find(|(k, _)| *k == full)
            .and_then(|(_, v)| match v {
                Val::Int(_) | Val::Float(_) => v.as_u64(),
                _ => None,
            })
    }
    fn mf(&self, key: &str) -> Option<&Val> {
        let full = format!("{}.{key}", self.arch);
        self.meta.iter().find(|(k, _)| *k == full).map(|(_, v)| v)
    }

    /// KV cache bytes per token for the given cache types (same rule as the Python helper).
    pub fn kv_bytes_per_token(&self, type_k: &str, type_v: &str) -> f64 {
        let layers = self.num("block_count").unwrap_or(0) as usize;
        let heads = self.mf("attention.head_count").and_then(|v| v.as_u64());
        let heads_kv: Vec<u64> = match self
            .mf("attention.head_count_kv")
            .or(self.mf("attention.head_count"))
        {
            Some(Val::List(l)) => l.iter().map(|v| v.as_u64().unwrap_or(0)).collect(),
            Some(v) => vec![v.as_u64().unwrap_or(0); layers],
            None => vec![0; layers],
        };
        let emb = self.num("embedding_length");
        let head_dim = match (emb, heads) {
            (Some(e), Some(h)) if h > 0 => e / h,
            _ => 128,
        };
        let k_len = self
            .num("attention.key_length")
            .filter(|v| *v > 0)
            .unwrap_or(head_dim);
        let v_len = self
            .num("attention.value_length")
            .filter(|v| *v > 0)
            .unwrap_or(k_len);
        let mut per: Vec<u64> = heads_kv;
        if let Some(iv) = self.num("full_attention_interval").filter(|i| *i > 1) {
            per = per
                .iter()
                .enumerate()
                .filter(|(i, _)| (*i as u64 + 1).is_multiple_of(iv))
                .map(|(_, h)| *h)
                .collect();
        }
        let b = |t: &str| match t {
            "f32" => 4.0,
            "q8_0" => 34.0 / 32.0,
            "q5_1" => 24.0 / 32.0,
            "q5_0" => 22.0 / 32.0,
            "q4_1" => 20.0 / 32.0,
            "q4_0" | "iq4_nl" => 18.0 / 32.0,
            _ => 2.0,
        };
        per.iter()
            .map(|h| *h as f64 * (k_len as f64 * b(type_k) + v_len as f64 * b(type_v)))
            .sum()
    }

    /// Same keys as `Model.summary()` in the Python helper.
    /// Same keys, same order as `inspect --json` in the Python helper.
    pub fn summary(&self) -> Json {
        let meta: Map<String, Json> = self
            .meta
            .iter()
            .filter_map(|(k, v)| v.json().map(|j| (k.clone(), j)))
            .collect();
        let hist: Map<String, Json> = self
            .type_histogram
            .iter()
            .map(|(k, (n, b))| (k.clone(), json!({"tensors": n, "bytes": b})))
            .collect();
        let file_type = match self.meta.iter().find(|(k, _)| k == "general.file_type") {
            Some((_, Val::Int(i))) => json!(*i as i64),
            _ => Json::Null,
        };
        let opt = |v: Option<u64>| v.map(|x| json!(x)).unwrap_or(Json::Null);
        let names = |v: &[PathBuf]| {
            v.iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
                .collect::<Vec<_>>()
        };
        json!({
            "path": self.first.to_string_lossy(), "name": self.name, "arch": self.arch, "layout": self.layout, "complete": self.complete(),
            "missing": names(&self.missing), "incomplete": self.incomplete, "size": self.size, "size_human": human_bytes(self.size),
            "parts": names(&self.parts), "shards_present": self.shards.len(), "tensor_count": self.tensor_count,
            "context_length": opt(self.num("context_length")), "layers": opt(self.num("block_count")),
            "experts": opt(self.num("expert_count")), "experts_used": opt(self.num("expert_used_count")),
            "expert_bytes": self.expert_bytes, "expert_bytes_human": human_bytes(self.expert_bytes), "input_bytes": self.input_bytes,
            "gpu_bytes": self.gpu_bytes, "gpu_bytes_human": human_bytes(self.gpu_bytes), "file_type": file_type,
            "kv_bytes_per_token_q8": self.kv_bytes_per_token("q8_0", "q8_0").round() as u64,
            "kv_bytes_per_token_f16": self.kv_bytes_per_token("f16", "f16").round() as u64,
            "type_histogram": hist, "unknown_types": self.unknown_types, "meta": meta, "sidecars": sidecars(&self.first),
        })
    }
}

/// Same text as `human_bytes` in the Python helper: "770.3 MiB", "512 B".
pub fn human_bytes(n: u64) -> String {
    let mut v = n as f64;
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"] {
        if v.abs() < 1024.0 || unit == "TiB" {
            return if unit == "B" {
                format!("{} B", v as u64)
            } else {
                format!("{v:.1} {unit}")
            };
        }
        v /= 1024.0;
    }
    format!("{v:.1} TiB")
}

/// MTP head, MTP vocabulary and mmproj next to the model (same rules as `sidecars()` in the Python helper).
pub fn sidecars(first: &Path) -> Json {
    let folder = first.parent().unwrap_or(Path::new("."));
    let mut names: Vec<PathBuf> = std::fs::read_dir(folder)
        .map(|r| {
            r.flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    let low = |p: &Path| p.file_name().unwrap().to_string_lossy().to_lowercase();
    let ext = |p: &Path, e: &str| p.extension().is_some_and(|x| x.eq_ignore_ascii_case(e));
    let find = |f: &dyn Fn(&PathBuf) -> bool| {
        names
            .iter()
            .find(|p| f(p))
            .map(|p| json!(p.to_string_lossy()))
            .unwrap_or(Json::Null)
    };
    json!({
        "mtp_head": find(&|p| is_sidecar(&p.file_name().unwrap().to_string_lossy()) && low(p).contains("mtp") && ext(p, "gguf")),
        "mtp_vocab": find(&|p| low(p).contains("mtp") && low(p).contains("vocab") && ext(p, "txt")),
        "mmproj": find(&|p| low(p).contains("mmproj") && ext(p, "gguf")),
    })
}

/// Parts, tensor count, offsets and types; same checks, texts and order as `verify_model` in the Python helper.
pub fn verify_model(m: &Model) -> Json {
    let (mut errors, mut warnings, mut checks): (Vec<String>, Vec<String>, Vec<Json>) =
        (vec![], vec![], vec![]);
    let mut add = |name: &str, ok: Option<bool>, detail: String, hint: &str| {
        let status = match ok {
            None => "warn",
            Some(true) => "ok",
            Some(false) => "fail",
        };
        checks.push(json!({"name": name, "status": status, "detail": detail, "hint": hint}));
    };
    let fname = |p: &Path| p.file_name().unwrap().to_string_lossy().to_string();
    add(
        "first part",
        Some(m.first.exists()),
        m.first.to_string_lossy().to_string(),
        "Check the path.",
    );
    if !m.missing.is_empty() || !m.incomplete.is_empty() {
        let mut d = format!("{} of {} parts missing", m.missing.len(), m.parts.len());
        if !m.incomplete.is_empty() {
            d += &format!(", {} incomplete", m.incomplete.len());
        }
        if !m.missing.is_empty() {
            errors.push(format!(
                "Missing model parts: {}",
                m.missing
                    .iter()
                    .take(6)
                    .map(|p| fname(p))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !m.incomplete.is_empty() {
            errors.push(format!(
                "Incomplete model parts: {}",
                m.incomplete
                    .iter()
                    .take(6)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        add(
            "all parts",
            Some(false),
            d,
            "Wait for the download to finish, then run verify again.",
        );
    } else {
        add(
            "all parts",
            Some(true),
            format!("{} part(s) present", m.parts.len()),
            "",
        );
    }
    let mut split_counts = std::collections::BTreeSet::new();
    let mut archs = std::collections::BTreeSet::new();
    let meta_of = |s: &Shard, k: &str| {
        s.meta
            .iter()
            .find(|(kk, _)| kk == k)
            .map(|(_, v)| v.clone())
    };
    for (index, s) in m.shards.iter().enumerate() {
        let name = fname(&s.path);
        if let Some(Val::Int(no)) = meta_of(s, "split.no") {
            if no != index as i128 {
                warnings.push(format!("{name}: split.no is {no}, expected {index}"));
            }
        }
        if let Some(Val::Int(c)) = meta_of(s, "split.count") {
            split_counts.insert(c);
        }
        if let Some(Val::Str(a)) = meta_of(s, "general.architecture") {
            archs.insert(a);
        }
        let mut iv: Vec<(u64, u64, &str)> = s
            .tensors
            .iter()
            .map(|t| (t.offset, t.offset + t.nbytes, t.name.as_str()))
            .collect();
        iv.sort();
        let mut overlap = false;
        for w in iv.windows(2) {
            if w[1].0 < w[0].1 {
                overlap = true;
                errors.push(format!("{name}: tensors {} and {} overlap", w[0].2, w[1].2));
            }
        }
        if !s.complete() {
            let expected = s.data_start + s.expected_end;
            warnings.push(format!(
                "{name}: still downloading or truncated ({} of about {})",
                human_bytes(s.file_size),
                human_bytes(expected)
            ));
            add(
                &format!("{name} layout"),
                Some(false),
                format!("{} tensors, file is incomplete", s.tensors.len()),
                "",
            );
            continue;
        }
        let out: Vec<_> = s
            .tensors
            .iter()
            .filter(|t| t.offset + t.nbytes > s.payload())
            .collect();
        if !out.is_empty() {
            errors.push(format!(
                "{name}: {} tensor(s) end behind the file",
                out.len()
            ));
        }
        if overlap || !out.is_empty() {
            add(
                &format!("{name} layout"),
                Some(false),
                format!("{} tensors, offsets do not fit", s.tensors.len()),
                "",
            );
        } else {
            add(
                &format!("{name} layout"),
                Some(true),
                format!(
                    "{} tensors, {} payload",
                    s.tensors.len(),
                    human_bytes(s.payload())
                ),
                "",
            );
        }
    }
    if split_counts.len() > 1 {
        errors.push(format!(
            "Shards disagree about split.count: {:?}",
            split_counts.iter().collect::<Vec<_>>()
        ));
    }
    if archs.len() > 1 {
        errors.push(format!(
            "Shards disagree about the architecture: {:?}",
            archs.iter().collect::<Vec<_>>()
        ));
    }
    let expected = m
        .shards
        .iter()
        .find_map(|s| match meta_of(s, "split.tensors.count") {
            Some(Val::Int(i)) => Some(i as usize),
            _ => None,
        });
    if let Some(e) = expected {
        if m.complete() && m.tensor_count != e {
            errors.push(format!(
                "Header says {e} tensors; the shards contain {}",
                m.tensor_count
            ));
        }
        add(
            "tensor count",
            Some(m.tensor_count == e || !m.complete()),
            format!(
                "{} in the files, {e} in split.tensors.count",
                m.tensor_count
            ),
            "",
        );
    }
    if !m.unknown_types.is_empty() {
        errors.push(format!(
            "Unknown or unsupported tensor types: {:?}",
            m.unknown_types
        ));
        add(
            "tensor types",
            Some(false),
            format!("unknown ids {:?}", m.unknown_types),
            "Use an engine that knows these types, or repair the file.",
        );
    } else {
        add(
            "tensor types",
            Some(true),
            format!("{} types, all known", m.type_histogram.len()),
            "",
        );
    }
    let (ex, used) = (m.num("expert_count"), m.num("expert_used_count"));
    if m.expert_bytes > 0
        && ex.is_some()
        && matches!((used, ex), (Some(u), Some(e)) if u > 0 && u > e)
    {
        errors.push("expert_used_count is larger than expert_count".into());
    }
    let side = sidecars(&m.first);
    if !side["mtp_head"].is_null() && side["mtp_vocab"].is_null() {
        warnings.push(
            "An MTP head is present but no MTP vocabulary file was found; MTP stays off.".into(),
        );
    }
    json!({"ok": errors.is_empty(), "errors": errors, "warnings": warnings, "checks": checks, "layout": m.layout, "tensor_count": m.tensor_count,
           "parts": m.parts.len(), "present": m.shards.len(), "size": m.size, "sidecars": side})
}

/// Pretty JSON exactly as Python's `json.dumps(indent=2)` writes it. serde writes `9.99e-6` and `1e16`, Python `9.99e-06`
/// and `1e+16`; only the exponent of a number differs, so only exponents followed by a delimiter are rewritten.
pub fn py_json(v: &Json) -> String {
    let t = serde_json::to_string_pretty(v).unwrap_or_default();
    let b: Vec<char> = t.chars().collect();
    let mut out = String::with_capacity(t.len() + 8);
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        out.push(c);
        i += 1;
        if c == 'e' && i >= 2 && b[i - 2].is_ascii_digit() {
            let mut j = i;
            let neg = b.get(j) == Some(&'-');
            if neg {
                j += 1;
            }
            let start = j;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            let end_ok = j == b.len() || matches!(b[j], ',' | '\n' | ' ' | '}' | ']');
            if j > start && end_ok {
                let digits: String = b[start..j].iter().collect();
                out.push(if neg { '-' } else { '+' });
                if digits.len() < 2 {
                    out.push('0');
                }
                out.push_str(&digits);
                i = j;
            }
        }
    }
    out
}
