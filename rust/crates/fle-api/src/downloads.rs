//! Downloads: models from Hugging Face and engines from GitHub, one job at a time. Port of `fle_studio/downloads.py`.
//!
//! A file is written as `<name>.part` and renamed when it is complete and its SHA-256 matches. An interrupted file
//! continues where it stopped (HTTP Range); the bytes already on disk are hashed again first, so the final check still
//! covers the whole file. Jobs survive a restart and continue by themselves.
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const CHUNK: usize = 1024 * 1024;
const MAX_RETRIES: u32 = 30;
const HF_HOSTS: [&str; 2] = ["huggingface.co", "hf.co"];

type Job = Map<String, Value>;
pub type Hook = Box<dyn Fn(&Job) -> Result<(), String> + Send + Sync>;

struct Inner {
    file: PathBuf,
    jobs: Mutex<Vec<Job>>,
    cv: Condvar,
    token: Box<dyn Fn() -> String + Send + Sync>,
    hooks: Mutex<HashMap<String, Arc<Hook>>>,
    /// seconds to wait before retry number n (tests make it short)
    backoff: Mutex<Box<dyn Fn(u32) -> f64 + Send + Sync>>,
}

#[derive(Clone)]
pub struct Downloads {
    inner: Arc<Inner>,
}

enum Fail {
    /// pause | cancel
    Stop(String),
    Err(String),
}

enum PullErr {
    Stop(String),
    Status(u16),
    Other(String),
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn rand_hex(n: usize) -> String {
    let mut b = vec![0u8; n];
    getrandom::getrandom(&mut b).expect("random bytes");
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn str_of(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}

impl Downloads {
    pub fn new(data_dir: &Path, token: Box<dyn Fn() -> String + Send + Sync>) -> Downloads {
        let file = data_dir.join("downloads.json");
        let mut jobs = vec![];
        if let Some(Value::Array(a)) = std::fs::read_to_string(&file)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        {
            for j in a {
                let Value::Object(mut job) = j else { continue };
                if matches!(
                    job.get("state").and_then(Value::as_str),
                    Some("downloading" | "verifying" | "queued")
                ) {
                    job.insert("state".into(), json!("queued")); // continue after a restart
                }
                job.insert("speed".into(), json!(0.0));
                job.shift_remove("_flag");
                jobs.push(job);
            }
        }
        let inner = Arc::new(Inner {
            file,
            jobs: Mutex::new(jobs),
            cv: Condvar::new(),
            token,
            hooks: Mutex::new(HashMap::new()),
            backoff: Mutex::new(Box::new(|tries| {
                f64::from(2u32.pow(tries.min(6))).min(60.0)
            })),
        });
        let d = Downloads { inner };
        let w = d.clone();
        std::thread::Builder::new()
            .name("downloads".into())
            .spawn(move || w.worker())
            .expect("downloads thread");
        d
    }

    pub fn set_hook(&self, kind: &str, hook: Hook) {
        self.inner
            .hooks
            .lock()
            .unwrap()
            .insert(kind.into(), Arc::new(hook));
    }

    pub fn set_backoff(&self, f: Box<dyn Fn(u32) -> f64 + Send + Sync>) {
        *self.inner.backoff.lock().unwrap() = f;
    }

    // ------------------------------------------------------------------ API

    /// Queue files `[{url, dest, size, sha256}]`. A job with the same dedupe key that is not finished is reused.
    pub fn add(
        &self,
        title: &str,
        files: &[Value],
        kind: &str,
        meta: Value,
        dedupe: &str,
    ) -> Value {
        let mut jobs = self.inner.jobs.lock().unwrap();
        if !dedupe.is_empty() {
            if let Some(job) = jobs.iter_mut().find(|j| {
                j.get("dedupe") == Some(&json!(dedupe))
                    && !matches!(str_of(&j["state"]), "done" | "cancelled")
            }) {
                if matches!(str_of(&job["state"]), "failed" | "paused") {
                    job.insert("state".into(), json!("queued"));
                    job.insert("error".into(), json!(""));
                    self.inner.cv.notify_all();
                }
                return Self::public(job);
            }
        }
        let files: Vec<Value> = files
            .iter()
            .map(|f| {
                let sha = f["sha256"].as_str().map(str::to_lowercase).filter(|s| !s.is_empty());
                json!({"url": f["url"], "dest": f["dest"], "size": f["size"].as_i64().unwrap_or(0), "sha256": sha, "done": 0})
            })
            .collect();
        let mut job = Job::new();
        for (k, v) in [
            ("id", json!(rand_hex(6))),
            ("title", json!(title)),
            ("kind", json!(kind)),
            ("dedupe", json!(dedupe)),
            ("meta", if meta.is_null() { json!({}) } else { meta }),
            ("state", json!("queued")),
            ("error", json!("")),
            ("created", json!(now())),
            ("finished", Value::Null),
            ("files", Value::Array(files)),
            ("speed", json!(0.0)),
        ] {
            job.insert(k.into(), v);
        }
        jobs.push(job.clone());
        self.save_locked(&jobs);
        self.inner.cv.notify_all();
        Self::public(&job)
    }

    /// pause | resume | cancel | remove. `None` when there is no such job.
    pub fn action(&self, id: &str, what: &str) -> Option<Value> {
        let mut jobs = self.inner.jobs.lock().unwrap();
        let at = jobs.iter().position(|j| j["id"] == id)?;
        let state = str_of(&jobs[at]["state"]).to_string();
        let mut removed = false;
        match what {
            "pause" if matches!(state.as_str(), "queued" | "downloading" | "verifying") => {
                if state == "queued" {
                    jobs[at].insert("state".into(), json!("paused"));
                } else {
                    jobs[at].insert("_flag".into(), json!("pause"));
                }
            }
            "resume" if matches!(state.as_str(), "paused" | "failed") => {
                jobs[at].insert("state".into(), json!("queued"));
                jobs[at].insert("error".into(), json!(""));
            }
            "cancel" if !matches!(state.as_str(), "done" | "cancelled") => {
                if matches!(state.as_str(), "downloading" | "verifying") {
                    jobs[at].insert("_flag".into(), json!("cancel"));
                } else {
                    jobs[at].insert("state".into(), json!("cancelled"));
                    discard_parts(&mut jobs[at]);
                }
            }
            "remove" if matches!(state.as_str(), "done" | "cancelled" | "failed") => {
                removed = true;
            }
            _ => {}
        }
        let out = Self::public(&jobs[at]);
        if removed {
            jobs.remove(at);
        }
        self.save_locked(&jobs);
        self.inner.cv.notify_all();
        Some(out)
    }

    pub fn list(&self) -> Vec<Value> {
        self.inner
            .jobs
            .lock()
            .unwrap()
            .iter()
            .map(Self::public)
            .collect()
    }

    /// The unfinished job that writes below `dest_prefix`, if any.
    pub fn active_for(&self, dest_prefix: &str) -> Option<Value> {
        let jobs = self.inner.jobs.lock().unwrap();
        jobs.iter()
            .find(|j| {
                matches!(str_of(&j["state"]), "queued" | "downloading" | "verifying")
                    && j["files"].as_array().is_some_and(|fs| {
                        fs.iter()
                            .any(|f| str_of(&f["dest"]).starts_with(dest_prefix))
                    })
            })
            .map(Self::public)
    }

    fn public(job: &Job) -> Value {
        let files = job
            .get("files")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let sum = |k: &str| {
            files
                .iter()
                .map(|f| f[k].as_i64().unwrap_or(0))
                .sum::<i64>()
        };
        let (total, done) = (sum("size"), sum("done"));
        let mut out: Map<String, Value> = job
            .iter()
            .filter(|(k, _)| !k.starts_with('_'))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        out.insert("bytes_total".into(), json!(total));
        out.insert("bytes_done".into(), json!(done));
        let progress = if total != 0 {
            done as f64 / total as f64
        } else if job.get("state") == Some(&json!("done")) {
            1.0
        } else {
            0.0
        };
        out.insert("progress".into(), json!(progress));
        Value::Object(out)
    }

    // ------------------------------------------------------------------ worker

    fn save_locked(&self, jobs: &[Job]) {
        if let Some(d) = self.inner.file.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let t = now();
        let keep: Vec<Value> = jobs
            .iter()
            .filter(|j| {
                str_of(&j["state"]) != "done"
                    || t - j.get("finished").and_then(Value::as_f64).unwrap_or(0.0) < 7.0 * 86400.0
            })
            .map(|j| {
                Value::Object(
                    j.iter()
                        .filter(|(k, _)| !k.starts_with('_'))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                )
            })
            .collect();
        let tmp = self.inner.file.with_extension("tmp");
        let _ = std::fs::write(&tmp, serde_json::to_string_pretty(&keep).unwrap())
            .and_then(|_| std::fs::rename(&tmp, &self.inner.file));
    }

    /// Change one job under the lock. `None` when it is gone.
    fn with<R>(&self, id: &str, f: impl FnOnce(&mut Job) -> R) -> Option<R> {
        let mut jobs = self.inner.jobs.lock().unwrap();
        jobs.iter_mut().find(|j| j["id"] == id).map(f)
    }

    fn worker(&self) {
        loop {
            let id = {
                let mut jobs = self.inner.jobs.lock().unwrap();
                loop {
                    if let Some(j) = jobs.iter_mut().find(|j| str_of(&j["state"]) == "queued") {
                        j.insert("state".into(), json!("downloading"));
                        j.insert("error".into(), json!(""));
                        j.shift_remove("_flag");
                        let id = str_of(&j["id"]).to_string();
                        self.save_locked(&jobs);
                        break id;
                    }
                    jobs = self.inner.cv.wait(jobs).unwrap();
                }
            };
            let res = self.run(&id).and_then(|()| {
                let kind = self
                    .with(&id, |j| str_of(&j["kind"]).to_string())
                    .unwrap_or_default();
                let hook = self.inner.hooks.lock().unwrap().get(&kind).cloned();
                if let Some(h) = hook {
                    self.with(&id, |j| j.insert("state".into(), json!("installing")));
                    let snapshot = self.with(&id, |j| j.clone()).unwrap_or_default();
                    h(&snapshot).map_err(Fail::Err)?;
                }
                Ok(())
            });
            self.with(&id, |j| match &res {
                Ok(()) => {
                    j.insert("state".into(), json!("done"));
                    j.insert("finished".into(), json!(now()));
                }
                Err(Fail::Stop(why)) => {
                    j.insert(
                        "state".into(),
                        json!(if why == "pause" {
                            "paused"
                        } else {
                            "cancelled"
                        }),
                    );
                    if why == "cancel" {
                        discard_parts(j);
                    }
                }
                Err(Fail::Err(m)) => {
                    j.insert("state".into(), json!("failed"));
                    j.insert("error".into(), json!(m));
                }
            });
            self.with(&id, |j| {
                j.insert("speed".into(), json!(0.0));
                j.shift_remove("_flag");
            });
            let jobs = self.inner.jobs.lock().unwrap();
            self.save_locked(&jobs);
        }
    }

    fn check(&self, id: &str) -> Result<(), Fail> {
        match self
            .with(id, |j| {
                j.get("_flag").and_then(Value::as_str).map(String::from)
            })
            .flatten()
        {
            Some(flag) => Err(Fail::Stop(flag)),
            None => Ok(()),
        }
    }

    fn run(&self, id: &str) -> Result<(), Fail> {
        let files = self
            .with(id, |j| j["files"].as_array().cloned().unwrap_or_default())
            .unwrap_or_default();
        let need: i64 = files
            .iter()
            .filter(|f| !Path::new(str_of(&f["dest"])).exists())
            .map(|f| (f["size"].as_i64().unwrap_or(0) - f["done"].as_i64().unwrap_or(0)).max(0))
            .sum();
        if let Some(first) = files.first() {
            let target = Path::new(str_of(&first["dest"]))
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf();
            let _ = std::fs::create_dir_all(&target);
            let free = crate::system::disk_free(&target) as i64;
            if need != 0 && free != 0 && free < need + 512 * 1024 * 1024 {
                return Err(Fail::Err(format!(
                    "Not enough disk space in {}: {:.1} GB needed, {:.1} GB free",
                    target.display(),
                    need as f64 / 1e9,
                    free as f64 / 1e9
                )));
            }
        }
        for i in 0..files.len() {
            self.fetch(id, i)?;
        }
        Ok(())
    }

    fn file_field(&self, id: &str, idx: usize, key: &str) -> Value {
        self.with(id, |j| j["files"][idx][key].clone())
            .unwrap_or(Value::Null)
    }
    fn set_file(&self, id: &str, idx: usize, key: &str, v: Value) {
        self.with(id, |j| j["files"][idx][key] = v);
    }

    fn fetch(&self, id: &str, idx: usize) -> Result<(), Fail> {
        let dest = PathBuf::from(str_of(&self.file_field(id, idx, "dest")));
        let name = dest.file_name().unwrap().to_string_lossy().to_string();
        let part = dest.with_file_name(format!("{name}.part"));
        let _ = std::fs::create_dir_all(dest.parent().unwrap_or(Path::new(".")));
        let size = self.file_field(id, idx, "size").as_i64().unwrap_or(0);
        if let Ok(md) = std::fs::metadata(&dest) {
            if size == 0 || md.len() as i64 == size {
                self.set_file(id, idx, "done", json!(md.len()));
                return Ok(());
            }
        }
        let sha = self
            .file_field(id, idx, "sha256")
            .as_str()
            .map(String::from);
        let mut st = self.resume_point(id, idx, &part, &sha, size)?;
        self.with(id, |j| j.insert("state".into(), json!("downloading")));
        let token = (self.inner.token)();
        let mut tries = 0u32;
        loop {
            self.check(id)?;
            let size_now = self.file_field(id, idx, "size").as_i64().unwrap_or(0);
            match self.pull(id, idx, &part, &mut st, &token, &sha) {
                Ok(()) => {
                    if size_now != 0 && (st.offset as i64) < size_now {
                        let reason = format!(
                            "IncompleteRead(0 bytes read, {} more expected)",
                            size_now - st.offset as i64
                        );
                        tries = self.backoff(id, tries, &reason)?;
                        continue;
                    }
                    break;
                }
                Err(PullErr::Stop(w)) => return Err(Fail::Stop(w)),
                Err(PullErr::Status(c)) => {
                    let url = str_of(&self.file_field(id, idx, "url")).to_string();
                    match c {
                        401 | 403 => return Err(Fail::Err("Access denied by the server. Gated repo? Accept its terms on Hugging Face and add a token in Settings.".into())),
                        404 => return Err(Fail::Err(format!("Not found: {url}"))),
                        416 if size_now != 0 && st.offset as i64 >= size_now => break,
                        _ => tries = self.backoff(id, tries, &format!("HTTP {c} {}", status_reason(c)))?,
                    }
                }
                Err(PullErr::Other(m)) => tries = self.backoff(id, tries, &m)?,
            }
        }
        if let (Some(h), Some(want)) = (st.h.take(), sha.as_ref()) {
            let got = hex::encode(h.finalize());
            if got != *want {
                let bad = dest.with_file_name(format!("{name}.bad"));
                let _ = std::fs::rename(&part, &bad);
                self.set_file(id, idx, "done", json!(0));
                return Err(Fail::Err(format!("{name} is damaged (checksum mismatch); it was moved to {}. Resume to download it again.", bad.file_name().unwrap().to_string_lossy())));
            }
        }
        std::fs::rename(&part, &dest).map_err(|e| Fail::Err(e.to_string()))?;
        Ok(())
    }

    /// Where an interrupted download continues: size of the .part file, and its checksum state if one is checked.
    fn resume_point(
        &self,
        id: &str,
        idx: usize,
        part: &Path,
        sha: &Option<String>,
        size: i64,
    ) -> Result<Transfer, Fail> {
        let mut h = sha.as_ref().map(|_| Sha256::new());
        let mut offset = std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);
        if size != 0 && offset as i64 > size {
            let _ = std::fs::remove_file(part);
            offset = 0;
        }
        if offset != 0 {
            if let Some(hasher) = h.as_mut() {
                self.with(id, |j| j.insert("state".into(), json!("verifying")));
                let mut src = std::fs::File::open(part).map_err(|e| Fail::Err(e.to_string()))?;
                let mut seen = 0u64;
                while seen < offset {
                    self.check(id)?;
                    let mut buf = vec![0u8; (8 * CHUNK as u64).min(offset - seen) as usize];
                    let n = src.read(&mut buf).map_err(|e| Fail::Err(e.to_string()))?;
                    if n == 0 {
                        break;
                    }
                    hasher.update(&buf[..n]);
                    seen += n as u64;
                    self.set_file(id, idx, "done", json!(seen));
                }
            }
        }
        self.set_file(id, idx, "done", json!(offset));
        Ok(Transfer {
            offset,
            h,
            window_t: Instant::now(),
            window_b: offset,
        })
    }

    /// One connection: request from `st.offset` on and append to the .part file until the stream ends.
    fn pull(
        &self,
        id: &str,
        idx: usize,
        part: &Path,
        st: &mut Transfer,
        token: &str,
        sha: &Option<String>,
    ) -> Result<(), PullErr> {
        let url = str_of(&self.file_field(id, idx, "url")).to_string();
        let agent = ureq::AgentBuilder::new()
            .redirects(0)
            .timeout_connect(Duration::from_secs(60))
            .timeout_read(Duration::from_secs(60))
            .user_agent(&format!("fle-studio/{}", env!("CARGO_PKG_VERSION")))
            .build();
        let mut current = url;
        let mut first = true;
        let resp = loop {
            let mut req = agent.get(&current);
            if st.offset != 0 {
                req = req.set("Range", &format!("bytes={}-", st.offset));
            }
            let host = host_of(&current);
            // the token goes to Hugging Face itself, never along a redirect to a CDN
            if first
                && !token.is_empty()
                && (HF_HOSTS.contains(&host.as_str()) || host.ends_with(".huggingface.co"))
            {
                req = req.set("Authorization", &format!("Bearer {token}"));
            }
            first = false;
            match req.call() {
                Ok(r) if (300..400).contains(&r.status()) => match r.header("Location") {
                    Some(l) => current = resolve(&current, l),
                    None => return Err(PullErr::Status(r.status())),
                },
                Ok(r) => break r,
                Err(ureq::Error::Status(c, _)) => return Err(PullErr::Status(c)),
                Err(e) => return Err(PullErr::Other(transport_reason(&e))),
            }
        };
        if st.offset != 0 && resp.status() == 200 {
            // no range support: start over
            st.offset = 0;
            st.h = sha.as_ref().map(|_| Sha256::new());
            let _ = std::fs::remove_file(part);
        }
        if self.file_field(id, idx, "size").as_i64().unwrap_or(0) == 0 {
            let len = resp
                .header("Content-Length")
                .filter(|l| !l.is_empty() && l.bytes().all(|b| b.is_ascii_digit()))
                .and_then(|l| l.parse::<u64>().ok());
            self.set_file(
                id,
                idx,
                "size",
                json!(len.map(|l| st.offset + l).unwrap_or(0)),
            );
        }
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(part)
            .map_err(|e| PullErr::Other(e.to_string()))?;
        let mut body = resp.into_reader();
        let mut buf = vec![0u8; CHUNK];
        loop {
            if let Err(Fail::Stop(w)) = self.check(id) {
                return Err(PullErr::Stop(w));
            }
            let n = body
                .read(&mut buf)
                .map_err(|e| PullErr::Other(e.to_string()))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])
                .map_err(|e| PullErr::Other(e.to_string()))?;
            if let Some(h) = st.h.as_mut() {
                h.update(&buf[..n]);
            }
            st.offset += n as u64;
            self.set_file(id, idx, "done", json!(st.offset));
            self.update_speed(id, st);
        }
        Ok(())
    }

    /// Smoothed bytes/s over windows of at least two seconds.
    fn update_speed(&self, id: &str, st: &mut Transfer) {
        let dt = st.window_t.elapsed().as_secs_f64();
        if dt >= 2.0 {
            let rate = (st.offset - st.window_b) as f64 / dt;
            self.with(id, |j| {
                let old = j["speed"].as_f64().unwrap_or(0.0);
                j.insert(
                    "speed".into(),
                    json!(if old == 0.0 {
                        rate
                    } else {
                        0.7 * old + 0.3 * rate
                    }),
                );
            });
            st.window_t = Instant::now();
            st.window_b = st.offset;
        }
    }

    fn backoff(&self, id: &str, tries: u32, reason: &str) -> Result<u32, Fail> {
        let tries = tries + 1;
        if tries > MAX_RETRIES {
            return Err(Fail::Err(format!(
                "Download failed after {MAX_RETRIES} attempts: {reason}"
            )));
        }
        self.with(id, |j| {
            j.insert(
                "error".into(),
                json!(format!(
                    "Connection problem, retrying ({tries}/{MAX_RETRIES}): {reason}"
                )),
            );
            j.insert("speed".into(), json!(0.0));
        });
        let wait = (self.inner.backoff.lock().unwrap())(tries);
        let deadline = Instant::now() + Duration::from_secs_f64(wait);
        while Instant::now() < deadline {
            self.check(id)?;
            std::thread::sleep(Duration::from_millis(if wait < 1.0 { 20 } else { 500 }));
        }
        self.with(id, |j| j.insert("error".into(), json!("")));
        Ok(tries)
    }
}

struct Transfer {
    offset: u64,
    h: Option<Sha256>,
    window_t: Instant,
    window_b: u64,
}

fn discard_parts(job: &mut Job) {
    if let Some(files) = job.get_mut("files").and_then(Value::as_array_mut) {
        for f in files {
            let dest = str_of(&f["dest"]).to_string();
            let _ = std::fs::remove_file(format!("{dest}.part"));
            if !Path::new(&dest).exists() {
                f["done"] = json!(0);
            }
        }
    }
}

fn host_of(url: &str) -> String {
    url.split_once("://")
        .map(|x| x.1)
        .unwrap_or(url)
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .rsplit('@')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("")
        .to_lowercase()
}

fn resolve(base: &str, loc: &str) -> String {
    if loc.contains("://") {
        return loc.to_string();
    }
    let (scheme, rest) = base.split_once("://").unwrap_or(("http", base));
    let authority = rest.split('/').next().unwrap_or("");
    if let Some(p) = loc.strip_prefix('/') {
        return format!("{scheme}://{authority}/{p}");
    }
    let dir = rest.rsplit_once('/').map(|x| x.0).unwrap_or(rest);
    format!("{scheme}://{dir}/{loc}")
}

fn status_reason(c: u16) -> &'static str {
    match c {
        400 => "Bad Request",
        408 => "Request Timeout",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "",
    }
}

fn transport_reason(e: &ureq::Error) -> String {
    use std::error::Error;
    let mut cur: &dyn Error = e;
    while let Some(s) = cur.source() {
        cur = s;
    }
    cur.to_string()
}
