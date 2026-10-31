//! Downloads: the list the page shows. Port of the read side of `fle_studio/downloads.py`: jobs are kept in
//! `downloads.json`, and a job that was running when Studio stopped is queued again. The transfer worker comes next.
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub struct Downloads {
    pub file: PathBuf,
    pub jobs: Vec<Map<String, Value>>,
}

impl Downloads {
    pub fn new(data_dir: &Path) -> Downloads {
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
        Downloads { file, jobs }
    }

    fn public(job: &Map<String, Value>) -> Value {
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

    pub fn list(&self) -> Vec<Value> {
        self.jobs.iter().map(Self::public).collect()
    }
}
