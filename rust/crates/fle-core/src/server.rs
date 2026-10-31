use crate::plan::Plan;
use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub struct Server {
    pub child: Child,
    pub base: String,
    pub log: std::path::PathBuf,
}

impl Server {
    pub fn start(server: &Path, plan: &Plan, log_dir: &Path) -> Result<Server> {
        std::fs::create_dir_all(log_dir)?;
        let log = log_dir.join("engine.log");
        let out = std::fs::File::create(&log)?;
        let mut cmd = Command::new(server);
        cmd.args(&plan.args)
            .stdout(Stdio::from(out.try_clone()?))
            .stderr(Stdio::from(out))
            .stdin(Stdio::null());
        for (k, v) in &plan.env {
            cmd.env(k, v);
        }
        if let Some(dir) = server.parent() {
            // the engine's shared libraries sit next to it
            #[cfg(target_os = "macos")]
            cmd.env("DYLD_LIBRARY_PATH", dir);
            #[cfg(target_os = "linux")]
            cmd.env("LD_LIBRARY_PATH", dir);
        }
        let child = cmd
            .spawn()
            .with_context(|| format!("could not start {}", server.display()))?;
        Ok(Server {
            child,
            base: format!("http://127.0.0.1:{}", plan.port),
            log,
        })
    }

    /// Wait until the engine answers /health. `tick` gets the last interesting log line about once a second.
    pub fn wait_ready(&mut self, timeout: Duration, tick: &mut dyn FnMut(&str)) -> Result<()> {
        let c = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()?;
        let start = Instant::now();
        loop {
            if let Some(st) = self.child.try_wait()? {
                bail!(
                    "the engine stopped ({st}); see {}\n{}",
                    self.log.display(),
                    self.tail(6)
                );
            }
            if c.get(format!("{}/health", self.base))
                .send()
                .is_ok_and(|r| r.status().is_success())
            {
                return Ok(());
            }
            tick(self.last_line().as_str());
            if start.elapsed() > timeout {
                bail!(
                    "the engine did not become ready in {} s; see {}",
                    timeout.as_secs(),
                    self.log.display()
                );
            }
            std::thread::sleep(Duration::from_millis(700));
        }
    }

    pub fn tail(&self, n: usize) -> String {
        let t = std::fs::read_to_string(&self.log).unwrap_or_default();
        let lines: Vec<&str> = t.lines().collect();
        lines[lines.len().saturating_sub(n)..].join("\n")
    }

    fn last_line(&self) -> String {
        self.tail(40)
            .lines()
            .rev()
            .find(|l| {
                [
                    "load",
                    "Load",
                    "ggml_metal",
                    "llama_",
                    "sched",
                    "init",
                    "warming",
                ]
                .iter()
                .any(|k| l.contains(k))
            })
            .unwrap_or("starting the engine ...")
            .chars()
            .take(120)
            .collect()
    }

    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}
