//! Start llama-server and, with --fallback, retry once with upstream llama.cpp if it dies while loading.
//! Port of `tools/fle/serve.py`. The watching is in `Supervisor`, so it can be tested with a fake process and a fake
//! health check; nothing here needs a real server.
use crate::launch::Plan;
use crate::FleError;
use anyhow::Result;
use std::time::{Duration, Instant};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum State {
    /// /health answered OK
    Ready,
    /// the process ended with a non-zero code before it was ready
    Failed,
    /// the process ended with code 0 before it was ready
    Exited,
    /// still loading when the wait ran out
    Timeout,
}

pub trait Process {
    fn poll(&mut self) -> Option<i32>;
    fn wait(&mut self, timeout: Option<Duration>) -> Option<i32>;
    fn terminate(&mut self);
    fn kill(&mut self);
}

pub struct Child(pub std::process::Child);
impl Process for Child {
    fn poll(&mut self) -> Option<i32> {
        self.0
            .try_wait()
            .ok()
            .flatten()
            .map(|s| s.code().unwrap_or(1))
    }
    fn wait(&mut self, timeout: Option<Duration>) -> Option<i32> {
        let start = Instant::now();
        loop {
            if let Some(c) = self.poll() {
                return Some(c);
            }
            if timeout.is_some_and(|t| start.elapsed() >= t) {
                return None;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    fn terminate(&mut self) {
        #[cfg(unix)]
        {
            let _ = std::process::Command::new("kill")
                .arg("-TERM")
                .arg(self.0.id().to_string())
                .status();
        }
        #[cfg(not(unix))]
        {
            let _ = self.0.kill();
        }
    }
    fn kill(&mut self) {
        let _ = self.0.kill();
    }
}

/// Create what a plan's environment points at. Plans only compute; this runs just before a server starts.
pub fn prepare_run_env(env: &[(String, String)]) {
    if let Some((_, target)) = env.iter().find(|(k, _)| k == "FLE_MOE_PROFILE") {
        if !target.is_empty() {
            if let Some(d) = std::path::Path::new(target).parent() {
                let _ = std::fs::create_dir_all(d); // the engine reports an unwritable profile path itself
            }
        }
    }
}

pub fn spawn_process(plan: &Plan) -> Result<Box<dyn Process>> {
    prepare_run_env(&plan.env);
    let mut c = std::process::Command::new(&plan.binary);
    c.args(&plan.args);
    for (k, v) in &plan.env {
        c.env(k, v);
    }
    match c.spawn() {
        Ok(ch) => Ok(Box::new(Child(ch))),
        Err(e) => Err(FleError::new(
            format!("Cannot start {}: {e}", plan.binary),
            "Check the path and its shared libraries.",
        )
        .into()),
    }
}

/// True when llama-server's /health answers 200 (it answers 503 while the model is loading).
pub fn http_health(host: &str, port: i64, timeout: f64) -> bool {
    let shown = if matches!(host, "0.0.0.0" | "" | "::") {
        "127.0.0.1"
    } else {
        host
    };
    let shown = if shown.contains(':') {
        format!("[{shown}]")
    } else {
        shown.to_string()
    };
    let Ok(c) = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs_f64(timeout))
        .build()
    else {
        return false;
    };
    c.get(format!("http://{shown}:{port}/health"))
        .send()
        .map(|r| r.status().as_u16() == 200)
        .unwrap_or(false)
}

/// Starts the server of a plan.
pub type Spawner<'a> = Box<dyn FnMut(&Plan) -> Result<Box<dyn Process>> + 'a>;

pub struct Supervisor<'a> {
    pub spawn: Spawner<'a>,
    pub health: Box<dyn FnMut(&Plan) -> bool + 'a>,
    pub ready_timeout: f64,
    pub poll_interval: f64,
    pub sleep: Box<dyn FnMut(f64) + 'a>,
    pub clock: Box<dyn FnMut() -> f64 + 'a>,
    pub say: Box<dyn FnMut(&str) + 'a>,
    pub progress: bool,
    pub progress_interval: f64,
    pub waited: f64,
}

impl<'a> Supervisor<'a> {
    pub fn real(ready_timeout: f64) -> Supervisor<'a> {
        let t0 = Instant::now();
        Supervisor {
            spawn: Box::new(spawn_process),
            health: Box::new(|p| http_health(&p.host, p.port, 2.0)),
            ready_timeout,
            poll_interval: 1.0,
            sleep: Box::new(|s| std::thread::sleep(Duration::from_secs_f64(s))),
            clock: Box::new(move || t0.elapsed().as_secs_f64()),
            say: Box::new(|t| eprintln!("{t}")),
            progress: std::io::IsTerminal::is_terminal(&std::io::stderr()),
            progress_interval: 10.0,
            waited: 0.0,
        }
    }

    /// Watch a starting server until it is ready, has ended, or the wait runs out. With progress on, a line
    /// "still loading ... N s" goes out about every `progress_interval` seconds.
    pub fn wait_ready(&mut self, proc: &mut dyn Process, plan: &Plan) -> (State, Option<i32>) {
        let started = (self.clock)();
        let deadline = started + self.ready_timeout;
        let mut next_report = started + self.progress_interval;
        let out = loop {
            if let Some(code) = proc.poll() {
                break (
                    if code == 0 {
                        State::Exited
                    } else {
                        State::Failed
                    },
                    Some(code),
                );
            }
            if (self.health)(plan) {
                break (State::Ready, None);
            }
            let now = (self.clock)();
            if now >= deadline {
                break (State::Timeout, None);
            }
            if self.progress && self.progress_interval > 0.0 && now >= next_report {
                (self.say)(&format!(
                    "fle: still loading ... {} s",
                    (now - started) as i64
                ));
                next_report = now + self.progress_interval;
            }
            (self.sleep)(self.poll_interval);
        };
        self.waited = (self.clock)() - started;
        out
    }

    fn finish(proc: &mut dyn Process) -> i32 {
        proc.wait(None).unwrap_or(1)
    }

    /// Run `plan`; if it fails while loading, build a fallback plan with `make_fallback(reason)` and run that once.
    /// `make_fallback` fails with a `FleError` when there is nothing to fall back to.
    pub fn run(
        &mut self,
        plan: &Plan,
        make_fallback: &mut dyn FnMut(&str) -> Result<Plan>,
    ) -> Result<i32> {
        let mut proc = (self.spawn)(plan)?;
        let (state, code) = self.wait_ready(proc.as_mut(), plan);
        match state {
            State::Ready => {
                let m = if self.progress {
                    format!("fle: ready after {} s", self.waited as i64)
                } else {
                    "fle: the server is ready.".to_string()
                };
                (self.say)(&m);
                Ok(Self::finish(proc.as_mut()))
            }
            State::Exited => Ok(code.unwrap_or(0)),
            State::Timeout => {
                (self.say)("fle: still loading after the wait limit; leaving the server running without fallback.");
                Ok(Self::finish(proc.as_mut()))
            }
            State::Failed => {
                let reason = format!(
                    "the server exited with code {} before it was ready",
                    code.unwrap_or(0)
                );
                let fb = match make_fallback(&reason) {
                    Ok(f) => f,
                    Err(e) => {
                        let (m, h) = match e.downcast_ref::<FleError>() {
                            Some(f) => (f.message.clone(), f.hint.clone()),
                            None => (e.to_string(), String::new()),
                        };
                        return Err(FleError::new(format!("The server failed to start ({reason}), and no fallback is possible: {m}"), h).into());
                    }
                };
                (self.say)(&format!("fle: {reason}."));
                (self.say)(&format!("fle: restarting once with upstream llama.cpp ({}, profile off, no FLE features).", fb.binary));
                let mut second = (self.spawn)(&fb)?;
                Ok(Self::finish(second.as_mut()))
            }
        }
    }
}
