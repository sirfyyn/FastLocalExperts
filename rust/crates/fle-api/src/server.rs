//! The HTTP side of FLE Studio: the guard (host name, origin, session token), the JSON conventions and the static page.
//! Port of the plumbing in `fle_studio/server.py`; routes live next to the code they call.
use crate::chats::{ChatError, Chats, MAX_BYTES as MAX_CHAT};
use crate::downloads::Downloads;
use crate::engine::{Engine, EngineErr};
use crate::engines::{EngineError, Engines};
use crate::launch::Profiles;
use crate::library::{LibError, Library};
use crate::live::Live;
use crate::settings::Config;
use axum::body::{to_bytes, Body};
use axum::extract::{ConnectInfo, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, Request, Response, StatusCode};
use axum::Router;
use base64::Engine as _;
use serde_json::{json, Map, Value};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const MAX_JSON: usize = 4 * 1024 * 1024;
pub const MAX_PROXY: usize = 64 * 1024 * 1024;
const LOOPBACK_NAMES: [&str; 3] = ["127.0.0.1", "::1", "localhost"];

pub struct Options {
    pub home: PathBuf,
    /// The built web page (index.html, js/, css). Later it is embedded in the binary.
    pub web_dir: Option<PathBuf>,
    pub version: String,
    /// Where GitHub's releases are asked (tests point it at a server of their own).
    pub github_api: String,
}

pub struct Studio {
    pub opts: Options,
    pub token: String,
    pub config: Arc<Mutex<Config>>,
    pub chats: Chats,
    pub chat_lock: Mutex<()>,
    pub library: Mutex<Library>,
    pub downloads: Downloads,
    pub engines: Engines,
    pub profiles: Mutex<Profiles>,
    pub engine: Engine,
    pub live: Live,
    /// the server listens beyond this machine (host mode); set by the starter
    pub lan: bool,
}

/// An error the page shows as `{"error": {"message": ...}}`.
pub struct ApiError(pub u16, pub String);

pub type Ctx = Arc<Studio>;

fn make_token() -> String {
    let mut b = [0u8; 24];
    getrandom::getrandom(&mut b).expect("random bytes");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
}

pub fn new_studio(opts: Options) -> Ctx {
    let config = Config::load(&opts.home.join("settings.json"));
    let home = opts.home.clone();
    let config = Arc::new(Mutex::new(config));
    let tok = config.clone();
    let downloads = Downloads::new(
        &home,
        Box::new(move || {
            tok.lock()
                .unwrap()
                .get("hf_token")
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default()
        }),
    );
    let engines = Engines::new(&home, downloads.clone(), config.clone(), &opts.github_api);
    let engine = Engine::new(&home);
    engine.set_restart_on_crash(
        config
            .lock()
            .unwrap()
            .get("launch.restart_on_crash")
            .map(crate::settings::py_truthy)
            .unwrap_or(false),
    );
    let chats = Chats::new(&home);
    let studio = Arc::new(Studio {
        opts,
        token: make_token(),
        config,
        chats,
        chat_lock: Mutex::new(()),
        library: Mutex::new(Library::new(&home)),
        engines,
        lan: false,
        live: Live::new(engine.clone(), 1.0),
        engine,
        profiles: Mutex::new(Profiles::new(&home)),
        downloads,
    });
    let weak = Arc::downgrade(&studio);
    studio
        .engine
        .set_on_load_failure(Box::new(move |plan, rc, error| {
            weak.upgrade()
                .and_then(|s| s.load_failure_fallback(plan, rc, error))
        }));
    studio
}

fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

fn base_headers(r: &mut Response<Body>, ctype: &str, extra: &[(&str, &str)]) {
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_str(ctype).unwrap());
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    h.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    h.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    for (k, v) in extra {
        // like Python's send_header: a header named twice is sent twice (Cache-Control on pages)
        h.append(
            axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
            HeaderValue::from_str(v).unwrap(),
        );
    }
}

pub fn send(
    status: u16,
    body: Vec<u8>,
    ctype: &str,
    extra: &[(&str, &str)],
    head: bool,
) -> Response<Body> {
    let len = body.len();
    let mut r = Response::new(if head {
        Body::empty()
    } else {
        Body::from(body)
    });
    *r.status_mut() = StatusCode::from_u16(status).unwrap();
    base_headers(&mut r, ctype, extra);
    r.headers_mut()
        .insert(header::CONTENT_LENGTH, HeaderValue::from(len));
    r
}

pub fn send_json(status: u16, data: &Value, head: bool) -> Response<Body> {
    send(
        status,
        serde_json::to_vec(data).unwrap(),
        "application/json; charset=utf-8",
        &[],
        head,
    )
}

fn error_json(status: u16, msg: &str, head: bool) -> Response<Body> {
    send_json(status, &json!({"error": {"message": msg}}), head)
}

fn host_ok(headers: &HeaderMap) -> bool {
    let host = headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .trim()
        .to_string();
    let name = if let Some(r) = host.strip_prefix('[') {
        r.split(']').next().unwrap_or("").to_string()
    } else {
        host.rsplit_once(':')
            .map(|x| x.0)
            .unwrap_or(&host)
            .to_string()
    };
    LOOPBACK_NAMES.contains(&name.to_lowercase().as_str())
}

/// The Origin must be this page's own host (a request from another site is refused), and a mutating call carries the
/// session token that the served page was given.
fn guard(
    st: &Studio,
    method: &Method,
    path: &str,
    headers: &HeaderMap,
    peer: &SocketAddr,
    head: bool,
) -> Option<Response<Body>> {
    if !host_ok(headers) {
        return Some(send(
            403,
            b"FLE Studio: this host name is not allowed.".to_vec(),
            "text/plain; charset=utf-8",
            &[],
            head,
        ));
    }
    if path == "/login" || path == "/api/remote/pair" {
        return None;
    }
    if !peer.ip().is_loopback() {
        return Some(error_json_auth(head));
    }
    if matches!(
        *method,
        Method::POST | Method::PUT | Method::DELETE | Method::PATCH
    ) {
        let origin = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok());
        if let Some(o) = origin.filter(|o| *o != "null" && !o.is_empty()) {
            let netloc = o
                .split_once("://")
                .map(|x| x.1)
                .unwrap_or(o)
                .split('/')
                .next()
                .unwrap_or("");
            if Some(netloc) != headers.get(header::HOST).and_then(|v| v.to_str().ok()) {
                return Some(error_json(403, "Cross-origin request refused", head));
            }
        }
        if origin == Some("null") {
            return Some(error_json(
                403,
                "Request from an opaque origin refused",
                head,
            ));
        }
        if (path.starts_with("/api/") || path.starts_with("/h/"))
            && !ct_eq(
                headers
                    .get("x-fle-token")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or(""),
                &st.token,
            )
        {
            return Some(send(
                403,
                serde_json::to_vec(
                    &json!({"error": {"message": "Session expired; reload the page"}}),
                )
                .unwrap(),
                "application/json; charset=utf-8",
                &[("X-FLE-Session", "expired")],
                head,
            ));
        }
    }
    None
}

fn error_json_auth(head: bool) -> Response<Body> {
    send_json(
        401,
        &json!({"error": {"message": "Sign in with the access key", "type": "auth"}}),
        head,
    )
}

fn query_of(q: Option<&str>) -> Map<String, Value> {
    let mut m = Map::new();
    for (k, v) in form_urlencoded::parse(q.unwrap_or("").as_bytes()) {
        if !v.is_empty() {
            m.insert(k.to_string(), Value::String(v.to_string()));
        }
    }
    m
}

pub struct Req {
    pub method: Method,
    pub path: String,
    pub query: Map<String, Value>,
    pub body: Vec<u8>,
    /// the body was larger than any route accepts
    pub too_large: bool,
}

impl Req {
    /// The JSON object in the body (an empty body is `{}`); more than `limit` bytes is a 413, like `read_json(limit)`.
    pub fn json(&self, limit: usize) -> Result<Map<String, Value>, ApiError> {
        if self.too_large || self.body.len() > limit {
            return Err(ApiError(413, "Request too large".into()));
        }
        if self.body.is_empty() {
            return Ok(Map::new());
        }
        match serde_json::from_slice::<Value>(&self.body) {
            Ok(Value::Object(m)) => Ok(m),
            Ok(_) => Err(ApiError(400, "Expected a JSON object".into())),
            Err(_) => Err(ApiError(400, "Invalid JSON".into())),
        }
    }
}

fn chat_err(e: ChatError) -> ApiError {
    match e {
        ChatError::NotFound => ApiError(404, "Not found".into()),
        ChatError::Invalid(m) => ApiError(400, m),
    }
}

/// `/api/chats/<16 hex>` -> the id
fn chat_id(p: &str) -> Option<&str> {
    let id = p.strip_prefix("/api/chats/")?;
    (id.len() == 16
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
    .then_some(id)
}

impl Studio {
    pub fn models_dir(&self) -> std::path::PathBuf {
        self.config.lock().unwrap().models_dir()
    }

    /// The engine a start should use (`engine_for`).
    pub fn engine_for(&self, want: &str, info: &Value) -> Result<Value, ApiError> {
        let (custom_path, choice) = {
            let c = self.config.lock().unwrap();
            let g = |k: &str| {
                c.get("engine")
                    .and_then(|e| e.get(k))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string()
            };
            (g("custom_path"), g("choice"))
        };
        let custom = if custom_path.is_empty() {
            None
        } else {
            self.engines.resolve("custom", info).ok()
        };
        if want == "custom" {
            return custom.ok_or_else(|| {
                ApiError(
                    400,
                    "Set a working llama-server path on the Engine page".into(),
                )
            });
        }
        if let Some(c) = &custom {
            if (want == "fle" || want == "upstream") && c["flavor"] == want && choice == "custom" {
                return Ok(c.clone());
            }
        }
        match self.engines.resolve(want, info) {
            Ok(e) => Ok(e),
            Err(EngineError::Runtime(m)) => match custom {
                Some(c) if want == "auto" || c["flavor"] == want => Ok(c),
                _ => Err(ApiError(409, m)),
            },
            Err(_) => Err(ApiError(409, "No engine".into())),
        }
    }

    /// The command a start would run (`make_plan`).
    pub fn make_plan(
        &self,
        model_ident: &str,
        profile_ident: Option<&Value>,
    ) -> Result<crate::launch::Plan, ApiError> {
        use crate::launch::{
            auto_plan, free_port, profile_plan, AutoInput, PlanError, ProfileInput,
        };
        enum Fail {
            Api(ApiError),
            Plan(PlanError),
        }
        let model = if model_ident.is_empty() {
            None
        } else {
            self.library
                .lock()
                .unwrap()
                .get(&self.models_dir(), model_ident)
        };
        let settings = Value::Object(self.config.lock().unwrap().data.clone());
        let given = profile_ident
            .filter(|v| crate::settings::py_truthy(v))
            .map(crate::settings::py_str);
        let prof_id = given
            .clone()
            .unwrap_or_else(|| match settings["launch"].get("profile") {
                Some(v) if crate::settings::py_truthy(v) => crate::settings::py_str(v),
                _ => "auto".into(),
            });
        let info = self.sysinfo();
        let want_port = settings["launch"]
            .get("port")
            .and_then(Value::as_i64)
            .unwrap_or(0)
            .clamp(0, 65535) as u16;
        let port = i64::from(free_port(want_port));
        let data_dir = self.opts.home.clone();
        let attempt = || -> Result<crate::launch::Plan, Fail> {
            if prof_id == "auto" {
                let Some(m) = &model else {
                    return Err(Fail::Api(ApiError(404, "Choose a model first".into())));
                };
                if m["complete"] != json!(true) {
                    return Err(Fail::Api(ApiError(
                        409,
                        "This model is not completely downloaded yet".into(),
                    )));
                }
                let eng = self.engine_for("auto", &info).map_err(Fail::Api)?;
                return auto_plan(&AutoInput {
                    model: m,
                    engine: &eng,
                    settings: &settings,
                    sysinfo: &info,
                    port,
                    data_dir: &data_dir,
                })
                .map_err(Fail::Plan);
            }
            let prof =
                self.profiles.lock().unwrap().get(&prof_id).ok_or_else(|| {
                    Fail::Api(ApiError(404, format!("No launch profile {prof_id}")))
                })?;
            let pengine = crate::settings::py_str(&prof.engine);
            let eng: Option<Value> = if pengine == "custom" {
                let bin = crate::settings::py_str(&prof.binary);
                // Only to learn which flags its binary knows; the profile still starts that binary.
                if !bin.is_empty() && !bin.contains('{') {
                    Some(self.engines.detect(&bin))
                } else {
                    None
                }
            } else {
                Some(self.engine_for(&pengine, &info).map_err(Fail::Api)?)
            };
            profile_plan(&ProfileInput {
                profile: &prof,
                model: model.as_ref(),
                engine: eng.as_ref(),
                settings: &settings,
                sysinfo: &info,
                port,
                data_dir: &data_dir,
            })
            .map_err(Fail::Plan)
        };
        match attempt() {
            Ok(p) => Ok(p),
            Err(Fail::Api(e)) => Err(e),
            Err(Fail::Plan(exc)) => {
                // Plan-time upstream fallback: if the FLE plan cannot be built (no NVIDIA GPU for the FLE engine, say),
                // retry once with an installed upstream llama-server and speculation off. This does not rescue a model
                // that fails after it has started loading (the engine's own load fallback does that).
                if prof_id != "auto" {
                    // (the page sends "auto" itself, so that is not a reason to refuse)
                    return Err(ApiError(400, exc.0));
                }
                let Some(up) = self.engines.resolve("upstream", &info).ok() else {
                    return Err(ApiError(
                        400,
                        format!(
                            "{} No upstream llama.cpp is installed to fall back to.",
                            exc.0
                        ),
                    ));
                };
                let mut plan = match self.upstream_plan(model.as_ref(), &settings, &up, &info, port)
                {
                    Ok(p) => p,
                    Err(fb) => {
                        return Err(ApiError(
                            400,
                            format!("{} The upstream fallback failed too: {}", exc.0, fb.0),
                        ))
                    }
                };
                let why = exc.0.trim().to_string();
                plan.fallback = format!(
                    "The FLE plan could not be built: {why}{} Studio fell back to upstream llama.cpp; FLE features and speculation are off.",
                    if why.ends_with('.') { "" } else { "." }
                );
                plan.notes.insert(0, plan.fallback.clone());
                Ok(plan)
            }
        }
    }

    /// The automatic plan for an upstream llama-server with speculation off (the fallback of the FLE plan).
    fn upstream_plan(
        &self,
        model: Option<&Value>,
        settings: &Value,
        upstream: &Value,
        info: &Value,
        port: i64,
    ) -> Result<crate::launch::Plan, crate::launch::PlanError> {
        let mut fb = settings.clone();
        fb["launch"]["mtp"] = json!(false);
        fb["launch"]["spec"] = json!("none");
        let m = model.cloned().unwrap_or(json!({}));
        crate::launch::auto_plan(&crate::launch::AutoInput {
            model: &m,
            engine: upstream,
            settings: &fb,
            sysinfo: info,
            port,
            data_dir: &self.opts.home,
        })
    }

    /// The engine ended while loading. With the automatic profile, start upstream llama.cpp once instead. Only for a plan that
    /// Studio built itself (profile auto), not for one that already is the upstream engine, and not when
    /// `launch.fallback_on_failure` is off or no upstream llama-server is installed.
    pub fn load_failure_fallback(
        &self,
        failed: &crate::launch::Plan,
        rc: i32,
        error: &str,
    ) -> Option<crate::launch::Plan> {
        let settings = Value::Object(self.config.lock().unwrap().data.clone());
        if !settings["launch"]
            .get("fallback_on_failure")
            .map(crate::settings::py_truthy)
            .unwrap_or(true)
            || failed.profile != "auto"
        {
            return None;
        }
        let info = self.sysinfo();
        let upstream = self.engines.resolve("upstream", &info).ok()?;
        if upstream["path"].as_str() == Some(failed.binary.as_str()) {
            return None;
        }
        let want_port = settings["launch"]
            .get("port")
            .and_then(Value::as_i64)
            .unwrap_or(0)
            .clamp(0, 65535) as u16;
        let port = i64::from(crate::launch::free_port(want_port));
        let mut plan = self
            .upstream_plan(Some(&failed.model), &settings, &upstream, &info, port)
            .ok()?;
        let what = format!("exit code {rc}")
            + &if error.is_empty() {
                String::new()
            } else {
                format!(": {error}")
            };
        plan.fallback = format!("The engine ended while loading ({what}). Starting upstream llama.cpp instead, once; FLE features and speculation are off. Switch this off under Engine > Advanced > Start upstream if loading fails.");
        plan.notes.insert(0, plan.fallback.clone());
        Some(plan)
    }

    /// What the user can do in one click so that the plan can use MTP, as buttons (`plan_fixes`).
    pub fn plan_fixes(&self, model_ident: &str, profile_ident: Option<&Value>) -> Vec<Value> {
        let launch_cfg = self
            .config
            .lock()
            .unwrap()
            .get("launch")
            .cloned()
            .unwrap_or(json!({}));
        let prof_id = match profile_ident.filter(|v| crate::settings::py_truthy(v)) {
            Some(v) => crate::settings::py_str(v),
            None => match launch_cfg.get("profile") {
                Some(v) if crate::settings::py_truthy(v) => crate::settings::py_str(v),
                _ => "auto".into(),
            },
        };
        let model = if model_ident.is_empty() {
            None
        } else {
            self.library
                .lock()
                .unwrap()
                .get(&self.models_dir(), model_ident)
        };
        let Some(model) = model.filter(|m| prof_id == "auto" && m["catalog_id"].is_string()) else {
            return vec![];
        };
        let mtp = if model["mtp"].is_object() {
            model["mtp"].clone()
        } else {
            json!({})
        };
        if mtp["supported"] != json!(true) || (mtp["head"].is_string() && mtp["vocab"].is_string())
        {
            return vec![];
        }
        let spec = match launch_cfg.get("spec") {
            Some(v) if crate::settings::py_truthy(v) => crate::settings::py_str(v).to_lowercase(),
            _ => "auto".into(),
        };
        if (spec != "auto" && spec != "mtp")
            || !launch_cfg
                .get("mtp")
                .map(crate::settings::py_truthy)
                .unwrap_or(true)
        {
            return vec![]; // MTP is off on purpose (or contradicted): a download would not change the plan
        }
        let Some(entry) = crate::catalog::by_id(model["catalog_id"].as_str().unwrap_or(""))
            .filter(|e| e["mtp"] == json!(true))
        else {
            return vec![];
        };
        let size = crate::catalog::mtp_head()["size"].as_i64().unwrap_or(0)
            + crate::catalog::mtp_vocab()["size"].as_i64().unwrap_or(0);
        vec![
            json!({"kind": "download_mtp", "model": entry["id"], "label": "Download MTP files", "size": size}),
        ]
    }

    /// Whether the Speculation setting changes the command: `{"applies", "reason"}` (`speculation_view`).
    pub fn speculation_view(
        &self,
        profile_ident: Option<&Value>,
        plan: Option<&crate::launch::Plan>,
    ) -> Value {
        let cfg_profile = self.config.lock().unwrap().get("launch.profile").cloned();
        let prof_id = match profile_ident.filter(|v| crate::settings::py_truthy(v)) {
            Some(v) => crate::settings::py_str(v),
            None => match cfg_profile {
                Some(v) if crate::settings::py_truthy(&v) => crate::settings::py_str(&v),
                _ => "auto".into(),
            },
        };
        if prof_id != "auto" {
            return json!({"applies": false, "reason": "A start profile's own command decides; speculation applies to the automatic start only."});
        }
        if let Some(p) = plan.filter(|p| !p.fallback.is_empty()) {
            return json!({"applies": false, "reason": p.fallback});
        }
        let flavor: Option<String> = match plan {
            Some(p) if p.profile == "auto" && (p.engine == "fle" || p.engine == "upstream") => {
                Some(p.engine.clone())
            }
            _ => self
                .engine_for("auto", &self.sysinfo())
                .ok()
                .map(|e| match e.get("flavor") {
                    Some(v) if crate::settings::py_truthy(v) => crate::settings::py_str(v),
                    _ => crate::settings::py_str(&e["kind"]),
                }),
        };
        if flavor.as_deref() == Some("upstream") {
            return json!({"applies": false, "reason": "Speculation needs the FLE engine; the selected engine is upstream llama.cpp. Pick an FLE engine to use it."});
        }
        json!({"applies": true, "reason": ""})
    }

    /// `make_plan` for the live preview: a plan that cannot be built is an answer, not a failure (`plan_view`).
    pub fn plan_view(&self, model_ident: &str, profile_ident: Option<&Value>) -> Value {
        let fixes = self.plan_fixes(model_ident, profile_ident);
        match self.make_plan(model_ident, profile_ident) {
            Err(e) => {
                json!({"ok": false, "error": e.1, "fixes": fixes, "speculation": self.speculation_view(profile_ident, None)})
            }
            Ok(plan) => {
                let public = plan.public();
                let launch_cfg = self
                    .config
                    .lock()
                    .unwrap()
                    .get("launch")
                    .cloned()
                    .unwrap_or(json!({}));
                let model = if model_ident.is_empty() {
                    None
                } else {
                    self.library
                        .lock()
                        .unwrap()
                        .get(&self.models_dir(), model_ident)
                };
                let spec = match launch_cfg.get("spec") {
                    Some(v) if crate::settings::py_truthy(v) => crate::settings::py_str(v),
                    _ => "auto".into(),
                };
                let chips = crate::planview::summary(
                    &public,
                    model.as_ref(),
                    &spec,
                    launch_cfg
                        .get("mtp")
                        .map(crate::settings::py_truthy)
                        .unwrap_or(true),
                );
                json!({"ok": true, "plan": public, "fixes": fixes, "speculation": self.speculation_view(profile_ident, Some(&plan)), "summary": chips})
            }
        }
    }

    /// Everything the page needs at once (`Studio.state`).
    pub fn state(&self) -> Value {
        let info = self.sysinfo();
        let (os, arch) = crate::engines::os_arch();
        let models = self.library.lock().unwrap().scan(&self.models_dir());
        json!({
            "app": "fle-studio", "version": self.opts.version, "platform": info["platform"],
            "systemd": cfg!(target_os = "linux") && fle_core::launch::which("systemctl").is_some(),
            "data_dir": self.opts.home.to_string_lossy(), "lan": self.lan,
            "settings": self.config.lock().unwrap().public(), "system": info.clone(), "engine": self.engine.status(), "models": models,
            "catalog": self.catalog_view(), "extras": self.extras_view(),
            "profiles": self.profiles.lock().unwrap().all().iter().map(|p| p.public()).collect::<Vec<_>>(),
            "engines": {"installed": self.engines.installed(), "variants": crate::engines::variants(&info, os, &arch)},
            "downloads": self.downloads.list(),
            "reference": {"machine": crate::catalog::reference_machine(), "report": crate::catalog::report_url()},
        })
    }

    /// What the loaded model can do, as far as the engine says (`capabilities`).
    pub fn capabilities(&self) -> Value {
        let props = self.engine.props();
        let modalities = props
            .get("modalities")
            .filter(|m| m.is_object())
            .cloned()
            .unwrap_or(json!({}));
        let caps = props
            .get("chat_template_caps")
            .filter(|m| m.is_object())
            .cloned()
            .unwrap_or(json!({}));
        let limit = match self.config.lock().unwrap().get("chat.max_upload_mb") {
            Some(v) => v.as_i64().unwrap_or(25).clamp(1, 512),
            None => 25,
        };
        json!({"vision": modalities.get("vision").is_some_and(crate::settings::py_truthy), "tools": caps.get("supports_tools").cloned().unwrap_or(Value::Null),
            "n_ctx": self.engine.status()["n_ctx"], "max_upload_mb": limit, "pdf": false})
    }

    /// One tool call (`ToolBox.call`).
    pub fn tools_call(&self, body: &Map<String, Value>) -> Result<Value, ApiError> {
        let s = self
            .config
            .lock()
            .unwrap()
            .get("tools")
            .cloned()
            .unwrap_or(json!({}));
        let name = body
            .get("name")
            .filter(|v| crate::settings::py_truthy(v))
            .map(crate::settings::py_str)
            .unwrap_or_default();
        let Some(tool) = crate::tools::tools(&s).into_iter().find(|t| t.name == name) else {
            return Err(ApiError(
                404,
                "That tool is not available (turned off, or its server is not connected).".into(),
            ));
        };
        if tool.confirm && body.get("approved") != Some(&json!(true)) {
            return Err(ApiError(409, "This tool needs your approval first.".into()));
        }
        let mut args = body
            .get("arguments")
            .filter(|v| crate::settings::py_truthy(v))
            .cloned()
            .unwrap_or(json!({}));
        if let Value::String(t) = &args {
            args = if t.trim().is_empty() {
                json!({})
            } else {
                match serde_json::from_str::<Value>(t) {
                    Ok(v) => v,
                    Err(_) => {
                        return Ok(
                            json!({"content": "The arguments are not valid JSON.", "is_error": true, "ms": 0}),
                        )
                    }
                }
            };
        }
        if !args.is_object() {
            return Ok(
                json!({"content": "The arguments must be a JSON object.", "is_error": true, "ms": 0}),
            );
        }
        let t0 = std::time::Instant::now();
        let (content, is_error) = match crate::tools::run(&tool, &s, &args) {
            Ok(c) => (c, false),
            Err(e) => (e.0, true),
        };
        Ok(json!({"content": content, "is_error": is_error, "ms": t0.elapsed().as_millis() as i64}))
    }

    pub fn sysinfo(&self) -> Value {
        crate::system::info(&self.models_dir())
    }

    /// Which of the two MTP files are in the models folder (`extras_view`).
    pub fn extras_view(&self) -> Value {
        let root = self.models_dir();
        let one = |r: &Value| {
            let dest = root
                .join(r["repo"].as_str().unwrap_or(""))
                .join(r["path"].as_str().unwrap_or(""));
            json!({"name": r["name"], "size": r["size"], "present": dest.exists()})
        };
        json!({"mtp_head": one(crate::catalog::mtp_head()), "mtp_vocab": one(crate::catalog::mtp_vocab())})
    }

    /// The files of a catalog model (and its MTP head and vocabulary): name for the job, and `[{url, dest, size, sha256}]`.
    pub fn catalog_files(
        ident: &str,
        with_mtp: bool,
        root: &std::path::Path,
    ) -> Result<(String, Vec<Value>), ApiError> {
        let model = crate::catalog::by_id(ident)
            .ok_or_else(|| ApiError(404, "Unknown catalog model".into()))?;
        let mut refs: Vec<Value> = model["files"].as_array().cloned().unwrap_or_default();
        if with_mtp && model["mtp"].as_bool() == Some(true) {
            refs.push(crate::catalog::mtp_head().clone());
            refs.push(crate::catalog::mtp_vocab().clone());
        }
        let files = refs
            .iter()
            .map(|r| {
                let (repo, path) = (r["repo"].as_str().unwrap_or(""), r["path"].as_str().unwrap_or(""));
                json!({"url": crate::hf::resolve_url(repo, path, "main"), "dest": root.join(repo).join(path).to_string_lossy(), "size": r["size"], "sha256": r["sha256"]})
            })
            .collect();
        Ok((model["name"].as_str().unwrap_or("").to_string(), files))
    }

    /// Queue a catalog model for download (`download_catalog`).
    pub fn download_catalog(&self, ident: &str, with_mtp: bool) -> Result<Value, ApiError> {
        let (name, files) = Self::catalog_files(ident, with_mtp, &self.models_dir())?;
        Ok(self.downloads.add(
            &name,
            &files,
            "model",
            json!({"catalog_id": ident}),
            &format!("catalog:{ident}"),
        ))
    }

    /// The catalog with what is already on disk and what is being downloaded (`catalog_view`).
    pub fn catalog_view(&self) -> Value {
        let info = crate::system::info(&self.models_dir());
        let models = self.library.lock().unwrap().scan(&self.models_dir());
        let jobs = self.downloads.list();
        let extras = self.extras_view();
        let mut out = vec![];
        for mut entry in crate::catalog::catalog(&info) {
            let id = entry["id"].clone();
            entry.insert(
                "mtp_status".into(),
                crate::catalog::mtp_status(
                    entry.get("mtp").is_some_and(crate::settings::py_truthy),
                    extras["mtp_head"]["present"] == json!(true),
                    extras["mtp_vocab"]["present"] == json!(true),
                ),
            );
            let local = models
                .iter()
                .rev()
                .find(|m| !m["catalog_id"].is_null() && m["catalog_id"] == id)
                .cloned();
            let key = format!("catalog:{}", id.as_str().unwrap_or(""));
            let job = jobs
                .iter()
                .rev()
                .find(|j| {
                    j["dedupe"] == key.as_str()
                        && !matches!(j["state"].as_str(), Some("done" | "cancelled"))
                })
                .cloned();
            entry.insert("local".into(), local.unwrap_or(Value::Null));
            entry.insert("job".into(), job.unwrap_or(Value::Null));
            out.push(Value::Object(entry));
        }
        Value::Array(out)
    }
}

/// `/api/downloads/<hex>/<pause|resume|cancel|remove>` -> (id, action)
fn download_action(p: &str) -> Option<(&str, &str)> {
    let rest = p.strip_prefix("/api/downloads/")?;
    let (id, what) = rest.split_once('/')?;
    (!id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        && matches!(what, "pause" | "resume" | "cancel" | "remove"))
    .then_some((id, what))
}

/// `/api/catalog/<id>/download` -> the catalog id (`[\w.-]+`)
fn catalog_download_id(p: &str) -> Option<&str> {
    let id = p.strip_prefix("/api/catalog/")?.strip_suffix("/download")?;
    (!id.is_empty()
        && id
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '-')))
    .then_some(id)
}

/// `/api/engines/<id>/remove` -> the engine id (`[\w.-]+`)
fn engine_remove_id(p: &str) -> Option<&str> {
    let id = p.strip_prefix("/api/engines/")?.strip_suffix("/remove")?;
    (!id.is_empty()
        && id
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '-')))
    .then_some(id)
}

/// `/api/profiles/<id>/delete` -> the profile id (`[\w.-]+`)
fn profile_delete_id(p: &str) -> Option<&str> {
    let id = p.strip_prefix("/api/profiles/")?.strip_suffix("/delete")?;
    (!id.is_empty()
        && id
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '-')))
    .then_some(id)
}

fn engine_err(e: EngineErr) -> ApiError {
    match e {
        EngineErr::Busy(m) => ApiError(409, m),
        EngineErr::Invalid(m) => ApiError(400, m),
    }
}

/// `/api/engine/service/<start|stop|restart>` -> the action
fn engine_service_action(p: &str) -> Option<&str> {
    p.strip_prefix("/api/engine/service/")
        .filter(|a| matches!(*a, "start" | "stop" | "restart"))
}

/// A call under `/api/p/<id>/`: no plugin is installed in this program, so none is on.
fn plugin_off(p: &str) -> ApiError {
    let rest = &p["/api/p/".len()..];
    let id = rest.split('/').next().unwrap_or("");
    let ok = !id.is_empty()
        && id.len() <= 32
        && id.as_bytes()[0].is_ascii_lowercase()
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    if ok {
        ApiError(404, format!("The plugin {id} is not on"))
    } else {
        ApiError(404, "Not found".into())
    }
}

/// `/api/plugins/<id>/<enable|disable>` -> the plugin id
fn plugin_toggle(p: &str) -> Option<&str> {
    let rest = p.strip_prefix("/api/plugins/")?;
    let (id, act) = rest.split_once('/')?;
    let ok = !id.is_empty()
        && id.len() <= 32
        && id.as_bytes()[0].is_ascii_lowercase()
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    (ok && matches!(act, "enable" | "disable")).then_some(id)
}

/// `/api/models/<12 hex>/remove` -> the id
fn model_remove_id(p: &str) -> Option<&str> {
    let id = p.strip_prefix("/api/models/")?.strip_suffix("/remove")?;
    (id.len() == 12
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
    .then_some(id)
}

fn route(st: &Ctx, r: &Req) -> Option<Result<Value, ApiError>> {
    let (m, p) = (r.method.as_str(), r.path.as_str());
    Some(match (m, p) {
        ("GET", "/api/ping") => Ok(json!({"app": "fle-studio", "version": st.opts.version})),
        ("GET", "/api/system") => Ok(crate::system::info(&st.config.lock().unwrap().models_dir())),
        ("GET", "/api/chat/capabilities") => Ok(st.capabilities()),
        ("GET", "/api/tools") => Ok(crate::tools::status(
            &st.config
                .lock()
                .unwrap()
                .get("tools")
                .cloned()
                .unwrap_or(json!({})),
        )),
        ("POST", "/api/tools/reload") => Ok(crate::tools::status(
            &st.config
                .lock()
                .unwrap()
                .get("tools")
                .cloned()
                .unwrap_or(json!({})),
        )),
        ("POST", "/api/tools/call") => r.json(MAX_JSON).and_then(|body| st.tools_call(&body)),
        ("GET", "/api/state") => Ok(st.state()),
        ("GET", "/api/settings") => Ok(st.config.lock().unwrap().public()),
        ("POST", "/api/settings") => r.json(MAX_JSON).and_then(|body| {
            st.config
                .lock()
                .unwrap()
                .update(&body)
                .map_err(|e| ApiError(400, e))
        }),
        ("GET", "/api/models") => {
            let models = st.library.lock().unwrap().scan(&st.models_dir());
            Ok(json!({"models": models, "extras": st.extras_view()}))
        }
        ("POST", "/api/models/add") => r.json(MAX_JSON).and_then(|body| {
            let path = body
                .get("path")
                .filter(|v| crate::settings::py_truthy(v))
                .map(crate::settings::py_str)
                .unwrap_or_default();
            st.library
                .lock()
                .unwrap()
                .add(&st.models_dir(), &path)
                .map_err(|e| ApiError(400, e))
        }),
        ("POST", _) if model_remove_id(p).is_some() => r.json(MAX_JSON).and_then(|body| {
            let delete = body.get("delete").is_some_and(crate::settings::py_truthy);
            match st.library.lock().unwrap().remove(
                &st.models_dir(),
                model_remove_id(p).unwrap(),
                delete,
            ) {
                Ok(()) => Ok(json!({"ok": true})),
                Err(LibError::NotFound) => Err(ApiError(404, "Not found".into())),
                Err(LibError::Invalid(m)) => Err(ApiError(400, m)),
            }
        }),
        ("POST", _) if catalog_download_id(p).is_some() => r.json(MAX_JSON).and_then(|body| {
            let mtp = body
                .get("mtp")
                .map(crate::settings::py_truthy)
                .unwrap_or(true);
            st.download_catalog(catalog_download_id(p).unwrap(), mtp)
        }),
        ("GET", "/api/engines") => {
            let info = st.sysinfo();
            let (os, arch) = crate::engines::os_arch();
            Ok(
                json!({"installed": st.engines.installed(), "variants": crate::engines::variants(&info, os, &arch)}),
            )
        }
        ("POST", "/api/engines/install") => r.json(MAX_JSON).and_then(|body| {
            let variant = body
                .get("variant")
                .filter(|v| crate::settings::py_truthy(v))
                .map(crate::settings::py_str)
                .unwrap_or_default();
            let info = st.sysinfo();
            let (os, arch) = crate::engines::os_arch();
            if !crate::engines::variants(&info, os, &arch)
                .iter()
                .any(|v| v["id"] == variant.as_str())
            {
                return Err(ApiError(
                    400,
                    "That engine does not fit this machine".into(),
                ));
            }
            st.engines.install(&variant, &info).map_err(|e| match e {
                EngineError::Runtime(m) => ApiError(409, m),
                EngineError::Network(m) => ApiError(502, format!("Could not reach GitHub: {m}")),
                EngineError::NotFound => ApiError(404, "Not found".into()),
            })
        }),
        ("POST", "/api/engines/check") => r.json(MAX_JSON).map(|body| {
            let path = body
                .get("path")
                .filter(|v| crate::settings::py_truthy(v))
                .map(crate::settings::py_str)
                .unwrap_or_default();
            st.engines.detect(&path)
        }),
        ("POST", _) if engine_remove_id(p).is_some() => {
            match st.engines.remove(engine_remove_id(p).unwrap()) {
                Ok(()) => Ok(json!({"ok": true})),
                Err(_) => Err(ApiError(404, "Not found".into())),
            }
        }
        ("GET", "/api/profiles") => Ok(
            json!({"profiles": st.profiles.lock().unwrap().all().iter().map(|p| p.public()).collect::<Vec<_>>()}),
        ),
        ("POST", "/api/profiles") => r.json(MAX_JSON).and_then(|body| {
            st.profiles
                .lock()
                .unwrap()
                .save(&body)
                .map(|p| p.public())
                .map_err(|e| ApiError(400, e.0))
        }),
        ("POST", _) if profile_delete_id(p).is_some() => {
            let id = profile_delete_id(p).unwrap();
            st.profiles.lock().unwrap().delete(id);
            let mut cfg = st.config.lock().unwrap();
            if cfg.get("launch.profile").and_then(Value::as_str) == Some(id) {
                let patch = json!({"launch": {"profile": "auto"}});
                let _ = cfg.update(patch.as_object().unwrap());
            }
            Ok(json!({"ok": true}))
        }
        ("GET", "/api/live") => {
            let since = r
                .query
                .get("since")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|v| !v.is_empty());
            match since.map(|v| v.parse::<f64>()) {
                None => Ok(st.live.snapshot(0.0)),
                Some(Ok(f)) => Ok(st.live.snapshot(f)),
                Some(Err(_)) => Err(ApiError(
                    400,
                    format!(
                        "could not convert string to float: {}",
                        crate::settings::py_repr(&json!(since.unwrap_or("")))
                    ),
                )),
            }
        }
        ("GET", "/api/engine") => Ok(st.engine.status()),
        ("GET", "/api/engine/log") => Ok(st.engine.log_since(
            r.query
                .get("since")
                .and_then(|v| v.as_str())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0),
            2000,
        )),
        ("POST", "/api/engine/plan") => r.json(MAX_JSON).map(|body| {
            let model = body
                .get("model")
                .filter(|v| crate::settings::py_truthy(v))
                .map(crate::settings::py_str)
                .unwrap_or_default();
            st.plan_view(&model, body.get("profile"))
        }),
        (_, _) if p.starts_with("/api/p/") => Err(plugin_off(p)),
        ("GET", "/api/plugins") => Ok(json!({"api_version": 1, "plugins": [], "error": ""})),
        ("POST", _) if plugin_toggle(p).is_some() => Err(ApiError(
            404,
            format!("No plugin named {} is installed", plugin_toggle(p).unwrap()),
        )),
        ("POST", "/api/engine/preview") => r.json(MAX_JSON).and_then(|body| {
            let model = body
                .get("model")
                .filter(|v| crate::settings::py_truthy(v))
                .map(crate::settings::py_str)
                .unwrap_or_default();
            st.make_plan(&model, body.get("profile"))
                .map(|p| p.public())
        }),
        ("POST", "/api/engine/start") => r.json(MAX_JSON).and_then(|body| {
            let last = st
                .config
                .lock()
                .unwrap()
                .get("last_model")
                .map(crate::settings::py_str)
                .unwrap_or_default();
            let model = body
                .get("model")
                .filter(|v| crate::settings::py_truthy(v))
                .map(crate::settings::py_str)
                .unwrap_or(last);
            let cfg_profile = st.config.lock().unwrap().get("launch.profile").cloned();
            let profile = body
                .get("profile")
                .filter(|v| crate::settings::py_truthy(v))
                .cloned()
                .or(cfg_profile);
            let plan = st.make_plan(&model, profile.as_ref())?;
            let roc = st
                .config
                .lock()
                .unwrap()
                .get("launch.restart_on_crash")
                .map(crate::settings::py_truthy)
                .unwrap_or(false);
            st.engine.set_restart_on_crash(roc);
            st.engine.start(plan).map_err(engine_err)?;
            let shown = profile
                .filter(crate::settings::py_truthy)
                .unwrap_or(json!("auto"));
            let patch = json!({"last_model": model, "launch": {"profile": shown}});
            st.config
                .lock()
                .unwrap()
                .update(patch.as_object().unwrap())
                .map_err(|e| ApiError(400, e))?;
            Ok(st.engine.status())
        }),
        ("POST", "/api/engine/stop") => {
            if !st.engine.bound().is_empty() {
                let _ = st
                    .config
                    .lock()
                    .unwrap()
                    .update(json!({"attach_auto": false}).as_object().unwrap());
            }
            st.engine.stop();
            Ok(st.engine.status())
        }
        ("POST", "/api/engine/attach") => r.json(MAX_JSON).and_then(|body| {
            let get = |k: &str| {
                body.get(k)
                    .filter(|v| crate::settings::py_truthy(v))
                    .map(crate::settings::py_str)
                    .unwrap_or_default()
            };
            let (url, unit) = (get("url"), get("service").trim().to_string());
            let mut patch = json!({"attach_url": url, "attach_service": unit});
            if let Some(k) = body.get("key").filter(|k| !k.is_null()) {
                patch["attach_key"] = json!(crate::settings::py_str(k));
            }
            st.config
                .lock()
                .unwrap()
                .update(patch.as_object().unwrap())
                .map_err(|e| ApiError(400, e))?;
            let key = st
                .config
                .lock()
                .unwrap()
                .get("attach_key")
                .map(crate::settings::py_str)
                .unwrap_or_default();
            let status = st.engine.attach(&url, &key, &unit).map_err(engine_err)?;
            let _ = st
                .config
                .lock()
                .unwrap()
                .update(json!({"attach_auto": true}).as_object().unwrap());
            Ok(status)
        }),
        ("POST", _) if engine_service_action(p).is_some() => Err(ApiError(
            400,
            "FLE Studio is not connected to a service.".into(),
        )),
        ("GET", "/api/catalog") => {
            Ok(json!({"catalog": st.catalog_view(), "extras": st.extras_view()}))
        }
        ("POST", _) if download_action(p).is_some() => {
            let (id, what) = download_action(p).unwrap();
            st.downloads
                .action(id, what)
                .ok_or(ApiError(404, "Not found".into()))
        }
        ("GET", "/api/downloads") => Ok(json!({"downloads": st.downloads.list()})),
        ("GET", "/api/chats") => Ok(json!({"chats": st.chats.list()})),
        ("POST", "/api/chats") => r.json(MAX_CHAT).and_then(|body| {
            let _g = st.chat_lock.lock().unwrap();
            st.chats.save(None, &body).map_err(chat_err)
        }),
        ("GET", _) if chat_id(p).is_some() => st.chats.get(chat_id(p).unwrap()).map_err(chat_err),
        ("PUT", _) if chat_id(p).is_some() => r.json(MAX_CHAT).and_then(|body| {
            let _g = st.chat_lock.lock().unwrap();
            st.chats.save(chat_id(p), &body).map_err(chat_err)
        }),
        ("DELETE", _) if chat_id(p).is_some() => st
            .chats
            .delete(chat_id(p).unwrap())
            .map(|_| json!({"ok": true}))
            .map_err(chat_err),
        _ => return None,
    })
}

const STATIC_TYPES: [(&str, &str); 12] = [
    (".html", "text/html"),
    (".js", "text/javascript"),
    (".mjs", "text/javascript"),
    (".css", "text/css"),
    (".svg", "image/svg+xml"),
    (".json", "application/json"),
    (".png", "image/png"),
    (".ico", "image/x-icon"),
    (".woff2", "font/woff2"),
    (".woff", "font/woff"),
    (".txt", "text/plain"),
    (".map", "application/json"),
];

/// The built web page: a folder when one is given (tests, development), else the copy inside the binary.
static WEB: include_dir::Dir<'static> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../../web/studio");

fn read_web(st: &Studio, name: &str) -> Option<Vec<u8>> {
    match &st.opts.web_dir {
        Some(dir) => std::fs::read(dir.join(name)).ok(),
        None => WEB.get_file(name).map(|f| f.contents().to_vec()),
    }
}

fn static_file(st: &Studio, path: &str, head: bool) -> Response<Body> {
    let mut name = if path == "/" || path == "/index.html" {
        "index.html".to_string()
    } else {
        path.trim_start_matches('/').to_string()
    };
    let ok = name
        .chars()
        .next()
        .is_some_and(|c| c.is_alphanumeric() || c == '_')
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '/' | '-'))
        && !name.contains("..");
    let not_found = || error_json(404, "Not found", head);
    if !ok {
        return not_found();
    }
    if let Some(rest) = name.strip_prefix("plugins/") {
        // a plugin's own files: none is installed in this program
        let id = rest.split('/').next().unwrap_or("");
        let valid = !id.is_empty()
            && id.len() <= 32
            && id.as_bytes()[0].is_ascii_lowercase()
            && id
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
            && rest.contains('/');
        return error_json(
            404,
            &if valid {
                format!("The plugin {id} is not on")
            } else {
                "Not found".to_string()
            },
            head,
        );
    }
    let mut data: Option<Vec<u8>> = read_web(st, &name);
    if data.is_none() {
        if !name.rsplit('/').next().unwrap_or("").contains('.') {
            name = "index.html".into(); // client-side routes fall back to the app
            data = read_web(st, &name);
        } else {
            return not_found();
        }
    }
    let Some(mut data) = data else {
        return not_found();
    };
    if name == "index.html" {
        let t = String::from_utf8_lossy(&data).replace("__FLE_TOKEN__", &st.token);
        data = t.into_bytes();
    }
    let suffix = name
        .rfind('.')
        .map(|i| name[i..].to_lowercase())
        .unwrap_or_default();
    let mut ctype = STATIC_TYPES
        .iter()
        .find(|(s, _)| *s == suffix)
        .map(|(_, c)| c.to_string())
        .unwrap_or_else(|| "application/octet-stream".into());
    if ctype.starts_with("text/") || ctype == "application/json" || ctype == "image/svg+xml" {
        ctype += "; charset=utf-8";
    }
    let csp = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'";
    send(
        200,
        data,
        &ctype,
        &[
            ("Content-Security-Policy", csp),
            ("X-Frame-Options", "DENY"),
            ("X-Content-Type-Options", "nosniff"),
            ("Cache-Control", "no-cache"),
        ],
        head,
    )
}

async fn handler(
    State(st): State<Ctx>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response<Body> {
    let (parts, body) = req.into_parts();
    let head = parts.method == Method::HEAD;
    let path = parts.uri.path().to_string();
    if let Some(r) = guard(&st, &parts.method, &path, &parts.headers, &peer, head) {
        return r;
    }
    if path.starts_with("/v1/") {
        return proxy(&st, parts, body, head).await;
    }
    let declared: usize = parts
        .headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let (body, too_large) = if declared > MAX_CHAT {
        (vec![], true)
    } else {
        match to_bytes(body, MAX_CHAT).await {
            Ok(b) => (b.to_vec(), false),
            Err(_) => (vec![], true),
        }
    };
    let r = Req {
        method: parts.method.clone(),
        path: path.clone(),
        query: query_of(parts.uri.query()),
        body,
        too_large,
    };
    let st2 = st.clone();
    let routed = tokio::task::spawn_blocking(move || route(&st2, &r))
        .await
        .unwrap_or(None);
    match routed {
        Some(Ok(v)) => send_json(200, &v, head),
        Some(Err(ApiError(s, m))) => error_json(s, &m, head),
        None if matches!(parts.method, Method::GET | Method::HEAD)
            && !path.starts_with("/api/") =>
        {
            static_file(&st, &path, head)
        }
        None => match not_yet(&path, &parts.method) {
            Some(what) => error_json(
                501,
                &format!("{what} is not in this version of FLE yet."),
                head,
            ),
            None => error_json(404, "Not found", head),
        },
    }
}

/// Features of the Python Studio that this server does not have yet: a clear 501 instead of a 404 that looks like a bug.
fn not_yet(path: &str, _method: &Method) -> Option<&'static str> {
    let is = |p: &str| path == p || path.starts_with(&format!("{p}/"));
    Some(if is("/api/hf") {
        "Searching and downloading from Hugging Face"
    } else if is("/api/bench") {
        "The speed test"
    } else if is("/api/forge") {
        "The model forge"
    } else if is("/api/tuning") {
        "The tuner"
    } else if is("/api/quality") {
        "The quality measurements"
    } else if is("/api/cache") {
        "The expert cache view"
    } else if is("/api/flags") {
        "The engine flag help"
    } else if is("/api/mcp") || is("/oauth") || is("/api/apps") || is("/app-frame") {
        "MCP servers"
    } else if is("/api/remote") || is("/h") || is("/api/hosts") {
        "Remote access"
    } else if is("/api/update") {
        "Updating from inside the program"
    } else {
        return None;
    })
}

pub fn router(st: Ctx) -> Router {
    Router::new().fallback(handler).with_state(st)
}

/// Serve on an open socket until `shutdown` finishes.
pub async fn serve_listener(
    st: Ctx,
    l: tokio::net::TcpListener,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(
        l,
        router(st).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown)
    .await
}

/// Serve until the process ends. Returns the bound address through `ready` once the socket is open.
pub async fn serve(
    st: Ctx,
    addr: SocketAddr,
    ready: Option<std::sync::mpsc::Sender<SocketAddr>>,
) -> std::io::Result<()> {
    let l = tokio::net::TcpListener::bind(addr).await?;
    if let Some(r) = ready {
        let _ = r.send(l.local_addr()?);
    }
    serve_listener(st, l, std::future::pending()).await
}

/// Refuse a chat request that asks for tools and for a forced answer format together (`screen_proxy_body`).
fn screen_proxy_body(path: &str, body: &[u8]) -> Result<(), String> {
    let p = path.split('?').next().unwrap_or("").trim_end_matches('/');
    if body.is_empty()
        || !p.ends_with("/chat/completions")
        || !body.windows(7).any(|w| w == b"\"tools\"")
    {
        return Ok(());
    }
    let Ok(Value::Object(doc)) = serde_json::from_slice::<Value>(body) else {
        return Ok(());
    }; // the engine will say what is wrong
    if !doc.get("tools").is_some_and(crate::settings::py_truthy)
        || doc.get("tool_choice") == Some(&json!("none"))
    {
        return Ok(());
    }
    let mut clash = vec![];
    for key in ["response_format", "json_schema", "grammar"] {
        let v = doc.get(key).cloned().unwrap_or(Value::Null);
        if key == "response_format" {
            if let Value::Object(o) = &v {
                if matches!(o.get("type"), None | Some(Value::Null))
                    || o.get("type") == Some(&json!("text"))
                {
                    continue;
                }
            }
        }
        if !(v.is_null() || v == json!("") || v == json!({}) || v == json!([])) {
            clash.push(key);
        }
    }
    if clash.is_empty() {
        return Ok(());
    }
    Err(format!("A chat request must not carry both tools and {}: the model would not call the tools and would answer with substitute JSON. Send them in separate requests.", clash.join(", ")))
}

enum Upstream {
    Failed(String),
    Answer {
        status: u16,
        ctype: String,
        stream: bool,
    },
}

/// Pass /v1/* to the engine; streams (SSE) are forwarded as they arrive.
async fn proxy(
    st: &Ctx,
    parts: axum::http::request::Parts,
    body: Body,
    head: bool,
) -> Response<Body> {
    let (state, base, key) = st.engine.target();
    if !(state == "ready" || state == "attached") || base.is_empty() {
        return send_json(
            503,
            &json!({"error": {"message": "No model is loaded. Load one in FLE Studio first.", "type": "unavailable_error", "code": 503}}),
            head,
        );
    }
    let declared: usize = parts
        .headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if declared > MAX_PROXY {
        return error_json(413, "Request too large", head);
    }
    let Ok(bytes) = to_bytes(body, MAX_PROXY).await else {
        return error_json(413, "Request too large", head);
    };
    let path_q = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| parts.uri.path().to_string());
    if let Err(m) = screen_proxy_body(&path_q, &bytes) {
        return error_json(400, &m, head);
    }
    let method = parts.method.to_string();
    let ctype = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let accept = parts
        .headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("*/*")
        .to_string();
    let live = st.live.clone();
    live.touch();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(16);
    let (meta_tx, meta_rx) = tokio::sync::oneshot::channel::<Upstream>();
    std::thread::spawn(move || {
        let url = format!("{}{}", base.trim_end_matches('/'), path_q);
        let agent = ureq::AgentBuilder::new()
            .redirects(0)
            .timeout_connect(Duration::from_secs(900))
            .timeout_read(Duration::from_secs(900))
            .build();
        let mut req = agent
            .request(&method, &url)
            .set("Content-Type", &ctype)
            .set("Accept", &accept);
        if !key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let result = if bytes.is_empty() && method == "GET" {
            req.call()
        } else {
            req.send_bytes(&bytes)
        };
        let resp = match result {
            Ok(r) => r,
            Err(ureq::Error::Status(_, r)) => r,
            Err(e) => {
                use std::error::Error;
                let mut cur: &dyn Error = &e;
                while let Some(s) = cur.source() {
                    cur = s;
                }
                let _ = meta_tx.send(Upstream::Failed(cur.to_string()));
                return;
            }
        };
        let rtype = resp
            .header("Content-Type")
            .unwrap_or("application/json")
            .to_string();
        let stream = rtype.contains("text/event-stream");
        let _ = meta_tx.send(Upstream::Answer {
            status: resp.status(),
            ctype: rtype,
            stream,
        });
        let mut reader = resp.into_reader();
        let mut all: Vec<u8> = vec![];
        let mut buf = vec![0u8; 65536];
        loop {
            let n = match std::io::Read::read(&mut reader, &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            if stream {
                if tx
                    .blocking_send(Ok(bytes::Bytes::copy_from_slice(&buf[..n])))
                    .is_err()
                {
                    break; // the browser stopped reading (Stop button); closing our side cancels the generation
                }
                all.extend_from_slice(&buf[..n]);
                if all.len() > 262144 {
                    all.drain(..all.len() - 262144);
                }
            } else {
                all.extend_from_slice(&buf[..n]);
            }
        }
        if stream {
            record_stream(&live, &all);
        } else {
            record_data(&live, &all);
            let _ = tx.blocking_send(Ok(bytes::Bytes::from(all)));
        }
    });
    let mut rx = rx;
    match meta_rx.await {
        Err(_) => error_json(502, "The engine did not answer", head),
        Ok(Upstream::Failed(why)) => send_json(
            502,
            &json!({"error": {"message": format!("The engine did not answer: {why}"), "type": "bad_gateway"}}),
            head,
        ),
        Ok(Upstream::Answer {
            status,
            ctype,
            stream: false,
        }) => {
            let mut data = vec![];
            while let Some(Ok(c)) = rx.recv().await {
                data.extend_from_slice(&c);
            }
            send(status, data, &ctype, &[], head)
        }
        Ok(Upstream::Answer {
            status,
            ctype,
            stream: true,
        }) => {
            let mut r = Response::new(Body::from_stream(
                tokio_stream::wrappers::ReceiverStream::new(rx),
            ));
            *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::OK);
            let h = r.headers_mut();
            h.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_str(&ctype)
                    .unwrap_or(HeaderValue::from_static("text/event-stream")),
            );
            h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            h.insert("x-accel-buffering", HeaderValue::from_static("no"));
            h.insert(header::CONNECTION, HeaderValue::from_static("close"));
            r
        }
    }
}

fn record_data(live: &Live, data: &[u8]) {
    if let Ok(Value::Object(o)) = serde_json::from_slice::<Value>(data) {
        if let Some(t) = o.get("timings").filter(|t| crate::settings::py_truthy(t)) {
            live.record_response(t, o.get("usage"));
        }
    }
}

fn record_stream(live: &Live, tail: &[u8]) {
    let text = String::from_utf8_lossy(tail).to_string();
    for line in text.lines().rev() {
        if line.starts_with("data:") && line.contains("\"timings\"") {
            let Ok(obj) = serde_json::from_str::<Value>(line[5..].trim()) else {
                continue;
            };
            live.record_response(obj.get("timings").unwrap_or(&json!({})), obj.get("usage"));
            return;
        }
    }
}
