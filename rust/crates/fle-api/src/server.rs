//! The HTTP side of FLE Studio: the guard (host name, origin, session token), the JSON conventions and the static page.
//! Port of the plumbing in `fle_studio/server.py`; routes live next to the code they call.
use crate::chats::{ChatError, Chats, MAX_BYTES as MAX_CHAT};
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

pub const MAX_JSON: usize = 4 * 1024 * 1024;
const LOOPBACK_NAMES: [&str; 3] = ["127.0.0.1", "::1", "localhost"];

pub struct Options {
    pub home: PathBuf,
    /// The built web page (index.html, js/, css). Later it is embedded in the binary.
    pub web_dir: Option<PathBuf>,
    pub version: String,
}

pub struct Studio {
    pub opts: Options,
    pub token: String,
    pub config: Mutex<Config>,
    pub chats: Chats,
    pub chat_lock: Mutex<()>,
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
    let chats = Chats::new(&opts.home);
    Arc::new(Studio {
        opts,
        token: make_token(),
        config: Mutex::new(config),
        chats,
        chat_lock: Mutex::new(()),
    })
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

fn route(st: &Ctx, r: &Req) -> Option<Result<Value, ApiError>> {
    let (m, p) = (r.method.as_str(), r.path.as_str());
    Some(match (m, p) {
        ("GET", "/api/ping") => Ok(json!({"app": "fle-studio", "version": st.opts.version})),
        ("GET", "/api/system") => Ok(crate::system::info(&st.config.lock().unwrap().models_dir())),
        ("GET", "/api/settings") => Ok(st.config.lock().unwrap().public()),
        ("POST", "/api/settings") => r.json(MAX_JSON).and_then(|body| {
            st.config
                .lock()
                .unwrap()
                .update(&body)
                .map_err(|e| ApiError(400, e))
        }),
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
    let (Some(dir), true) = (&st.opts.web_dir, ok) else {
        return not_found();
    };
    let mut data = std::fs::read(dir.join(&name));
    if data.is_err() {
        if !name.rsplit('/').next().unwrap_or("").contains('.') {
            name = "index.html".into(); // client-side routes fall back to the app
            data = std::fs::read(dir.join(&name));
        } else {
            return not_found();
        }
    }
    let Ok(mut data) = data else {
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
        None => error_json(404, "Not found", head),
    }
}

pub fn router(st: Ctx) -> Router {
    Router::new().fallback(handler).with_state(st)
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
    axum::serve(
        l,
        router(st).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
}
