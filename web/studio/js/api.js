// Talking to the FLE Studio server. Every call that changes something carries the per-run token from the page.

const TOKEN = document.querySelector('meta[name="fle-token"]')?.content || "";

class ApiError extends Error {
  constructor(status, message, details = null) {
    super(message);
    this.status = status;
    this.code = details?.code || "";
    this.details = details;
  }
}

// ----------------------------------------------------------------------------------------------- transport
// Where a call goes is a plugin's business: with the Remote access plugin and another host chosen, every /api/ and /v1/
// request goes to /h/<host id>/…, which this Studio passes on to the host. The plugin fills in this object before the
// page asks for anything (a boot script with `early`). Doing it on fetch itself means pages and helpers that call fetch
// directly need no special case. Without a plugin every path stays as it is.

export const transport = {
  url: (path) => path, // the address to use for a path
  confirm: null, // async (method, path) => bool: a host asks the person at this page to confirm an install
  problem: null, // (details) => void: a call failed because of the host (not reachable, certificate changed)
  gone: null, // (status, data) => bool: true when the answer means the chosen host does not exist any more (handled)
};

// The address to use for a path that the page puts into a link or an <img>/<a download> itself.
export const hostUrl = (path) => transport.url(path);

const nativeFetch = window.fetch.bind(window);
window.fetch = (input, init) => nativeFetch(typeof input === "string" ? hostUrl(input) : input, init);

// FLE Studio was restarted and this page still holds the old run's token: reload once to get the new one.
// At most every 15 s, so a server that keeps refusing cannot put the page into a reload loop.
function reloadForNewSession(res) {
  if (res.status !== 403 || res.headers.get("X-FLE-Session") !== "expired") return false;
  let last = 0;
  try { last = Number(sessionStorage.getItem("fle-session-reload") || 0); } catch { /* storage blocked */ }
  if (Date.now() - last < 15000) return false;
  try { sessionStorage.setItem("fle-session-reload", String(Date.now())); } catch { /* reload anyway */ }
  location.reload();
  return true;
}

async function api(method, path, body) {
  const opts = { method, headers: { "X-FLE-Token": TOKEN } };
  if (body instanceof Blob) { // a file: sent as it is
    opts.headers["Content-Type"] = body.type || "application/octet-stream";
    opts.body = body;
  } else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new ApiError(0, "FLE Studio is not reachable. Is it still running?");
  }
  if (res.status === 401) {
    location.href = "/login";
    throw new ApiError(401, "Sign in first");
  }
  if (reloadForNewSession(res)) throw new ApiError(403, "FLE Studio was restarted; reloading the page");
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = null; }
  // A host asks for a confirmation of installs and updates; the person at this page gives it, the call is repeated.
  if (res.status === 428 && data?.error?.confirm && transport.confirm && !opts.headers["X-FLE-Confirm"]) {
    if (!(await transport.confirm(method, path))) throw new ApiError(0, "Cancelled");
    opts.headers["X-FLE-Confirm"] = "1";
    const again = await fetch(path, opts);
    const raw = await again.text();
    try { data = raw ? JSON.parse(raw) : null; } catch { data = null; }
    if (!again.ok) throw new ApiError(again.status, data?.error?.message || `${again.status} ${again.statusText}`, data?.error);
    return data;
  }
  if (res.status === 404 && transport.gone?.(res.status, data)) {
    throw new ApiError(404, "That host is gone; going back to this machine");
  }
  if (!res.ok) {
    transport.problem?.(data?.error);
    throw new ApiError(res.status, data?.error?.message || `${res.status} ${res.statusText}`, data?.error);
  }
  return data;
}

export const get = (p) => api("GET", p);
export const post = (p, b = {}) => api("POST", p, b);
export const put = (p, b) => api("PUT", p, b);
export const del = (p) => api("DELETE", p);
export const upload = (p, file) => api("POST", p, file); // a File or Blob as the request body

// Streams an OpenAI-style chat completion through Studio's /v1 proxy. onDelta gets every parsed chunk.
export async function streamChat(body, { signal, onChunk }) {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-FLE-Token": TOKEN, Accept: "text/event-stream" },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  if (!res.ok) {
    if (reloadForNewSession(res)) throw new ApiError(403, "FLE Studio was restarted; reloading the page");
    let msg = `${res.status} ${res.statusText}`;
    let details = null;
    try { details = (await res.json())?.error || null; msg = details?.message || msg; } catch { /* keep status text */ }
    transport.problem?.(details);
    throw new ApiError(res.status, msg, details);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, cut).trim();
      buf = buf.slice(cut + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return;
      let chunk;
      try { chunk = JSON.parse(data); } catch { continue; }
      if (chunk.error) throw new ApiError(500, chunk.error.message || "The engine reported an error");
      onChunk(chunk);
    }
  }
}
