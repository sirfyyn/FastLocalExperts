// FLE Studio web app: shared state, the polling loop, the router and the sidebar.

import { del, get } from "./api.js";
import { initChatSearch, searchActive } from "./chat-search.js";
import { initHosts } from "./hosts.js";
import { clear, confirmDialog, fmt, h, icon, stateBadge, toast } from "./ui.js";
import { initUpdates } from "./update.js";

// ----------------------------------------------------------------------------------------------- store

export const store = {
  state: null, // /api/state: settings, system, models, catalog, extras, profiles, engines, downloads, reference
  engine: null, // /api/engine
  live: [], // samples from /api/live, about one per second
  liveMeta: {}, // metrics_available, last_response, totals, cache, accept_logged
  chats: [],
  downloads: [],
  online: true,
};

const subs = new Map();
export function on(topic, fn) {
  if (!subs.has(topic)) subs.set(topic, new Set());
  subs.get(topic).add(fn);
  return () => subs.get(topic).delete(fn);
}
export function emit(topic) {
  for (const fn of subs.get(topic) || []) {
    try { fn(); } catch (e) { console.error(e); }
  }
}

export async function refreshState() {
  const st = await get("/api/state");
  store.state = st;
  store.downloads = st.downloads;
  if (!store.engine) store.engine = st.engine;
  applyTheme();
  emit("state");
  emit("downloads");
  return st;
}

export async function refreshChats() {
  const r = await get("/api/chats");
  store.chats = r.chats;
  renderChatList();
}

export const engineUp = () => ["ready", "attached"].includes(store.engine?.state);
export const engineModelName = () => store.engine?.model?.name || store.engine?.plan?.model?.name || "";

// ----------------------------------------------------------------------------------------------- polling

let lastLive = 0;
let lastEngineKey = "";
const ACTIVE_DL = new Set(["queued", "downloading", "verifying", "installing"]);

async function tick() {
  try {
    const eng = await get("/api/engine");
    const key = JSON.stringify([eng.state, eng.phase, eng.error, eng.load, eng.model?.id, eng.n_ctx, eng.log_seq,
      eng.cache, eng.accept_logged, eng.reachable]);
    const stateChanged = eng.state !== store.engine?.state;
    store.engine = eng;
    if (key !== lastEngineKey) { lastEngineKey = key; emit("engine"); }
    if (stateChanged) renderEngineCard();

    const live = await get(`/api/live?since=${lastLive}`);
    if (live.history.length) {
      store.live.push(...live.history);
      if (store.live.length > 900) store.live.splice(0, store.live.length - 900);
      lastLive = live.history[live.history.length - 1].t;
    }
    store.liveMeta = { metrics_available: live.metrics_available, last_response: live.last_response,
      totals: live.totals, cache: live.cache, accept_logged: live.accept_logged };
    emit("live");
    renderEngineCard();

    const wantDl = store.downloads.some((d) => ACTIVE_DL.has(d.state)) || /^\/(models|engine)/.test(location.pathname);
    if (wantDl) {
      const before = new Map(store.downloads.map((d) => [d.id, d.state]));
      const r = await get("/api/downloads");
      store.downloads = r.downloads;
      emit("downloads");
      renderNav();
      const finished = r.downloads.some((d) => (d.state === "done" || d.state === "failed") && before.get(d.id) && before.get(d.id) !== d.state);
      if (finished) refreshState().catch(() => {});
    }
    if (!store.online) { store.online = true; toast("Connected to FLE Studio again", "good"); }
  } catch (e) {
    if (store.online && e.status === 0) { store.online = false; toast(e.message, "bad", 8000); }
  }
  setTimeout(tick, document.hidden ? 4000 : 1000);
}

// ----------------------------------------------------------------------------------------------- theme

export function applyTheme() {
  const t = store.state?.settings?.theme || "system";
  if (t === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", t);
}

// ----------------------------------------------------------------------------------------------- router

const pages = {
  chat: () => import("./chat.js"),
  models: () => import("./models.js"),
  engine: () => import("./engine.js"),
  lab: () => import("./lab.js"),
  cache: () => import("./cache.js"),
  forge: () => import("./forge.js"),
  settings: () => import("./settings.js"),
};
const routes = [
  [/^\/$/, "chat"], [/^\/c\/([0-9a-f]{16})$/, "chat"], [/^\/models\/?$/, "models"], [/^\/engine\/?$/, "engine"],
  [/^\/lab\/?$/, "lab"], [/^\/cache\/?$/, "cache"], [/^\/forge\/?$/, "forge"], [/^\/settings\/?$/, "settings"],
];
let current = null;
let currentName = "";

export function navigate(path, { replace = false } = {}) {
  if (path === location.pathname + location.search && !replace) return;
  history[replace ? "replaceState" : "pushState"](null, "", path);
  route();
}

async function route() {
  const path = location.pathname;
  let name = "chat";
  let arg = null;
  for (const [re, n] of routes) {
    const m = path.match(re);
    if (m) { name = n; arg = m[1] || null; break; }
  }
  document.getElementById("app").classList.remove("nav-open");
  if (current?.unmount) current.unmount();
  current = null;
  currentName = name;
  const main = document.getElementById("main");
  const mod = await pages[name]();
  if (currentName !== name) return; // the user navigated on while the module loaded
  current = mod.mount(main, arg) || null;
  renderNav();
  renderChatList();
}

export function topbar(title, ...right) {
  const menuBtn = h("button", { class: "btn ghost icon menu-btn", "aria-label": "Menu",
    onclick: () => document.getElementById("app").classList.toggle("nav-open") }, icon("menu"));
  return h("header", { class: "topbar" }, menuBtn,
    typeof title === "string" ? h("h1", null, title) : title, h("div", { class: "grow" }), ...right);
}

// ----------------------------------------------------------------------------------------------- sidebar

const NAV = [
  ["/", "chat", "Chat"], ["/models", "models", "Models"], ["/engine", "engine", "Engine"],
  ["/lab", "lab", "Experiments"], ["/cache", "grid", "Expert cache"], ["/forge", "forge", "Forge"],
  ["/settings", "settings", "Settings"],
];

let navKey = "";
function renderNav() {
  const nav = document.getElementById("nav");
  const path = location.pathname;
  const activeDl = store.downloads.filter((d) => ACTIVE_DL.has(d.state)).length;
  // The live poll calls this every few seconds on /models and /engine; rebuilding the links there
  // replaced the link under a pressed mouse button and the click got lost.
  const key = `${path} ${activeDl}`;
  if (key === navKey && nav.childElementCount) return;
  navKey = key;
  clear(nav, NAV.map(([href, ic, label]) => {
    const active = href === "/" ? path === "/" || path.startsWith("/c/") : path.startsWith(href);
    return h("a", { href, "data-link": true, class: active ? "active" : "" }, icon(ic), label,
      ic === "models" && activeDl ? h("span", { class: "badge", title: "Downloads running" }, activeDl) : null);
  }));
}

function chatGroup(ts) {
  const d = new Date(ts * 1000);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const diff = (today - new Date(d.getFullYear(), d.getMonth(), d.getDate())) / 86400000;
  if (diff <= 0) return "Today";
  if (diff === 1) return "Yesterday";
  if (diff < 7) return "Previous 7 days";
  return "Older";
}

function renderChatList() {
  if (searchActive()) return; // the search box shows its hits in this list
  const list = document.getElementById("chat-list");
  const active = (location.pathname.match(/^\/c\/([0-9a-f]{16})$/) || [])[1];
  if (!store.chats.length) {
    clear(list, h("div", { class: "chat-empty" }, "Your conversations stay on this computer."));
    return;
  }
  const out = [];
  let group = "";
  for (const c of store.chats) {
    const g = chatGroup(c.updated);
    if (g !== group) { group = g; out.push(h("div", { class: "chat-group" }, g)); }
    const x = h("button", { class: "btn ghost icon sm x", title: "Delete chat", "aria-label": "Delete chat",
      onclick: async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const ok = await confirmDialog({ title: "Delete this chat?", body: `“${c.title}” will be removed from this computer.`, ok: "Delete", danger: true });
        if (!ok) return;
        await del(`/api/chats/${c.id}`);
        if (c.id === active) navigate("/");
        refreshChats();
      } }, icon("trash", "sm"));
    out.push(h("a", { class: `chat-item${c.id === active ? " active" : ""}`, href: `/c/${c.id}`, "data-link": true, title: c.title },
      h("span", { class: "t" }, c.title), x));
  }
  clear(list, out);
}

function renderEngineCard() {
  const card = document.getElementById("engine-card");
  const e = store.engine;
  if (!e) return;
  const name = engineModelName();
  const last = store.live[store.live.length - 1];
  const rate = last && last.decode ? `${fmt.rate(last.decode)} tok/s` : "";
  let sub = "Open the engine page";
  if (e.state === "starting") sub = `${e.phase || "Loading"}${e.load?.progress ? ` · ${Math.round(e.load.progress * 100)} %` : ""}`;
  else if (e.state === "crashed") sub = e.error || "The engine stopped";
  else if (e.state === "ready" || e.state === "attached") sub = e.n_ctx ? `Context ${fmt.ctx(e.n_ctx)} tokens` : "Ready";
  else if (e.state === "stopping") sub = e.service ? `Stopping ${e.service.unit}` : "Stopping";
  else if (e.service) sub = `${e.service.unit} is not running`;
  clear(card,
    h("div", { class: "row" }, stateBadge(e.state), h("span", { class: "grow" }), h("span", { class: "rate" }, rate)),
    h("div", { class: "model", title: name }, name || "No model loaded"),
    h("div", { class: "small muted", style: { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }, sub));
}

// ----------------------------------------------------------------------------------------------- start

async function start() {
  document.getElementById("new-chat").append(icon("plus"), h("span", null, "New chat"));
  const card = document.getElementById("engine-card");
  card.addEventListener("click", () => navigate("/engine"));
  card.addEventListener("keydown", (e) => { if (e.key === "Enter") navigate("/engine"); });
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[data-link]");
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || a.origin !== location.origin) return;
    e.preventDefault();
    navigate(a.pathname + a.search);
  });
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); navigate("/"); }
  });
  window.addEventListener("popstate", route);
  initChatSearch({ onClear: () => refreshChats().catch(() => {}) });
  try {
    await refreshState();
  } catch (e) {
    toast(e.message, "bad", 10000);
    return;
  }
  document.getElementById("brand-ver").textContent = `v${store.state.version}`;
  store.engine = store.state.engine;
  renderEngineCard();
  initUpdates();
  initHosts();
  refreshChats().catch(() => {});
  route();
  tick();
}

start();
