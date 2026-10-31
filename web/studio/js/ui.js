// Small DOM toolkit: element builder, icons, number formats, toasts, dialogs and the two live-data shapes
// (sparkline, ring). No framework; every page builds plain elements with h().

const SVG_NS = "http://www.w3.org/2000/svg";

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  append(el, children);
  return el;
}

function setAttrs(el, attrs) {
  if (!attrs) return;
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "html") el.innerHTML = v; // only ever used with HTML we generated and escaped ourselves
    else if (k === "value") el.value = v;
    else if (k === "checked") el.checked = !!v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, String(v));
  }
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function clear(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

function svg(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v !== null && v !== undefined && v !== false) el.setAttribute(k, String(v));
  }
  for (const c of children.flat(Infinity)) if (c) el.append(c);
  return el;
}

// ----------------------------------------------------------------------------------------------- icons

const ICONS = {
  chat: "M5 5.5h14a1.5 1.5 0 0 1 1.5 1.5v8.5A1.5 1.5 0 0 1 19 17h-8.5L6 20.5V17H5a1.5 1.5 0 0 1-1.5-1.5V7A1.5 1.5 0 0 1 5 5.5z",
  models: "M12 3 20 7.5v9L12 21l-8-4.5v-9L12 3zM12 12l8-4.5M12 12v9M12 12 4 7.5",
  engine: "M8 6h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2zM10 10h4v4h-4zM9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3",
  lab: "M9.5 3h5M10.5 3v6.2L5.3 18a2 2 0 0 0 1.7 3h10a2 2 0 0 0 1.7-3l-5.2-8.8V3M7.5 15h9",
  forge: "M4 7h13a3 3 0 0 1-3 3h-1v3h2v3H8v-3h2v-3H8a4 4 0 0 1-4-3V7zM17 7h3",
  settings: "M4 7h9M17 7h3M4 12h3M11 12h9M4 17h11M19 17h1M15 5v4M9 10v4M17 15v4",
  plus: "M12 5v14M5 12h14",
  send: "M12 19V5M6 11l6-6 6 6",
  stop: "M7 7h10v10H7z",
  copy: "M9 9h10v11H9zM5 15V4h10",
  check: "M5 12.5l4.5 4.5L19 7.5",
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
  edit: "M4 20h4L19 9l-4-4L4 16v4zM14 6l4 4",
  refresh: "M20 11a8 8 0 1 0-2.3 5.6M20 4v7h-7",
  download: "M12 4v11M7 10l5 5 5-5M5 20h14",
  play: "M8 5.5v13l11-6.5-11-6.5z",
  power: "M12 3v8M7.2 6.2a7 7 0 1 0 9.6 0",
  link: "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
  chevron: "M9 6l6 6-6 6",
  down: "M6 9l6 6 6-6",
  x: "M6 6l12 12M18 6 6 18",
  panel: "M4 5h16v14H4zM15 5v14",
  search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM20 20l-4-4",
  folder: "M3.5 7a1.5 1.5 0 0 1 1.5-1.5h4.5l2 2.5H19a1.5 1.5 0 0 1 1.5 1.5v8A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5V7z",
  bolt: "M13 3 5 13.5h6L10 21l8-10.5h-6L13 3z",
  alert: "M12 4 2.8 19.5h18.4L12 4zM12 10v4.5M12 17.3v.2",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v5.5M12 7.6v.2",
  ok: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM8 12.3l2.7 2.7L16 9.5",
  pause: "M8 5v14M16 5v14",
  menu: "M4 7h16M4 12h16M4 17h16",
  brain: "M9 4.5a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 1V6.5a2 2 0 0 0-3-2zM15 4.5a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 1",
  sliders: "M5 4v16M12 4v16M19 4v16M3 9h4M10 15h4M17 7h4",
  external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  terminal: "M4 5h16v14H4zM8 10l3 2.5L8 15M13 15h4",
  cpu: "M7 7h10v10H7zM10 10h4v4h-4zM9.5 4v3M14.5 4v3M9.5 17v3M14.5 17v3M4 9.5h3M4 14.5h3M17 9.5h3M17 14.5h3",
  grid: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
  layers: "M12 4 3 9l9 5 9-5-9-5zM3 14l9 5 9-5",
  spark: "M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18",
  clip: "M20 11.5l-8 8a5 5 0 0 1-7-7l8.5-8.5a3.2 3.2 0 0 1 4.5 4.5l-8.5 8.5a1.6 1.6 0 0 1-2.3-2.3L15 7.5",
  doc: "M7 3.5h7l4 4V20a.5.5 0 0 1-.5.5h-10a.5.5 0 0 1-.5-.5V4a.5.5 0 0 1 .5-.5zM14 3.5V8h4M9 12.5h6M9 16h6",
  more: "M5 12h.1M12 12h.1M19 12h.1",
  wrench: "M14.5 6a4 4 0 0 0 4.8 5.2L9 21.5a2.1 2.1 0 0 1-3-3L16.2 8.2A4 4 0 0 0 14.5 6z",
  upload: "M12 16V5M7 10l5-5 5 5M5 20h14",
  star: "M12 3.5l2.6 5.4 5.9.8-4.3 4.1 1 5.9L12 16.9l-5.2 2.8 1-5.9-4.3-4.1 5.9-.8L12 3.5z",
  live: "M5 12a7 7 0 0 1 14 0M8.5 12a3.5 3.5 0 0 1 7 0M12 12h.1M5 17.5h14",
};

export function icon(name, cls = "") {
  return svg("svg", { class: `i ${cls}`.trim(), viewBox: "0 0 24 24", "aria-hidden": "true" },
    svg("path", { d: ICONS[name] || ICONS.info }));
}

// The FLE mark: nine experts, three of them active.
export function mark(size = 20) {
  const cells = [[4.5, 4.5, 1], [13, 4.5, 0], [21.5, 4.5, 0], [4.5, 13, 0], [13, 13, 0], [21.5, 13, 1],
    [4.5, 21.5, 0], [13, 21.5, 1], [21.5, 21.5, 0]];
  return svg("svg", { viewBox: "0 0 32 32", width: size, height: size, "aria-hidden": "true" },
    cells.map(([x, y, on]) => svg("rect", {
      x, y, width: 6, height: 6, rx: 1.7,
      fill: on ? "url(#brand-g)" : "currentColor", opacity: on ? 1 : 0.22,
    })));
}

// ----------------------------------------------------------------------------------------------- formats

const nf0 = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const nf2 = new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export const fmt = {
  int: (n) => (n === null || n === undefined || Number.isNaN(n) ? "–" : nf0.format(n)),
  one: (n) => (n === null || n === undefined || Number.isNaN(n) ? "–" : nf1.format(n)),
  two: (n) => (n === null || n === undefined || Number.isNaN(n) ? "–" : nf2.format(n)),
  rate: (n) => (n === null || n === undefined ? "–" : n >= 100 ? nf0.format(n) : nf1.format(n)),
  pct: (x, digits = 0) => (x === null || x === undefined ? "–" :
    new Intl.NumberFormat(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(x * 100) + " %"),
  bytes(n, digits = 1) {
    if (n === null || n === undefined) return "–";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    let v = n;
    while (v >= 1000 && i < units.length - 1) { v /= 1000; i += 1; }
    return `${i === 0 ? nf0.format(v) : new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(v)} ${units[i]}`;
  },
  gib(n, digits = 1) {
    if (n === null || n === undefined) return "–";
    return `${new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(n / 2 ** 30)} GiB`;
  },
  // Memory of one process: GiB when large, MB below that (so a small engine does not read "0 GiB").
  mem(n) {
    return n !== null && n !== undefined && n < 2 ** 30 ? this.bytes(n, 0) : this.gib(n);
  },
  tokens(n) {
    if (n === null || n === undefined) return "–";
    if (n >= 1e6) return `${nf1.format(n / 1e6)}M`;
    if (n >= 10000) return `${nf0.format(n / 1000)}k`;
    if (n >= 1000) return `${nf1.format(n / 1000)}k`;
    return nf0.format(n);
  },
  // Context windows: llama.cpp sizes are powers of two, and "32k" means 32,768 there, not 33k.
  ctx(n) {
    return n && n % 1024 === 0 ? `${nf0.format(n / 1024)}k` : this.tokens(n);
  },
  dur(sec) {
    if (sec === null || sec === undefined || Number.isNaN(sec)) return "–";
    if (sec < 1) return `${nf0.format(sec * 1000)} ms`;
    if (sec < 60) return `${nf1.format(sec)} s`;
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    if (m < 60) return `${m} min ${s} s`;
    return `${Math.floor(m / 60)} h ${m % 60} min`;
  },
  ago(ts) {
    if (!ts) return "";
    const d = (Date.now() / 1000) - ts;
    if (d < 60) return "just now";
    if (d < 3600) return `${Math.floor(d / 60)} min ago`;
    if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
    return new Date(ts * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  },
  date(ts) {
    return new Date(ts * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  },
};

// ----------------------------------------------------------------------------------------------- feedback

export function toast(message, kind = "info", ms = 4200) {
  const box = document.getElementById("toasts");
  const el = h("div", { class: `toast ${kind}`, role: kind === "bad" ? "alert" : "status" },
    icon(kind === "bad" ? "alert" : kind === "good" ? "ok" : "info"), h("div", { class: "grow" }, message));
  box.append(el);
  setTimeout(() => el.remove(), ms);
}

export function confirmDialog({ title, body, ok = "OK", danger = false }) {
  return new Promise((resolve) => {
    const close = (v) => { back.remove(); document.removeEventListener("keydown", onKey); resolve(v); };
    const onKey = (e) => { if (e.key === "Escape") close(false); };
    const okBtn = h("button", { class: `btn ${danger ? "danger" : "primary"}`, onclick: () => close(true) }, ok);
    const back = h("div", { class: "modal-back", onclick: (e) => { if (e.target === back) close(false); } },
      h("div", { class: "modal", role: "dialog", "aria-modal": "true" },
        h("h3", null, title), body ? h("p", null, body) : null,
        h("div", { class: "row end" }, h("button", { class: "btn ghost", onclick: () => close(false) }, "Cancel"), okBtn)));
    document.body.append(back);
    document.addEventListener("keydown", onKey);
    okBtn.focus();
  });
}

// A dropdown anchored under `anchor`. items: [{label, hint, icon, onclick} | "sep" | {head}]
export function openMenu(anchor, items) {
  document.querySelectorAll(".menu").forEach((m) => m.remove());
  const r = anchor.getBoundingClientRect();
  const menu = h("div", { class: "menu", role: "menu" },
    items.map((it) => {
      if (it === "sep") return h("div", { class: "sep" });
      if (it.head) return h("div", { class: "head" }, it.head);
      return h("button", { role: "menuitem", onclick: () => { menu.remove(); it.onclick(); } },
        it.icon ? icon(it.icon, "sm") : null, h("span", { class: "grow" }, it.label), it.hint ? h("small", null, it.hint) : null);
    }));
  document.body.append(menu);
  const w = menu.offsetWidth;
  const left = Math.max(8, Math.min(window.innerWidth - w - 8, r.left));
  menu.style.left = `${left}px`;
  const below = r.bottom + 6;
  menu.style.top = below + menu.offsetHeight > window.innerHeight - 8 ? `${Math.max(8, r.top - menu.offsetHeight - 6)}px` : `${below}px`;
  const away = (e) => { if (!menu.contains(e.target)) { menu.remove(); document.removeEventListener("mousedown", away, true); } };
  setTimeout(() => document.addEventListener("mousedown", away, true));
  return menu;
}

export async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = h("textarea", { style: { position: "fixed", opacity: "0" } });
    ta.value = text;
    document.body.append(ta);
    ta.select();
    document.execCommand("copy"); // eslint-disable-line -- fallback for http on a LAN address
    ta.remove();
  }
  if (btn) {
    const old = [...btn.childNodes];
    btn.replaceChildren(icon("check", "sm"));
    setTimeout(() => btn.replaceChildren(...old), 1200);
  }
}

export function copyButton(getText, label = "Copy") {
  const b = h("button", { class: "btn ghost icon sm copy", title: label, "aria-label": label });
  b.append(icon("copy", "sm"));
  b.addEventListener("click", (e) => { e.stopPropagation(); copyText(typeof getText === "function" ? getText() : getText, b); });
  return b;
}

// ----------------------------------------------------------------------------------------------- live shapes

// Sparkline: values may contain null (no data in that second); the line breaks there.
export function sparkline(values, { color = "var(--accent)", max = null, height = 44, id = "s" } = {}) {
  const n = values.length;
  const top = max ?? Math.max(1, ...values.filter((v) => v !== null && v !== undefined));
  const W = 300;
  const H = height;
  const x = (i) => (n <= 1 ? W : (i / (n - 1)) * W);
  const y = (v) => H - 3 - (Math.min(v, top) / top) * (H - 8);
  const segs = [];
  let cur = [];
  values.forEach((v, i) => {
    if (v === null || v === undefined) { if (cur.length) segs.push(cur); cur = []; } else cur.push([x(i), y(v)]);
  });
  if (cur.length) segs.push(cur);
  const gid = `g-${id}`;
  const out = svg("svg", { class: "spark", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img" },
    svg("defs", null, svg("linearGradient", { id: gid, x1: 0, y1: 0, x2: 0, y2: 1 },
      svg("stop", { offset: 0, "stop-color": color, "stop-opacity": 0.28 }),
      svg("stop", { offset: 1, "stop-color": color, "stop-opacity": 0 }))),
    svg("line", { class: "base", x1: 0, y1: H - 0.5, x2: W, y2: H - 0.5 }));
  for (const s of segs) {
    if (s.length === 1) {  // a lone sample: draw a short dash that points into the chart, not out of it
      const [a, b] = s[0];
      if (a > W / 2) s.unshift([a - 8, b]); else s.push([a + 8, b]);
    }
    const pts = s.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join(" ");
    out.append(svg("polygon", { class: "ar", points: `${s[0][0].toFixed(1)},${H} ${pts} ${s[s.length - 1][0].toFixed(1)},${H}`, fill: `url(#${gid})` }));
    out.append(svg("polyline", { class: "ln", points: pts, stroke: color }));
  }
  return out;
}

// Ring gauge for a share between 0 and 1.
export function ring(value, { color = "var(--cyan)", label = "" } = {}) {
  const r = 30;
  const c = 2 * Math.PI * r;
  const v = value === null || value === undefined ? null : Math.max(0, Math.min(1, value));
  return h("div", { class: "ring" },
    svg("svg", { viewBox: "0 0 74 74", role: "img", "aria-label": `${label}: ${v === null ? "no data" : Math.round(v * 100) + " %"}` },
      svg("circle", { class: "track", cx: 37, cy: 37, r, fill: "none", "stroke-width": 7 }),
      v === null ? null : svg("circle", {
        cx: 37, cy: 37, r, fill: "none", stroke: color, "stroke-width": 7, "stroke-linecap": "round",
        "stroke-dasharray": `${(c * v).toFixed(1)} ${c.toFixed(1)}`, transform: "rotate(-90 37 37)",
      }),
      (() => { const t = svg("text", { class: "val", x: 37, y: 43, "text-anchor": "middle" }); t.textContent = v === null ? "–" : `${Math.round(v * 100)}%`; return t; })()),
    h("div", { class: "lbl" }, label));
}

export function meter(label, used, total, cls = "", text = null) {
  const share = total ? Math.max(0, Math.min(1, used / total)) : 0;
  return h("div", { class: "meter" },
    h("div", { class: "top" }, h("span", null, label), h("span", null, text ?? `${fmt.gib(used)} / ${fmt.gib(total, 0)}`)),
    h("div", { class: `bar ${cls}` }, h("i", { style: { width: `${(share * 100).toFixed(1)}%` } })));
}

export function stateBadge(state, extra = "") {
  const map = {
    ready: ["good live", "Ready"], attached: ["good live", "Connected"], starting: ["info live", "Loading"],
    stopping: ["warn", "Stopping"], crashed: ["bad", "Stopped with an error"], stopped: ["", "Not running"],
  };
  const [cls, text] = map[state] || ["", state || "–"];
  return h("span", { class: `pill ${cls}` }, h("span", { class: "dot" }), extra ? `${text} ${extra}` : text);
}

export function debounce(fn, ms) {
  let t = 0;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
