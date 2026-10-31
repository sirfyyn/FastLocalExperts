// Expert cache: what the FLE engine says about its GPU expert cache while it runs. The hit rate over time comes from
// the log lines the engine prints today; the per-layer view and the hot experts need two more lines it does not print
// yet (docs/EXPERT-CACHE-DATA.md), and the page says so instead of drawing empty charts.

import { get } from "./api.js";
import { engineUp, navigate, store, topbar } from "./app.js";
import { loadStyle } from "./tuner.js";
import { clear, fmt, h, icon } from "./ui.js";

const NS = "http://www.w3.org/2000/svg";

function line(series, key, { cls = "ln", w = 600, ht = 150, pad = 22 } = {}) {
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${ht}`);
  svg.setAttribute("class", "chart-svg");
  svg.setAttribute("preserveAspectRatio", "none");
  for (const f of [0, 0.25, 0.5, 0.75, 1]) {
    const y = pad / 2 + (ht - pad) * (1 - f);
    const g = document.createElementNS(NS, "line");
    g.setAttribute("x1", 28); g.setAttribute("x2", w); g.setAttribute("y1", y); g.setAttribute("y2", y); g.setAttribute("class", "grid");
    svg.append(g);
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", 0); t.setAttribute("y", y + 3); t.textContent = `${Math.round(f * 100)}`;
    svg.append(t);
  }
  const pts = series.filter((p) => p[key] !== null && p[key] !== undefined);
  if (pts.length > 1) {
    const x0 = pts[0].steps, x1 = pts[pts.length - 1].steps || 1;
    const path = document.createElementNS(NS, "polyline");
    const scaled = key === "fill" ? 100 : 1; // fill is 0..1, rates are 0..100
    path.setAttribute("points", pts.map((p) => {
      const x = 28 + ((p.steps - x0) / Math.max(1, x1 - x0)) * (w - 30);
      const y = pad / 2 + (ht - pad) * (1 - (p[key] * scaled) / 100);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" "));
    path.setAttribute("class", cls);
    svg.append(path);
  }
  return svg;
}

// Low hit rate red, high green; the text on top stays dark.
const heat = (rate) => (rate === null || rate === undefined ? "var(--soft-2)" : `hsl(${Math.round(Math.max(0, Math.min(100, rate)) * 1.2)}, 70%, 60%)`);

export function mount(main) {
  loadStyle();
  const box = h("div", { class: "stack" });
  main.replaceChildren(topbar("Expert cache"), h("div", { class: "page" }, h("div", { class: "page-inner" },
    h("div", { class: "page-head" }, h("div", null, h("h2", null, "Expert cache"),
      h("p", null, "How well the GPU expert cache of the FLE engine works while you use it: which share of the experts a token needs "
        + "are already on the GPU, how full the cache is, and which experts are used the most."))), box)));

  let snap = null;
  let layer = null;
  let timer = 0;
  let gone = false;

  async function load() {
    clearTimeout(timer);
    try {
      snap = await get(`/api/cache${layer !== null ? `?layer=${layer}` : ""}`);
      render();
    } catch (e) {
      clear(box, h("div", { class: "callout bad" }, icon("alert", "sm"), h("div", null, e.message)));
    }
    if (!gone) timer = setTimeout(load, 2000);
  }

  function render() {
    const s = snap;
    const parts = [];
    if (!engineUp() && !s.available.overall) {
      parts.push(h("div", { class: "callout info" }, icon("info", "sm"), h("div", null,
        "No engine is running. Start the FLE engine on the Engine page; the cache numbers appear here once it has generated about 500 tokens.",
        h("div", { style: { marginTop: "8px" } }, h("button", { class: "btn", onclick: () => navigate("/engine") }, icon("engine", "sm"), "Engine")))));
    }
    const o = s.overall;
    parts.push(h("div", { class: "card pad-lg" },
      h("div", { class: "cache-top" },
        fact("Hit rate, whole run", o.hit_rate !== null ? `${o.hit_rate.toFixed(1)} %` : "–", "Experts found in the cache out of all that were needed"),
        fact("Hit rate, last interval", o.recent_rate !== null ? `${o.recent_rate.toFixed(1)} %` : "–", "Between the two newest log lines"),
        fact("Cache", s.enabled ? `${s.enabled.layers} layers × ${s.enabled.slots}` : "–", s.enabled?.text || "The engine has not reported its cache yet"),
        fact("Filled", s.fill.now !== null ? `${(s.fill.now * 100).toFixed(0)} %` : "–", "Share of the cache slots in use (needs the moe-layer lines)")),
      o.series.length > 1 ? h("div", { style: { marginTop: "14px" } },
        h("div", { class: "small muted" }, "Hit rate (%) over decode steps"), line(o.series, "recent_rate"),
        s.fill.series.length > 1 ? [h("div", { class: "small muted" }, "Cache fill (%)"), line(s.fill.series, "fill", { cls: "ln b" })] : null) : null));
    if (s.missing.length) {
      parts.push(h("div", { class: "callout warn" }, icon("alert", "sm"), h("div", { class: "stack tight" },
        h("b", null, "What this engine does not tell us yet"),
        s.missing.map((m) => h("div", { class: "small" }, m)),
        h("div", { class: "small muted" }, "The exact lines the engine would have to print are listed in docs/EXPERT-CACHE-DATA.md."))));
    }
    if (s.available.layers) parts.push(layerCard(s));
    clear(box, parts);
  }

  function fact(k, v, hint) {
    return h("div", { class: "fact", title: hint }, h("span", { class: "v" }, v), h("span", { class: "k" }, k));
  }

  function layerCard(s) {
    const sel = layer !== null && s.layers.some((l) => l.layer === layer) ? layer : null;
    const cells = s.layers.map((l) => h("button", { class: `layer-cell${l.layer === sel ? " sel" : ""}`, style: { background: heat(l.hit_rate) },
      title: `Layer ${l.layer}: hit rate ${l.hit_rate ?? "–"} %, ${l.filled ?? "?"} of ${l.slots ?? "?"} slots filled`,
      onclick: () => { layer = l.layer === layer ? null : l.layer; load(); } }, String(l.layer)));
    const detail = sel === null ? h("p", { class: "small muted", style: { margin: 0 } }, "Select a layer to see its hit rate over time and its most used experts.")
      : layerDetail(s, sel);
    return h("div", { class: "card pad-lg stack" },
      h("div", { class: "card-head" }, icon("layers"), h("div", null, h("h3", null, "Layers"),
        h("p", null, "Each square is one layer, green where most experts were found in the cache."))),
      h("div", { class: "layer-grid" }, cells), detail);
  }

  function layerDetail(s, n) {
    const l = s.layers.find((x) => x.layer === n);
    const hot = s.hot.layers[String(n)] || [];
    const top = hot.length ? hot[0][1] : 1;
    return h("div", { class: "stack" },
      h("dl", { class: "kv" },
        h("dt", null, "Hit rate"), h("dd", null, l.hit_rate !== null ? `${l.hit_rate} % (${fmt.int(l.hits)} hits, ${fmt.int(l.misses)} misses)` : "–"),
        h("dt", null, "Slots"), h("dd", null, l.slots ? `${l.filled ?? "?"} of ${l.slots} filled` : "–")),
      s.layer_series?.series.length > 1 ? [h("div", { class: "small muted" }, "Hit rate of this layer (%) over decode steps"),
        line(s.layer_series.series, "recent_rate")] : null,
      hot.length ? [h("div", { class: "small muted" }, "Most used experts (expert number, tokens routed to it)"),
        h("div", { class: "hot-grid" }, hot.map(([id, count]) => h("span", { class: "hot-cell", title: `Expert ${id}: ${fmt.int(count)}`,
          style: { background: `hsl(200, 70%, ${85 - Math.round((count / top) * 45)}%)` } }, String(id))))]
        : h("p", { class: "small muted", style: { margin: 0 } }, s.available.hot ? "No expert counts for this layer yet." : "The engine does not print the most used experts yet."));
  }

  load();
  return { unmount() { gone = true; clearTimeout(timer); } };
}
