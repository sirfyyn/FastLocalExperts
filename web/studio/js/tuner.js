// "Best for my machine": the auto-tuner card of the Engine page, and the small helpers the other pages share for
// showing a measurement with its interval and a comparison verdict.

import { get, post } from "./api.js";
import { clear, fmt, h, icon, toast } from "./ui.js";

let styled = false;
export function loadStyle() {
  if (styled) return;
  styled = true;
  document.head.append(h("link", { rel: "stylesheet", href: "/tuning.css" }));
}

// "84.2 tok/s" with "[82.9 to 85.1]" when the run count allows an interval (3 or more).
export function ciText(s, digits = 1) {
  if (!s || s.median === null || s.median === undefined) return "–";
  const v = (x) => (digits === 0 ? fmt.int(x) : fmt.rate(x));
  return v(s.median);
}
export function ciRange(s) {
  if (!s || s.lo === null || s.lo === undefined) return "";
  return `${fmt.rate(s.lo)} to ${fmt.rate(s.hi)}`;
}
export function withCi(s) {
  const r = ciRange(s);
  return h("span", { class: "num" }, ciText(s), r ? h("span", { class: "ci" }, ` [${r}]`) : null,
    s && s.n ? h("span", { class: "ci" }, ` n=${s.n}`) : null);
}

const VERDICT = {
  better: ["better", "faster"],
  worse: ["worse", "slower"],
  "within noise": ["noise", "within noise"],
  "not enough runs": ["few", "not enough runs"],
};
export function verdictPill(verdict, better = "faster", worse = "slower") {
  const [cls, label] = VERDICT[verdict] || ["few", verdict || ""];
  return h("span", { class: `verdict ${cls}` }, cls === "better" ? better : cls === "worse" ? worse : label);
}
export function pctText(c) {
  if (!c || c.pct === null || c.pct === undefined) return "";
  const sign = c.pct > 0 ? "+" : "";
  const range = c.pct_lo !== null && c.pct_lo !== undefined ? ` (${c.pct_lo > 0 ? "+" : ""}${c.pct_lo.toFixed(1)} to ${c.pct_hi > 0 ? "+" : ""}${c.pct_hi.toFixed(1)} %)` : "";
  return `${sign}${c.pct.toFixed(1)} %${range}`;
}

const BUDGETS = [[10, "10 minutes"], [20, "20 minutes"], [45, "45 minutes"], [90, "90 minutes"], [240, "4 hours"]];

// opts: {model: () => id, profile: () => id, engineRunning: () => bool, onUse: (profileId) => Promise}
export function mountBest(box, opts) {
  loadStyle();
  let data = { current: null, results: [], best: null, params: [], defaults: { params: [], runs: 3, budget_min: 20 } };
  const form = { budget: 20, runs: 3, params: null, screen: true };
  let timer = 0;
  let gone = false;

  async function load() {
    clearTimeout(timer);
    try {
      data = await get(`/api/tuning?model=${encodeURIComponent(opts.model() || "")}`);
    } catch (e) {
      if (!gone) toast(e.message, "bad");
      return;
    }
    if (!form.params) form.params = new Set(data.defaults.params);
    render();
    if (data.current?.state === "running" && !gone) timer = setTimeout(load, 1200);
  }

  const running = () => data.current?.state === "running" && data.current.kind === "tune";

  async function tune() {
    try {
      await post("/api/tuning/start", { model: opts.model(), profile: "auto", budget_min: form.budget, runs: form.runs,
        params: [...form.params], screen: form.screen });
      load();
    } catch (e) {
      toast(e.message, "bad", 8000);
    }
  }

  async function use() {
    try {
      await opts.onUse(data.best.id);
    } catch (e) {
      toast(e.message, "bad", 8000);
    }
  }

  function render() {
    const cur = data.current?.kind === "tune" ? data.current : null;
    const latest = cur || data.results.find((r) => r.kind === "tune" && r.model?.id === opts.model());
    const head = h("div", { class: "card-head" }, icon("bolt"), h("div", { class: "grow" }, h("h3", null, "Best for my machine"),
      h("p", null, "Tries a few settings one after another, measures each several times, and keeps what is clearly faster. "
        + "The result is saved as a launch profile that starts the same way every time.")));
    if (running()) { clear(box, head, progress(cur)); return; }

    const best = data.best;
    const blocks = [];
    if (best) {
      const t = best.meta?.tuned || {};
      blocks.push(h("div", { class: "tune-best" },
        h("div", { class: "small muted" }, `Tuned for this machine on ${t.date || "an earlier day"}`),
        h("b", null, t.decode ? `${fmt.rate(t.decode.median)} tokens per second` : best.name),
        t.base_decode?.median && t.decode ? h("div", { class: "small" },
          `${((t.decode.median / t.base_decode.median - 1) * 100).toFixed(1)} % against the automatic settings, `
          + `median of ${t.decode.n} runs, ${t.trials} settings tried.`) : null));
    } else {
      blocks.push(h("p", { class: "small muted", style: { margin: 0 } },
        "Nothing is tuned for this model on this machine yet. The first run needs the engine stopped; it starts and stops "
        + "the engine by itself, once for every setting it tries."));
    }
    if (latest && !cur?.error && latest.state === "failed") blocks.push(h("div", { class: "callout bad" }, icon("alert", "sm"), h("div", null, latest.error)));
    if (latest && latest.state === "cancelled") blocks.push(h("div", { class: "callout warn" }, icon("alert", "sm"), h("div", null, "The last tuning run was cancelled; the settings from before stay as they are.")));
    if (latest?.state === "done" || latest?.state === "failed") blocks.push(summary(latest));

    const engineOn = opts.engineRunning();
    const canTune = !!opts.model() && !engineOn;
    const form1 = h("details", { class: "stack" },
      h("summary", { class: "small muted", style: { cursor: "pointer" } }, "What to try and for how long"),
      h("div", { class: "stack", style: { marginTop: "10px" } },
        h("div", { class: "grid two" },
          h("label", { class: "field" }, h("span", null, "Time budget"), h("select", { class: "select", onchange: (e) => { form.budget = Number(e.target.value); } },
            BUDGETS.map(([m, l]) => h("option", { value: m, selected: m === form.budget }, l)))),
          h("label", { class: "field" }, h("span", null, "Runs per setting"), h("select", { class: "select", onchange: (e) => { form.runs = Number(e.target.value); } },
            [2, 3, 5, 8].map((n) => h("option", { value: n, selected: n === form.runs }, `${n} runs`))))),
        h("div", { class: "checks" }, data.params.map((p) => h("label", null,
          h("input", { type: "checkbox", checked: form.params.has(p.key), onchange: (e) => { e.target.checked ? form.params.add(p.key) : form.params.delete(p.key); } }), p.label))),
        h("label", { class: "checks" }, h("span", null, h("input", { type: "checkbox", checked: form.screen, onchange: (e) => { form.screen = e.target.checked; } }),
          " Screen llama.cpp settings quickly with llama-bench first (stock llama.cpp only)")),
        h("p", { class: "small faint", style: { margin: 0 } },
          "A setting only replaces the current one when it is faster and the 95 % interval of the difference stays above zero. "
          + "A setting that does not load (out of memory) is skipped. The tuner never starts a trial it cannot finish within the budget.")));

    const buttons = h("div", { class: "row wrap" },
      best ? h("button", { class: "btn primary", onclick: use, disabled: !opts.model() }, icon("play", "sm"), "Start with the tuned settings") : null,
      h("button", { class: best ? "btn" : "btn primary", onclick: tune, disabled: !canTune,
        title: engineOn ? "Stop the engine first" : "" }, icon("bolt", "sm"), best ? "Tune again" : "Find the best settings"),
      engineOn ? h("span", { class: "small muted" }, "Stop the engine to tune.") : null);
    clear(box, head, h("div", { class: "stack" }, blocks, form1, buttons));
  }

  function progress(cur) {
    const used = Date.now() / 1000 - cur.started;
    const pct = Math.min(100, Math.round((used / cur.budget_s) * 100));
    return h("div", { class: "stack" },
      h("div", { class: "row small" }, h("b", null, cur.step || "Starting"), h("span", { class: "grow" }),
        h("span", { class: "muted num" }, `${fmt.dur(used)} of at most ${fmt.dur(cur.budget_s)}`)),
      h("div", { class: "bar" }, h("i", { style: { width: `${pct}%` } })),
      cur.best ? h("div", { class: "small" }, "Best so far: ", h("b", null, cur.best.label), " at ", withCi(cur.best.decode), " tokens per second") : null,
      cur.trials.length ? trialTable(cur) : null,
      h("div", null, h("button", { class: "btn danger", onclick: () => post("/api/tuning/cancel").then(load) }, icon("stop", "sm"), "Cancel")));
  }

  function trialTable(run) {
    const rows = [];
    if (run.baseline) rows.push(h("tr", null, h("td", null, "Starting settings"), h("td", { class: "r" }, withCi(run.baseline.decode)),
      h("td", { class: "r" }, withCi(run.baseline.prefill)), h("td"), h("td")));
    for (const t of run.trials) {
      rows.push(h("tr", null, h("td", null, h("span", { class: "faint small" }, `${t.param_label}: `), t.label),
        t.state === "ok" ? [h("td", { class: "r" }, withCi(t.decode)), h("td", { class: "r" }, withCi(t.prefill)),
          h("td", null, verdictPill(t.vs_best.verdict), " ", h("span", { class: "ci" }, pctText(t.vs_best))),
          h("td", null, t.kept ? h("span", { class: "pill good" }, "kept") : null)]
          : h("td", { colspan: 4, class: "small", style: { color: "var(--bad)" } }, t.error)));
    }
    return h("div", { class: "table-wrap" }, h("table", { class: "t" },
      h("thead", null, h("tr", null, h("th", null, "Setting"), h("th", { class: "r" }, "Decode tok/s"), h("th", { class: "r" }, "Prefill tok/s"),
        h("th", null, "Against the best so far"), h("th"))),
      h("tbody", null, rows)));
  }

  function summary(run) {
    const bits = [];
    if (run.trials?.length || run.baseline) bits.push(h("details", null,
      h("summary", { class: "small muted", style: { cursor: "pointer" } }, `Last run: ${run.trials.length} settings tried in ${fmt.dur(run.elapsed || 0)}`),
      h("div", { style: { marginTop: "8px" } }, trialTable(run))));
    for (const n of run.notes || []) bits.push(h("p", { class: "small muted", style: { margin: 0 } }, n));
    if (run.best?.command) bits.push(h("div", { class: "cmd" }, run.best.command));
    return h("div", { class: "stack tight" }, bits);
  }

  load();
  return { reload: load, unmount() { gone = true; clearTimeout(timer); } };
}
