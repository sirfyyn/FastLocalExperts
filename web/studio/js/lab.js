// Experiments: a repeatable speed test of the loaded model, and every earlier result side by side, so a change in
// the start settings or the profile shows up as a number.

import { get, post } from "./api.js";
import { engineModelName, engineUp, navigate, on, store, topbar } from "./app.js";
import { mountExtras } from "./lab_extra.js";
import { loadStyle, withCi } from "./tuner.js";
import { clear, confirmDialog, copyButton, fmt, h, icon, toast } from "./ui.js";

const TEST_INFO = {
  "Short answer": "a few sentences, up to 256 tokens",
  Code: "a function with doctests, up to 768 tokens",
  "Long prompt": "about 6,000 tokens in, measures prefill",
};

export function mount(main) {
  const unsub = [];
  loadStyle();
  const abaBox = h("div", { class: "card pad-lg" });
  const qualityBox = h("div", { class: "card pad-lg" });
  let kldByModel = {};
  const runBox = h("div", { class: "card pad-lg" });
  const chartBox = h("div", { class: "card" });
  const tableBox = h("div", { class: "card", style: { padding: 0 } });
  main.replaceChildren(topbar("Experiments"), h("div", { class: "page" }, h("div", { class: "page-inner" },
    h("div", { class: "page-head" }, h("div", null, h("h2", null, "Experiments"),
      h("p", null, "Measure the loaded model with the same three requests every time, then change a setting or a profile and measure again. "
        + "Results stay on this computer with the exact command that produced them."))),
    runBox, chartBox, tableBox, abaBox, qualityBox)));

  let bench = { current: null, results: [], tests: [] };
  let open = null; // id of the result whose details are shown
  let timer = 0;
  let runs = 3;
  let label = "";
  let knowledge = false;

  async function load() {
    try {
      bench = await get("/api/bench");
    } catch (e) {
      toast(e.message, "bad");
      return;
    }
    renderRun();
    renderChart();
    renderTable();
    clearTimeout(timer);
    if (bench.current?.state === "running") timer = setTimeout(load, 1000);
  }

  // ------------------------------------------------------------------------------------------- run

  function renderRun() {
    if (runBox.contains(document.activeElement) && document.activeElement.tagName === "INPUT") return;
    const cur = bench.current;
    const running = cur?.state === "running";
    const up = engineUp();
    const profile = store.engine?.plan?.profile;
    const labelInput = h("input", { class: "input", value: label, placeholder: profile && profile !== "auto" ? profileName(profile) : "e.g. expert cache 16 GiB",
      oninput: (e) => { label = e.target.value; }, disabled: running });
    const runsSel = h("select", { class: "select", disabled: running, onchange: (e) => { runs = Number(e.target.value); } },
      [1, 3, 5, 10].map((n) => h("option", { value: n, selected: n === runs }, `${n} run${n > 1 ? "s" : ""}`)));
    const pct = cur?.total ? Math.round((cur.done / cur.total) * 100) : 0;

    clear(runBox,
      h("div", { class: "card-head" }, icon("lab"), h("div", { class: "grow" }, h("h3", null, "Speed test"),
        h("p", null, up ? `Measures ${engineModelName() || "the loaded model"} as it runs now.` : "Load a model first; the test talks to the running engine.")),
      !up ? h("button", { class: "btn", onclick: () => navigate("/engine") }, icon("engine", "sm"), "Engine") : null),
      h("div", { class: "grid three" }, Object.entries(TEST_INFO).map(([name, what]) =>
        h("div", { class: "fact" }, h("span", { class: "v", style: { fontSize: "13.5px" } }, name), h("span", { class: "k" }, what)))),
      h("div", { class: "row wrap", style: { marginTop: "14px" } },
        h("label", { class: "field", style: { flex: "1 1 260px" } }, h("span", null, "Label"), labelInput),
        h("label", { class: "field", style: { width: "140px" } }, h("span", null, "Repeat"), runsSel),
        h("label", { class: "field", style: { alignSelf: "end", flexDirection: "row", gap: "6px", alignItems: "center" } },
          h("input", { type: "checkbox", checked: knowledge, disabled: running, onchange: (e) => { knowledge = e.target.checked; } }),
          h("span", null, "Also ask the knowledge questions")),
        h("div", { class: "field", style: { alignSelf: "end" } }, running
          ? h("button", { class: "btn danger", onclick: cancel }, icon("stop", "sm"), "Cancel")
          : h("button", { class: "btn primary", onclick: start, disabled: !up }, icon("play", "sm"), "Run the test"))),
      running ? h("div", { class: "stack tight", style: { marginTop: "14px" } },
        h("div", { class: "row small" }, h("b", null, cur.step || "Starting"), h("span", { class: "grow" }), h("span", { class: "muted num" }, `${cur.done} of ${cur.total}`)),
        h("div", { class: "bar" }, h("i", { style: { width: `${pct}%` } })),
        cur.tests.length ? testsTable(cur.tests) : null) : null,
      !running && cur && cur.state !== "done" ? h("div", { class: `callout ${cur.state === "failed" ? "bad" : "warn"}`, style: { marginTop: "14px" } },
        icon("alert", "sm"), h("div", null, cur.state === "failed" ? `The last test failed: ${cur.error || "unknown error"}` : "The last test was cancelled.")) : null,
      h("p", { class: "small faint", style: { margin: "12px 0 0" } },
        "With sampling on, decode speed and MTP acceptance vary from run to run; the summary uses medians, so repeat at least three times before comparing."));
  }

  function profileName(id) {
    return (store.state?.profiles || []).find((p) => p.id === id)?.name || id;
  }

  async function start() {
    try {
      bench.current = await post("/api/bench/start", { runs, label: label.trim(), knowledge });
      renderRun();
      clearTimeout(timer);
      timer = setTimeout(load, 800);
    } catch (e) {
      toast(e.message, "bad", 7000);
    }
  }

  async function cancel() {
    try {
      await post("/api/bench/cancel");
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  // ------------------------------------------------------------------------------------------- comparison

  function title(r) {
    return r.label || (r.context?.profile && r.context.profile !== "auto" ? profileName(r.context.profile) : "") || r.context?.model || "Run";
  }

  function renderChart() {
    const done = bench.results.filter((r) => r.summary && (r.summary.decode || r.summary.prefill)).slice(0, 10);
    if (!done.length) { clear(chartBox, h("div", { class: "empty" }, "Your results show up here as bars, newest first.")); return; }
    const block = (key, unit, cls) => {
      const vals = done.map((r) => r.summary[key]).filter((v) => v !== null && v !== undefined);
      const top = Math.max(...vals, 1);
      return h("div", { class: "stack tight" },
        h("div", { class: "small muted" }, unit),
        h("div", { class: "bench-bars" }, done.map((r) => {
          const v = r.summary[key];
          return h("div", { class: "bench-bar", title: `${title(r)} · ${fmt.date(r.started)}` },
            h("span", { class: "lbl" }, title(r)),
            h("span", { class: "track" }, v ? h("i", { style: { width: `${(v / top) * 100}%`, background: cls } }) : null),
            h("span", { class: "val" }, v ? fmt.rate(v) : "–"));
        })));
    };
    clear(chartBox,
      h("div", { class: "card-head" }, icon("bolt"), h("div", null, h("h3", null, "Compare"),
        h("p", null, "Medians per test run; longer bars are faster."))),
      h("div", { class: "grid two" }, block("decode", "Generation, tokens per second", "var(--accent)"),
        block("prefill", "Reading the long prompt, tokens per second", "var(--cyan)")));
  }

  // ------------------------------------------------------------------------------------------- results

  function renderTable() {
    const rows = bench.results;
    if (!rows.length) { clear(tableBox, h("div", { class: "empty" }, "No results yet.")); return; }
    const body = [];
    for (const r of rows) {
      const s = r.summary || {};
      const model = r.context?.model || "";
      body.push(h("tr", { class: "click", onclick: () => { open = open === r.id ? null : r.id; renderTable(); } },
        h("td", { class: "nowrap muted" }, fmt.date(r.started)),
        h("td", null, h("b", null, title(r)), r.state !== "done" ? h("span", { class: "pill warn", style: { marginLeft: "6px" } }, r.state) : null,
          model && model !== title(r) ? h("div", { class: "small faint" }, model) : null),
        h("td", { class: "r" }, s.decode ? (s.decode_ci?.n > 1 ? withCi(s.decode_ci) : fmt.rate(s.decode)) : "–"),
        h("td", { class: "r" }, s.prefill ? (s.prefill_ci?.n > 1 ? withCi(s.prefill_ci) : fmt.int(s.prefill)) : "–"),
        h("td", { class: "r" }, s.ttft_ms ? `${fmt.int(s.ttft_ms)} ms` : "–"),
        h("td", { class: "r" }, s.accept !== null && s.accept !== undefined ? fmt.pct(s.accept) : "–"),
        h("td", { class: "r" }, qualityCell(r)),
        h("td", { class: "r" }, h("button", { class: "btn ghost icon sm", title: "Delete this result", "aria-label": "Delete",
          onclick: (e) => { e.stopPropagation(); remove(r); } }, icon("trash", "sm")))));
      if (open === r.id) body.push(h("tr", null, h("td", { colspan: 8, style: { background: "var(--soft)" } }, details(r))));
    }
    clear(tableBox, h("div", { class: "table-wrap", style: { border: 0 } }, h("table", { class: "t" },
      h("thead", null, h("tr", null, h("th", null, "When"), h("th", null, "Run"), h("th", { class: "r" }, "Decode tok/s"),
        h("th", { class: "r" }, "Prefill tok/s"), h("th", { class: "r" }, "First token"), h("th", { class: "r" }, "MTP accepted"), h("th", { class: "r" }, "Quality"), h("th"))),
      h("tbody", null, body))));
  }

  // Next to the speed: the knowledge score of the same run, and the newest KL divergence measured for this file.
  function qualityCell(r) {
    const q = r.quality;
    const kld = kldByModel[r.context?.model];
    const parts = [];
    if (q?.n) parts.push(h("div", { title: `Knowledge probe: ${q.correct} of ${q.n} correct, 95 % interval ${fmt.pct(q.ci_lo)} to ${fmt.pct(q.ci_hi)}` },
      `${fmt.pct(q.accuracy)} `, h("span", { class: "ci" }, `[${fmt.pct(q.ci_lo)} to ${fmt.pct(q.ci_hi)}]`)));
    if (kld) parts.push(h("div", { class: "small faint", title: `Mean KL divergence against ${kld.reference} (lower is better)` }, `KLD ${kld.mean_kld.toFixed(3)}`));
    return parts.length ? parts : "–";
  }

  function details(r) {
    const c = r.context || {};
    return h("div", { class: "stack", style: { padding: "6px 2px" } },
      h("dl", { class: "kv" },
        c.machine ? [h("dt", null, "Machine"), h("dd", null, c.machine)] : null,
        c.engine ? [h("dt", null, "Engine"), h("dd", null, `${c.engine}${c.build ? ` · build ${c.build}` : ""}`)] : null,
        c.profile ? [h("dt", null, "Start"), h("dd", null, c.profile === "auto" ? "Automatic settings" : profileName(c.profile))] : null,
        c.n_ctx ? [h("dt", null, "Context"), h("dd", null, `${fmt.int(c.n_ctx)} tokens`)] : null,
        [h("dt", null, "Runs"), h("dd", null, `${r.runs}, ${r.tests.length} requests`)]),
      testsTable(r.tests),
      c.command ? h("div", { class: "cmd" }, c.command, copyButton(c.command, "Copy the command")) : null);
  }

  function testsTable(tests) {
    return h("div", { class: "table-wrap" }, h("table", { class: "t" },
      h("thead", null, h("tr", null, h("th", null, "Request"), h("th", { class: "r" }, "Prompt"), h("th", { class: "r" }, "Prefill tok/s"),
        h("th", { class: "r" }, "Generated"), h("th", { class: "r" }, "Decode tok/s"), h("th", { class: "r" }, "First token"),
        h("th", { class: "r" }, "MTP"), h("th", { class: "r" }, "Wall"))),
      h("tbody", null, tests.map((t) => h("tr", null,
        h("td", null, `${t.name}`, h("span", { class: "faint small" }, ` · run ${t.run}`)),
        h("td", { class: "r" }, fmt.int(t.prompt_n)),
        h("td", { class: "r" }, t.prefill ? fmt.int(t.prefill) : "–"),
        h("td", { class: "r" }, fmt.int(t.generated)),
        h("td", { class: "r" }, t.decode ? fmt.rate(t.decode) : "–"),
        h("td", { class: "r" }, t.ttft_ms ? `${fmt.int(t.ttft_ms)} ms` : "–"),
        h("td", { class: "r" }, t.accept !== null && t.accept !== undefined ? fmt.pct(t.accept) : "–"),
        h("td", { class: "r" }, fmt.dur(t.wall_s)))))));
  }

  async function remove(r) {
    if (!(await confirmDialog({ title: "Delete this result?", body: `${title(r)} from ${fmt.date(r.started)}`, ok: "Delete", danger: true }))) return;
    try {
      await post(`/api/bench/${r.id}/delete`);
      load();
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  unsub.push(on("engine", () => { if (bench.current?.state !== "running") renderRun(); }));
  const extras = mountExtras(abaBox, qualityBox, { onQuality: (q) => { kldByModel = q.kld_by_model || {}; renderTable(); } });
  load();
  return { unmount() { clearTimeout(timer); extras.unmount(); unsub.forEach((u) => u()); } };
}
