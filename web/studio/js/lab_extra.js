// Experiments, second half: the A/B/A comparison of two settings and the quality measurements (KL divergence of a
// quantized file against a reference, and a small knowledge probe), so speed is never read without its price.

import { get, post } from "./api.js";
import { engineUp, store } from "./app.js";
import { clear, confirmDialog, fmt, h, icon, toast } from "./ui.js";
import { loadStyle, pctText, verdictPill, withCi } from "./tuner.js";

const completeModels = () => (store.state?.models || []).filter((m) => m.kind === "model" && m.complete && !m.error);
const profileName = (id) => (id === "auto" ? "Automatic settings" : (store.state?.profiles || []).find((p) => p.id === id)?.name || id);

export function mountExtras(abaBox, qualityBox, { onQuality } = {}) {
  loadStyle();
  let tuning = { current: null, results: [] };
  let quality = { current: null, results: [], builtin: {}, mkqa_license: "" };
  const form = { a: "auto", b: "", runs: 3, label: "", ref: "", cand: "", set: "builtin", lang: "en" };
  let timer = 0;
  let gone = false;

  async function load() {
    clearTimeout(timer);
    try {
      [tuning, quality] = await Promise.all([get("/api/tuning"), get("/api/quality")]);
    } catch (e) {
      if (!gone) toast(e.message, "bad");
      return;
    }
    renderAba();
    renderQuality();
    onQuality?.(quality);
    const busy = tuning.current?.state === "running" || quality.current?.state === "running";
    if (busy && !gone) timer = setTimeout(load, 1200);
  }

  // ------------------------------------------------------------------------------------------- A/B/A

  const abaRunning = () => tuning.current?.state === "running" && tuning.current.kind === "aba";

  async function runAba() {
    try {
      await post("/api/tuning/aba", { model: store.state?.settings?.last_model, a: form.a, b: form.b, runs: form.runs, label: form.label });
      load();
    } catch (e) {
      toast(e.message, "bad", 8000);
    }
  }

  function renderAba() {
    if (abaBox.contains(document.activeElement) && document.activeElement.tagName === "INPUT") return;
    const profiles = store.state?.profiles || [];
    const opts = [["auto", "Automatic settings"], ...profiles.map((p) => [p.id, p.name])];
    if (!form.b) form.b = (profiles.find((p) => p.id.startsWith("tuned-")) || profiles[0])?.id || "auto";
    const sel = (key) => h("select", { class: "select", disabled: abaRunning(), onchange: (e) => { form[key] = e.target.value; } },
      opts.map(([v, l]) => h("option", { value: v, selected: v === form[key] }, l)));
    const cur = tuning.current?.kind === "aba" ? tuning.current : null;
    const last = cur || tuning.results.find((r) => r.kind === "aba");
    const engineOn = engineUp() || !["stopped", "crashed"].includes(store.engine?.state);
    clear(abaBox,
      h("div", { class: "card-head" }, icon("layers"), h("div", { class: "grow" }, h("h3", null, "A/B/A comparison"),
        h("p", null, "Measures setting A, then B, then A again with the same number of runs. If the two A blocks differ, the machine "
          + "drifted (heat, background work) and a difference to B is not trusted."))),
      h("div", { class: "grid two" },
        h("label", { class: "field" }, h("span", null, "A (the reference)"), sel("a")),
        h("label", { class: "field" }, h("span", null, "B (the change)"), sel("b"))),
      h("div", { class: "row wrap", style: { marginTop: "12px" } },
        h("label", { class: "field", style: { width: "150px" } }, h("span", null, "Runs per block"),
          h("select", { class: "select", disabled: abaRunning(), onchange: (e) => { form.runs = Number(e.target.value); } },
            [3, 5, 8, 10].map((n) => h("option", { value: n, selected: n === form.runs }, `${n} runs`)))),
        h("div", { class: "field", style: { alignSelf: "end" } }, abaRunning()
          ? h("button", { class: "btn danger", onclick: () => post("/api/tuning/cancel").then(load) }, icon("stop", "sm"), "Cancel")
          : h("button", { class: "btn primary", onclick: runAba, disabled: engineOn || !store.state?.settings?.last_model || form.a === form.b,
            title: engineOn ? "Stop the engine first; the comparison starts it itself" : "" }, icon("play", "sm"), "Compare"))),
      engineOn && !abaRunning() ? h("p", { class: "small muted", style: { margin: "10px 0 0" } }, "Stop the engine first: the comparison starts and stops it by itself for each block.") : null,
      abaRunning() ? h("div", { class: "stack tight", style: { marginTop: "12px" } }, h("div", { class: "small" }, h("b", null, cur.step || "Starting")),
        h("div", { class: "bar indet" }, h("i"))) : null,
      last && last.kind === "aba" ? abaResult(last) : null);
  }

  function abaResult(run) {
    if (run.state === "failed") return h("div", { class: "callout bad", style: { marginTop: "12px" } }, icon("alert", "sm"), h("div", null, run.error));
    if (!run.result) return run.state === "cancelled" ? h("div", { class: "callout warn", style: { marginTop: "12px" } }, icon("alert", "sm"), "Cancelled.") : null;
    const row = (name, key, unit) => {
      const r = run.result[key];
      const ph = Object.fromEntries(run.trials.map((t) => [t.phase, t]));
      const ci = (p) => ph[p][key === "decode" ? "decode" : "prefill"];
      return h("tr", null, h("td", null, name),
        ["A1", "B", "A2"].map((p) => h("td", { class: "r" }, withCi(ci(p)))),
        h("td", null, verdictPill(r.verdict), " ", h("span", { class: "ci" }, pctText(r))),
        h("td", { class: "small muted" }, r.drift ? `The A blocks differ by ${r.drift_pct} %. ` : "", r.reason || ""));
    };
    return h("div", { class: "stack", style: { marginTop: "14px" } },
      h("div", { class: "small muted" }, `A = ${profileName(run.a.profile)}, B = ${profileName(run.b.profile)}, ${run.runs} runs per block, ${fmt.date(run.started)}`),
      h("div", { class: "table-wrap" }, h("table", { class: "t" },
        h("thead", null, h("tr", null, h("th"), h("th", { class: "r" }, "A, first"), h("th", { class: "r" }, "B"), h("th", { class: "r" }, "A, again"),
          h("th", null, "B against A"), h("th"))),
        h("tbody", null, row("Generation, tok/s", "decode"), row("Reading the prompt, tok/s", "prefill")))));
  }

  // ------------------------------------------------------------------------------------------- quality

  const qRunning = () => quality.current?.state === "running";

  async function runKld() {
    try {
      await post("/api/quality/kld", { reference: form.ref, candidate: form.cand });
      load();
    } catch (e) {
      toast(e.message, "bad", 9000);
    }
  }
  async function runProbe() {
    try {
      await post("/api/quality/probe", { source: form.set, label: form.label });
      load();
    } catch (e) {
      toast(e.message, "bad", 9000);
    }
  }
  async function importMkqa() {
    const ok = await confirmDialog({ title: "Download MKQA questions?",
      body: `They come from the MKQA dataset (Apple), licensed ${quality.mkqa_license} They are saved in your data folder only and are not part of FLE Studio.`, ok: "Download" });
    if (!ok) return;
    try {
      const r = await post("/api/quality/mkqa", { lang: form.lang, count: 48 });
      toast(`${r.count} questions saved`, "good");
      form.set = "mkqa";
      load();
    } catch (e) {
      toast(e.message, "bad", 9000);
    }
  }

  function renderQuality() {
    const models = completeModels();
    if (!form.ref) form.ref = models[0]?.id || "";
    if (!form.cand) form.cand = (models.find((m) => m.id !== form.ref) || models[0])?.id || "";
    const modelSel = (key) => h("select", { class: "select", disabled: qRunning(), onchange: (e) => { form[key] = e.target.value; } },
      models.map((m) => h("option", { value: m.id, selected: m.id === form[key] }, m.name)));
    const cur = quality.current;
    const results = quality.results.slice(0, 8);
    clear(qualityBox,
      h("div", { class: "card-head" }, icon("brain"), h("div", { class: "grow" }, h("h3", null, "Quality"),
        h("p", null, "Speed from a smaller file is only good news if the answers stay as good. KL divergence compares the next-word "
          + "probabilities of a file with a reference file (0 = the same); the knowledge probe asks the running model short factual questions."))),
      h("div", { class: "stack" },
        h("div", { class: "grid two" },
          h("label", { class: "field" }, h("span", null, "Reference (the larger file)"), modelSel("ref")),
          h("label", { class: "field" }, h("span", null, "File to measure"), modelSel("cand"))),
        h("div", { class: "row wrap" },
          qRunning() ? h("button", { class: "btn danger", onclick: () => post("/api/quality/cancel").then(load) }, icon("stop", "sm"), "Cancel")
            : [h("button", { class: "btn primary", onclick: runKld, disabled: models.length < 2 }, icon("play", "sm"), "Measure KL divergence"),
              h("button", { class: "btn", onclick: runProbe, disabled: !engineUp() || qRunning(), title: engineUp() ? "" : "Load a model first" },
                icon("brain", "sm"), "Ask the running model"),
              h("label", { class: "field", style: { width: "190px" } }, h("select", { class: "select", onchange: (e) => { form.set = e.target.value; } },
                h("option", { value: "builtin", selected: form.set === "builtin" }, `Our questions (${quality.builtin?.license || "CC0"})`),
                h("option", { value: "mkqa", selected: form.set === "mkqa" }, "MKQA (downloaded)"))),
              h("button", { class: "btn ghost", onclick: importMkqa }, icon("download", "sm"), "Get MKQA questions")]),
        qRunning() ? h("div", { class: "stack tight" }, h("div", { class: "row small" }, h("b", null, cur.step || "Starting"), h("span", { class: "grow" }),
          cur.kind === "kld" ? h("span", { class: "muted num" }, `${Math.round((cur.progress || 0) * 100)} %`) : null),
          h("div", { class: cur.kind === "kld" ? "bar" : "bar indet" }, h("i", { style: { width: `${Math.round((cur.progress || 0) * 100)}%` } }))) : null,
        cur && cur.state === "failed" ? h("div", { class: "callout bad" }, icon("alert", "sm"), h("div", null, cur.error)) : null,
        results.length ? h("div", { class: "table-wrap" }, h("table", { class: "t" },
          h("thead", null, h("tr", null, h("th", null, "When"), h("th", null, "What"), h("th", { class: "r" }, "Result"), h("th"))),
          h("tbody", null, results.map(resultRow)))) : h("p", { class: "small muted", style: { margin: 0 } }, "No quality results yet."),
        h("p", { class: "small faint", style: { margin: 0 } },
          `The knowledge probe is a smoke test: ${quality.builtin?.name || "a small set of questions"} (public domain). A handful of questions cannot rank two files; `
          + "the interval next to each score says how much it could move. The KL text is short and the reference run is cached, so a second file against the same reference is quick.")));
  }

  function resultRow(r) {
    const del = h("button", { class: "btn ghost icon sm", "aria-label": "Delete", onclick: async () => { await post(`/api/quality/${r.id}/delete`); load(); } }, icon("trash", "sm"));
    if (r.kind === "kld") {
      const m = r.result?.mean_kld;
      return h("tr", null, h("td", { class: "nowrap muted" }, fmt.date(r.started)),
        h("td", null, h("b", null, r.candidate.name), h("div", { class: "small faint" }, `KL divergence against ${r.reference.name}`)),
        h("td", { class: "r" }, r.state === "done" && m !== undefined ? [fmt.two(m).length < 5 ? m.toFixed(4) : fmt.two(m),
          r.result.mean_kld_unc ? h("span", { class: "ci" }, ` ± ${r.result.mean_kld_unc.toFixed(4)}`) : null] : h("span", { class: "pill warn" }, r.state)), h("td", null, del));
    }
    const pr = r.result;
    return h("tr", null, h("td", { class: "nowrap muted" }, fmt.date(r.started)),
      h("td", null, h("b", null, r.candidate.name), h("div", { class: "small faint" }, `Knowledge probe${r.label ? `, ${r.label}` : ""}`)),
      h("td", { class: "r" }, r.state === "done" && pr ? [`${pr.correct} of ${pr.n} (${fmt.pct(pr.accuracy)})`,
        h("span", { class: "ci" }, ` [${fmt.pct(pr.ci_lo)} to ${fmt.pct(pr.ci_hi)}]`)] : h("span", { class: "pill warn" }, r.state)), h("td", null, del));
  }

  load();
  return { reload: load, unmount() { gone = true; clearTimeout(timer); } };
}
