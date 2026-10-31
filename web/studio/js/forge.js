// Forge: build your own quant. A guided flow (models, recipe, pruning, verification, build) around one recipe that
// can be saved, shared and built again. The work itself runs in jobs on the server; this page only collects the
// choices, shows the numbers the calculator returns, and follows a running job.

import { get, post } from "./api.js";
import { topbar } from "./app.js";
import { clear, confirmDialog, copyText, fmt, h, icon, toast } from "./ui.js";

const STEPS = [
  ["start", "Start"], ["models", "Models"], ["recipe", "Recipe"], ["prune", "Prune"], ["verify", "Verify"], ["build", "Build"],
];
const STORE_KEY = "fle-forge-draft";
const DONE_STATES = ["done", "failed", "cancelled", "interrupted"];
const LETTERS = "BKLMNPQRTUVWXYZ";

const blank = () => ({
  name: "my-quant",
  mode: "hybrid",
  sources: [{ id: "B", path: "", file: "", role: "base", info: null }, { id: "K", path: "", file: "", role: "donor", info: null }],
  hybrid: { budget_gb: 0, ram_gb: 0, vram_gb: 0, layers: {} },
  quantize: { type: "iq4_nl", imatrix: true, allow_requantize: false, rules: [] },
  imatrix: { corpus: "", chunks: 200, context: 2048 },
  reap: { enabled: false, keep: 0, score: "reap", table: "" },
  verify: { enabled: true, text: "", chunks: 20, context: 2048, reference_path: "", reference_file: "" },
});

function loadDraft() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) return { ...blank(), ...JSON.parse(raw) };
  } catch { /* storage blocked or damaged: start blank */ }
  return blank();
}

let draft = loadDraft();
function saveDraft() {
  try {
    const copy = JSON.parse(JSON.stringify(draft));
    for (const s of copy.sources) s.info = null;
    localStorage.setItem(STORE_KEY, JSON.stringify(copy));
  } catch { /* not worth a message */ }
}

// ----------------------------------------------------------------------------------------------- recipe <-> draft

function baseId() {
  return draft.sources.find((s) => s.role === "base")?.id || "";
}

function toRecipe() {
  const d = draft;
  const sources = {};
  for (const s of d.sources) {
    if (s.path.trim() || s.file) sources[s.id] = { path: s.path.trim(), file: s.file || "", role: s.role, size: s.info?.size || 0 };
  }
  let reference = "";
  if (d.verify.enabled) {
    if (d.verify.reference_path.trim() || d.verify.reference_file) {
      sources.R = { path: d.verify.reference_path.trim(), file: d.verify.reference_file || "", role: "reference", size: 0 };
      reference = "R";
    } else {
      reference = d.mode === "quantize" ? "S" : baseId();
    }
  }
  const r = {
    name: d.name.trim() || "my-quant", mode: d.mode, sources,
    imatrix: { corpus: d.imatrix.corpus.trim(), chunks: Number(d.imatrix.chunks) || 0, context: Number(d.imatrix.context) || 2048 },
    reap: { enabled: d.reap.enabled, keep: Number(d.reap.keep) || 0, score: d.reap.score, table: d.reap.table.trim(), chunks: Number(d.imatrix.chunks) || 0 },
    verify: { enabled: d.verify.enabled, reference, text: d.verify.text.trim(), chunks: Number(d.verify.chunks) || 0, context: Number(d.verify.context) || 2048 },
  };
  if (d.mode === "hybrid") {
    const base = baseId();
    const layers = {};
    for (const [layer, sid] of Object.entries(d.hybrid.layers)) if (sid !== base) layers[layer] = sid;
    r.hybrid = { base, layers, budget_gb: Number(d.hybrid.budget_gb) || 0 };
  } else {
    r.quantize = { source: "S", type: d.quantize.type, imatrix: d.quantize.imatrix, allow_requantize: d.quantize.allow_requantize,
      rules: d.quantize.rules.filter((x) => x.pattern.trim()).map((x) => ({ pattern: x.pattern.trim(), type: x.type })) };
  }
  return r;
}

function fromRecipe(r, needs) {
  const d = blank();
  d.name = r.name;
  d.mode = r.mode;
  d.sources = Object.entries(r.sources).filter(([, s]) => s.role !== "reference").map(([id, s]) =>
    ({ id, path: s.path || "", file: s.file || "", role: s.role, info: null }));
  const ref = r.sources[r.verify.reference];
  if (ref && ref.role === "reference") { d.verify.reference_path = ref.path || ""; d.verify.reference_file = ref.file || ""; }
  if (r.mode === "hybrid") {
    d.hybrid = { budget_gb: r.hybrid.budget_gb, ram_gb: 0, vram_gb: 0, layers: { ...r.hybrid.layers } };
    const base = r.hybrid.base;
    for (const s of d.sources) s.role = s.id === base ? "base" : "donor";
  } else {
    d.quantize = { type: r.quantize.type, imatrix: r.quantize.imatrix, allow_requantize: r.quantize.allow_requantize,
      rules: r.quantize.rules.map((x) => ({ ...x })) };
    for (const s of d.sources) s.role = "base";
  }
  d.imatrix = { corpus: r.imatrix.corpus, chunks: r.imatrix.chunks, context: r.imatrix.context };
  d.reap = { enabled: r.reap.enabled, keep: r.reap.keep, score: r.reap.score, table: r.reap.table };
  d.verify = { ...d.verify, enabled: r.verify.enabled, text: r.verify.text, chunks: r.verify.chunks, context: r.verify.context };
  d.needs = needs;
  return d;
}

function setMode(mode) {
  if (draft.mode === mode) return;
  draft.mode = mode;
  if (mode === "quantize") draft.sources = [{ id: "S", path: draft.sources[0]?.path || "", file: "", role: "base", info: null }];
  else draft.sources = [{ id: "B", path: draft.sources[0]?.path || "", file: "", role: "base", info: null },
    { id: "K", path: "", file: "", role: "donor", info: null }];
  draft.calc = null;
  draft.hybrid.layers = {};
  saveDraft();
}

// ----------------------------------------------------------------------------------------------- small widgets

function field(label, input, hint, style) {
  return h("label", { class: "field", style }, h("span", null, label), input, hint ? h("span", { class: "hint" }, hint) : null);
}

function textInput(obj, key, { placeholder = "", mono = false, onchange = null, id = null } = {}) {
  return h("input", { class: `input${mono ? " mono" : ""}`, value: obj[key] ?? "", placeholder, id, spellcheck: "false",
    oninput: (e) => { obj[key] = e.target.value; saveDraft(); if (onchange) onchange(); } });
}

function numInput(obj, key, { min = 0, step = 1, onchange = null, id = null } = {}) {
  return h("input", { class: "input", type: "number", min, step, value: obj[key] ?? 0, id,
    oninput: (e) => { obj[key] = e.target.value === "" ? 0 : Number(e.target.value); saveDraft(); if (onchange) onchange(); } });
}

function select(obj, key, options, onchange) {
  return h("select", { class: "select", onchange: (e) => { obj[key] = e.target.value; saveDraft(); if (onchange) onchange(); } },
    options.map((o) => { const [v, l] = Array.isArray(o) ? o : [o, o]; return h("option", { value: v, selected: v === obj[key] }, l); }));
}

function toggle(obj, key, label, onchange) {
  return h("label", { class: "check" }, h("span", { class: "switch" },
    h("input", { type: "checkbox", checked: !!obj[key], onchange: (e) => { obj[key] = e.target.checked; saveDraft(); if (onchange) onchange(); } }),
    h("i")), h("span", null, label));
}

async function busy(btn, fn) {
  if (btn) btn.disabled = true;
  try {
    return await fn();
  } catch (e) {
    toast(e.message, "bad", 8000);
    return null;
  } finally {
    if (btn) btn.disabled = false;
  }
}

const gb = (bytes) => (bytes === null || bytes === undefined ? "–" : bytes < 1e8 ? fmt.bytes(bytes) : `${fmt.one(bytes / 1e9)} GB`);

function callout(kind, text, ...extra) {
  return h("div", { class: `callout ${kind}` }, icon(kind === "bad" ? "alert" : kind === "good" ? "ok" : kind === "warn" ? "alert" : "info", "sm"),
    h("div", null, text, ...extra));
}

function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const a = h("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ----------------------------------------------------------------------------------------------- page

export function mount(main) {
  let step = "start";
  let overview = null;
  let job = null; // the job being followed (full record), or null
  let logSeq = 0;
  let timer = 0;
  let gone = false;

  const panel = h("div", { class: "fg-panel" });
  const stepper = h("nav", { class: "fg-steps", "aria-label": "Forge steps" });
  const jobBox = h("div", { class: "stack" });
  const earlier = h("div", { class: "card" });
  const logView = h("div", { class: "log", role: "log", "aria-live": "off", id: "fg-log" });

  main.replaceChildren(topbar("Forge"), h("div", { class: "page" }, h("div", { class: "page-inner" },
    h("div", { class: "page-head" }, h("div", null, h("h2", null, "Forge"),
      h("p", null, "Build your own quant with the FLE recipe: take whole expert layers from the quants you already have, prune experts, or quantize "
        + "a full-precision model. The result is checked against the original, and the recipe can be shared so anyone can build it again."))),
    stepper, panel, jobBox, earlier)));

  // ------------------------------------------------------------------------------------------- stepper

  function renderStepper() {
    clear(stepper, STEPS.map(([id, label], i) => h("button", {
      class: `fg-step${id === step ? " on" : ""}`, "data-step": id, onclick: () => go(id), "aria-current": id === step ? "step" : null,
    }, h("span", { class: "n" }, i + 1), h("span", null, label))));
  }

  function go(id) {
    step = id;
    renderStepper();
    renderPanel();
  }

  const nav = (prev, next) => h("div", { class: "row between", style: { marginTop: "6px" } },
    prev ? h("button", { class: "btn", onclick: () => go(prev) }, "Back") : h("span"),
    next ? h("button", { class: "btn primary", onclick: () => go(next) }, "Next") : h("span"));

  function renderPanel() {
    const fn = { start: panelStart, models: panelModels, recipe: panelRecipe, prune: panelPrune, verify: panelVerify, build: panelBuild }[step];
    clear(panel, fn());
  }

  // ------------------------------------------------------------------------------------------- start

  function toolsTable() {
    const rows = (overview?.tools || []).map((t) => h("tr", null,
      h("td", null, h("b", null, t.name)),
      h("td", null, h("span", { class: `pill ${t.ok ? "good" : "warn"}` }, h("span", { class: "dot" }), t.ok ? "Found" : (t.problem || "Missing"))),
      h("td", { class: "faint" }, t.version || ""),
      h("td", { class: "faint", style: { overflowWrap: "anywhere" } }, t.ok ? `${t.path} (${t.source})` : "")));
    return h("div", { class: "table-wrap" }, h("table", { class: "t" },
      h("thead", null, h("tr", null, h("th", null, "Tool"), h("th", null, "Status"), h("th", null, "Version"), h("th", null, "Where"))),
      h("tbody", null, rows.length ? rows : h("tr", null, h("td", { colspan: 4, class: "faint" }, "Looking for the tools...")))));
  }

  function panelStart() {
    const dirInput = h("input", { class: "input mono", id: "fg-tools-dir", value: overview?.tools_dir || "", spellcheck: "false",
      placeholder: "Folder with llama-quantize, llama-imatrix, llama-perplexity" });
    const scan = h("button", { class: "btn", onclick: (e) => busy(e.currentTarget, async () => {
      overview = await post("/api/forge/tools", { tools_dir: dirInput.value.trim() });
      renderPanel();
    }) }, icon("refresh", "sm"), "Search again");
    const missing = (overview?.tools || []).filter((t) => !t.ok && t.name !== "llama-fle-reap");
    return h("div", { class: "stack" },
      h("div", { class: "card pad-lg stack" },
        h("div", { class: "card-head" }, icon("layers"), h("div", { class: "grow" }, h("h3", null, "What do you want to make?"),
          h("p", null, "Pick the way you start. You can load a recipe from someone else on the last step instead."))),
        h("div", { class: "grid two" },
          modeCard("hybrid", "Hybrid from quants you have", "Whole expert layers come from the quants you already own, chosen to fit a size. Nothing is requantized, so it is fast and loses nothing."),
          modeCard("quantize", "Quantize a full-precision model", "Run llama-quantize on a BF16 or F16 model, with an importance matrix and per-tensor rules.")),
        h("div", { class: "grid two" }, field("Name of the result", textInput(draft, "name", { id: "fg-name" }), "Used for the file name and the profile."))),
      h("div", { class: "card stack" },
        h("div", { class: "card-head" }, icon("terminal"), h("div", { class: "grow" }, h("h3", null, "llama.cpp tools"),
          h("p", null, "Forge drives the command-line tools of your llama.cpp build. They are searched in the folder below, in the engines installed through Studio, and on your PATH. A pure hybrid needs none of them."))),
        toolsTable(),
        h("div", { class: "row wrap" }, h("div", { class: "grow" }, dirInput), scan),
        missing.length && draft.mode === "quantize" ? callout("warn", `Missing: ${missing.map((t) => t.name).join(", ")}. Install an engine on the Engine page or enter the folder above.`) : null),
      nav(null, "models"));
  }

  function modeCard(mode, title, text) {
    return h("button", { class: `fg-mode${draft.mode === mode ? " on" : ""}`, "data-mode": mode, onclick: () => { setMode(mode); renderPanel(); } },
      h("b", null, title), h("span", null, text));
  }

  // ------------------------------------------------------------------------------------------- models

  async function inspect(src, btn) {
    await busy(btn, async () => {
      src.info = await post("/api/forge/inspect", { path: src.path });
      src.file = src.path.replace(/\\/g, "/").split("/").pop();
      saveDraft();
      renderPanel();
    });
  }

  function infoLine(info) {
    if (!info) return null;
    return h("div", { class: "fg-info" },
      h("span", null, `${info.arch}`), h("span", null, `${info.layers} layers`),
      info.experts ? h("span", null, `${info.experts} experts, ${info.experts_used} used`) : h("span", { class: "bad-text" }, "no experts"),
      info.bpw ? h("span", null, `${fmt.two(info.bpw)} bits per weight in the experts`) : null,
      h("span", null, gb(info.size)), info.parts > 1 ? h("span", null, `${info.parts} files`) : null,
      ...info.types.map((t) => h("span", { class: "tag" }, `${t.types}: ${t.layers.length} layers`)));
  }

  function sourceRow(src, i) {
    const hyb = draft.mode === "hybrid";
    const roleName = src.role === "base" ? "Base" : "Donor";
    const read = h("button", { class: "btn", onclick: (e) => inspect(src, e.currentTarget), disabled: !src.path.trim() }, "Read it");
    return h("div", { class: "fg-source", "data-source": src.id },
      h("div", { class: "row wrap" },
        h("span", { class: `fg-id fg-c${i}` }, src.id),
        hyb ? h("span", { class: "small muted", style: { width: "44px" } }, roleName) : h("span", { class: "small muted" }, "Source"),
        h("div", { class: "grow", style: { minWidth: "240px" } }, textInput(src, "path", { mono: true, placeholder: src.file ? `a copy of ${src.file}` : "Path of a .gguf file (the first part of a split model)",
          onchange: () => { src.info = null; draft.calc = null; read.disabled = !src.path.trim(); } })),
        read,
        hyb && src.role !== "base" ? h("button", { class: "btn ghost icon", "aria-label": `Remove ${src.id}`, title: "Remove",
          onclick: () => { draft.sources = draft.sources.filter((s) => s !== src); draft.calc = null; saveDraft(); renderPanel(); } }, icon("x", "sm")) : null),
      src.file && !src.path.trim() ? h("div", { class: "small warn-text" }, `The recipe was made with ${src.file}. Point this at your copy.`) : null,
      infoLine(src.info));
  }

  function panelModels() {
    const hyb = draft.mode === "hybrid";
    const add = hyb ? h("button", { class: "btn", onclick: () => {
      const used = new Set(draft.sources.map((s) => s.id));
      const id = [...LETTERS].find((c) => !used.has(c));
      if (!id) return;
      draft.sources.push({ id, path: "", file: "", role: "donor", info: null });
      renderPanel();
    } }, icon("plus", "sm"), "Add a donor quant") : null;
    return h("div", { class: "stack" },
      h("div", { class: "card stack" },
        h("div", { class: "card-head" }, icon("models"), h("div", { class: "grow" }, h("h3", null, hyb ? "The quants to combine" : "The model to quantize"),
          h("p", null, hyb
            ? "The base gives the metadata and every tensor that is not an expert. Each donor must be another quant of the same model: same layers, same shapes. Different quant types are what makes the mix useful."
            : "Use a BF16, F16 or F32 file. Quantizing from an already quantized file is possible, but loses more."))),
        draft.sources.map((s, i) => sourceRow(s, i)), add,
        draft.needs?.length ? callout("info", "This recipe was made with the files below. You do not need the same download, only the same model in the same quant.",
          h("ul", { class: "fg-list" }, draft.needs.map((n) => h("li", null, `${n.id}: ${n.file}${n.size ? ` (${gb(n.size)})` : ""}`)))) : null),
      nav("start", "recipe"));
  }

  // ------------------------------------------------------------------------------------------- recipe

  function sourcePaths() {
    const out = {};
    for (const s of draft.sources) if (s.path.trim()) out[s.id] = s.path.trim();
    return out;
  }

  async function calculate(btn, withLayers) {
    const h2 = draft.hybrid;
    const body = { mode: "hybrid", base: baseId(), sources: sourcePaths(), budget_gb: Number(h2.budget_gb) || 0,
      ram_gb: Number(h2.ram_gb) || 0, vram_gb: Number(h2.vram_gb) || 0, reap_keep: draft.reap.enabled ? Number(draft.reap.keep) || 0 : 0 };
    if (withLayers) body.layers = h2.layers;
    const calc = await busy(btn, () => post("/api/forge/calculate", body));
    if (!calc) return;
    draft.calc = calc;
    h2.layers = { ...calc.layers };
    saveDraft();
    renderPanel();
  }

  function layerGrid(calc) {
    const ids = draft.sources.map((s) => s.id);
    const entries = Object.entries(calc.layers).sort((a, b) => Number(a[0]) - Number(b[0]));
    return h("div", { class: "fg-grid", role: "group", "aria-label": "Source of each expert layer" }, entries.map(([layer, sid]) => {
      const idx = Math.max(0, ids.indexOf(sid));
      return h("button", { class: `fg-cell fg-c${idx}`, title: `Layer ${layer} from ${sid}. Click to use the next source.`, "data-layer": layer,
        onclick: (e) => {
          draft.hybrid.layers[layer] = ids[(ids.indexOf(draft.hybrid.layers[layer]) + 1) % ids.length];
          calculate(e.currentTarget, true);
        } }, h("small", null, layer), h("b", null, sid));
    }));
  }

  function previewBlock(calc) {
    const p = calc.preview;
    const sel = calc.selection;
    const verdicts = {
      all_gpu: ["good", "Fits entirely on the GPU."],
      experts_cpu: ["good", "Fits with the experts in RAM and the rest on the GPU, the usual way to run a large mixture-of-experts model."],
      cpu_only: ["warn", "Does not fit the GPU as it is; it would run from RAM."],
      too_big: ["bad", "Does not fit the memory you entered."],
    };
    const [kind, text] = verdicts[p.verdict] || ["info", p.verdict];
    const counts = sel?.counts || {};
    return h("div", { class: "stack" },
      h("div", { class: "grid four" },
        fact("Result", gb(calc.file_bytes)), fact("Experts", gb(p.experts)),
        fact("Everything else", gb(p.dense + p.inputs)), fact("Context memory", gb(p.kv))),
      sel && !sel.feasible ? callout("bad", sel.note || "The budget is below the smallest possible result.") : null,
      sel?.note && sel.feasible ? callout("info", sel.note) : null,
      h("div", { class: "row wrap small muted" }, Object.entries(counts).map(([sid, n]) => h("span", { class: "tag" }, `${sid}: ${n} layers`)),
        calc.pruned_bytes ? h("span", { class: "tag" }, `after pruning: ${gb(calc.pruned_bytes)}`) : null),
      (Number(draft.hybrid.ram_gb) || Number(draft.hybrid.vram_gb)) ? callout(kind, text) : null);
  }

  const fact = (k, v) => h("div", { class: "fact" }, h("span", { class: "v" }, v), h("span", { class: "k" }, k));

  function panelRecipe() {
    if (draft.mode === "quantize") return panelQuantize();
    const h2 = draft.hybrid;
    const ready = draft.sources.filter((s) => s.path.trim()).length >= 2 && baseId();
    return h("div", { class: "stack" },
      h("div", { class: "card stack" },
        h("div", { class: "card-head" }, icon("sliders"), h("div", { class: "grow" }, h("h3", null, "Choose the layers"),
          h("p", null, "Give a size budget and the calculator picks, for every expert layer, the donor that keeps the most within it. Click a layer to use another source."))),
        h("div", { class: "grid four" },
          field("Size budget (GB)", numInput(h2, "budget_gb", { step: 0.5, id: "fg-budget" })),
          field("RAM (GB)", numInput(h2, "ram_gb", { step: 1, id: "fg-ram" }), "To judge the fit."),
          field("GPU memory (GB)", numInput(h2, "vram_gb", { step: 1, id: "fg-vram" }), "0 for no GPU."),
          h("div", { class: "field", style: { alignSelf: "end" } },
            h("button", { class: "btn primary", id: "fg-calc", disabled: !ready, onclick: (e) => calculate(e.currentTarget, false) }, icon("bolt", "sm"), "Calculate"))),
        !ready ? h("div", { class: "small muted" }, "Enter at least two files on the Models step first.") : null,
        draft.calc ? [layerGrid(draft.calc), previewBlock(draft.calc)] : null,
        h("p", { class: "small faint", style: { margin: 0 } },
          "Without measured errors the calculator assumes that more bits per weight always means less error. That orders the quants correctly but does not know which layers are sensitive; the check at the end tells you how well the mix did.")),
      nav("models", "prune"));
  }

  function panelQuantize() {
    const q = draft.quantize;
    const types = overview?.types || ["q4_0", "q8_0", "iq4_nl", "iq3_s"];
    const estimateBox = h("div", { id: "fg-estimate" });
    const showEstimate = (est) => clear(estimateBox, est ? h("div", { class: "grid three" },
      fact("About", gb(est.estimate.estimate_bytes)), fact("Bits per weight", fmt.two(est.estimate.bpw)),
      fact("Tensors kept in their type", String(est.estimate.tensors_kept))) : null);
    const rules = q.rules.map((rule, i) => h("div", { class: "row wrap" },
      h("div", { class: "grow" }, textInput(rule, "pattern", { mono: true, placeholder: "Part of a tensor name, e.g. ffn_down_exps" })),
      select(rule, "type", types), h("button", { class: "btn ghost icon", "aria-label": "Remove rule", onclick: () => { q.rules.splice(i, 1); renderPanel(); } }, icon("x", "sm"))));
    return h("div", { class: "stack" },
      h("div", { class: "card stack" },
        h("div", { class: "card-head" }, icon("sliders"), h("div", { class: "grow" }, h("h3", null, "Quantization"),
          h("p", null, "The type is used for every weight matrix unless a rule below names it. Vectors such as norms keep their precision."))),
        h("div", { class: "grid three" }, field("Default type", select(q, "type", overview?.ftypes || types), "Names like q4_k_m are mixtures that llama-quantize composes per tensor."),
          h("div", { class: "field" }, h("span", null, "Importance matrix"), toggle(q, "imatrix", "Compute one first (recommended)")),
          h("div", { class: "field" }, h("span", null, "Source is already quantized"), toggle(q, "allow_requantize", "Allow requantizing"))),
        h("div", { class: "label" }, "Rules for single tensors"), rules,
        h("div", { class: "row wrap" }, h("button", { class: "btn", onclick: () => { q.rules.push({ pattern: "", type: q.type }); renderPanel(); } }, icon("plus", "sm"), "Add a rule"),
          h("button", { class: "btn", id: "fg-estimate-btn", disabled: !draft.sources[0].path.trim(), onclick: (e) => busy(e.currentTarget, async () => {
            showEstimate(await post("/api/forge/calculate", { mode: "quantize", source_path: draft.sources[0].path.trim(),
              quantize: { type: q.type, rules: q.rules.filter((x) => x.pattern.trim()) } }));
          }) }, "Estimate the size")),
        estimateBox,
        h("p", { class: "small faint", style: { margin: 0 } }, "The size is an estimate from the tensor shapes; llama-quantize decides some details itself.")),
      nav("models", "prune"));
  }

  // ------------------------------------------------------------------------------------------- prune

  function panelPrune() {
    const r = draft.reap;
    const info = draft.sources.find((s) => s.info?.experts)?.info;
    const fle = (overview?.tools || []).find((t) => t.name === "llama-fle-reap");
    const experts = info?.experts || 0;
    return h("div", { class: "stack" },
      h("div", { class: "card stack" },
        h("div", { class: "card-head" }, icon("cpu"), h("div", { class: "grow" }, h("h3", null, "Prune experts (REAP), optional"),
          h("p", null, "Removes the experts that the router uses least, in every layer, and shrinks the router to match. The file gets smaller by about the share removed, and quality drops by an amount you can measure on the next step."))),
        toggle(r, "enabled", "Prune experts after the build", () => renderPanel()),
        r.enabled ? [
          h("div", { class: "grid three" },
            field(`Experts to keep per layer${experts ? ` (of ${experts})` : ""}`, numInput(r, "keep", { min: 1, id: "fg-keep" }),
              info ? `At least ${info.experts_used}, the number used per token.` : "Read a model on the Models step to see the count."),
            field("How to rank experts", select(r, "score", overview?.scores || ["reap", "freq", "mass", "man", "energy"]),
              "reap weighs how often and how strongly an expert is used; freq only counts."),
            field("Router statistics file (optional)", textInput(r, "table", { mono: true, placeholder: "tsv from llama-fle-reap" }))),
          fle?.ok ? callout("good", "llama-fle-reap was found and will measure the router on your calibration text.")
            : callout("warn", "llama-fle-reap was not found. Forge will derive routing counts from an importance matrix instead, a weaker signal. FLE engines ship the real tool."),
          callout("warn", "Pruning cannot be undone, and models with grouped expert routing are refused. Prune the best quality you have, not a quant that was already mixed down.")] : null),
      nav("recipe", "verify"));
  }

  // ------------------------------------------------------------------------------------------- calibration + verify

  function corpusCard() {
    const out = h("div", { id: "fg-corpus-report", class: "small muted" });
    const inputs = h("textarea", { class: "textarea mono", rows: 3, id: "fg-corpus-inputs", spellcheck: "false",
      placeholder: "One file or folder per line. Text, markdown, code and json are used." });
    const used = draft.mode === "quantize" && draft.quantize.imatrix || draft.reap.enabled;
    const report = (made) => {
      draft.imatrix.corpus = made.path;
      saveDraft();
      renderPanel();
      toast(`Calibration text written: ${fmt.int(made.chars)} characters`, "good");
    };
    return h("div", { class: "card stack" },
      h("div", { class: "card-head" }, icon("folder"), h("div", { class: "grow" }, h("h3", null, "Calibration text"),
        h("p", null, used ? "The importance matrix and the router statistics are measured on this text. A mixture-of-experts model needs a lot of varied text; a short one never reaches many of the experts."
          : "Only needed when you compute an importance matrix or prune."))),
      h("div", { class: "row wrap" },
        h("div", { class: "grow", style: { minWidth: "260px" } }, field("Text file", textInput(draft.imatrix, "corpus", { mono: true, id: "fg-corpus", placeholder: "Path of a text file" }))),
        h("button", { class: "btn", id: "fg-corpus-bundled", style: { alignSelf: "end" }, onclick: (e) => busy(e.currentTarget, async () => report(await post("/api/forge/corpus", { bundled: true }))) },
          "Use the text that comes with Studio")),
      h("p", { class: "small faint", style: { margin: 0 } }, "The bundled text is about 1 MB of English and German prose, code, dialogue and numbers, all public domain, CC0 or MIT (origin and license of every file are in forge/corpus/SOURCES.md). A good start; text from your own use fits your use better."),
      h("div", { class: "grid three" }, field("Chunks", numInput(draft.imatrix, "chunks"), "0 reads the whole text."), field("Context", numInput(draft.imatrix, "context", { min: 64 }))),
      h("details", { class: "fg-details" }, h("summary", null, "Mix a text from your own files"),
        h("div", { class: "stack", style: { marginTop: "8px" } }, inputs,
          h("div", { class: "row wrap" },
            h("button", { class: "btn", id: "fg-corpus-make", onclick: (e) => busy(e.currentTarget, async () => {
              const list = inputs.value.split("\n").map((x) => x.trim()).filter(Boolean);
              report(await post("/api/forge/corpus", { inputs: list, chars: 2000000 }));
            }) }, "Mix"),
            h("button", { class: "btn", onclick: (e) => busy(e.currentTarget, async () => report(await post("/api/forge/corpus", { stdlib: true, chars: 2000000 }))) },
              "Use the Python library as a first trial"),
            out),
          h("p", { class: "small faint", style: { margin: 0 } }, "Forge ships no text. Check the license of whatever you mix in before you share the result's recipe or numbers."))));
  }

  function referenceTable() {
    const rows = (overview?.references || []).slice(0, 8);
    if (!rows.length) return null;
    return h("details", { class: "fg-details" }, h("summary", null, "Published FLE figures for orientation"),
      h("div", { class: "table-wrap", style: { marginTop: "8px" } }, h("table", { class: "t" },
        h("thead", null, h("tr", null, h("th", null, "Model"), h("th", { class: "r" }, "KLD (E / W / K)"), h("th", { class: "r" }, "Size"))),
        h("tbody", null, rows.map((c) => h("tr", null, h("td", null, c.name), h("td", { class: "r" }, c.kld.map((x) => x.toFixed(4)).join(" / ")),
          h("td", { class: "r" }, gb(c.size))))))),
      h("p", { class: "small faint" }, "Measured on the FLE project's own texts, not yours. Your numbers are only comparable to each other."));
  }

  function panelVerify() {
    const v = draft.verify;
    return h("div", { class: "stack" },
      corpusCard(),
      h("div", { class: "card stack" },
        h("div", { class: "card-head" }, icon("ok"), h("div", { class: "grow" }, h("h3", null, "Check the result"),
          h("p", null, "The result and the original model read the same text. The KL divergence says how far the result's probabilities are from the original's; top-1 says how often both pick the same next token."))),
        toggle(v, "enabled", "Compare the result with the original", () => renderPanel()),
        v.enabled ? [
          field(draft.mode === "quantize" ? "Reference model (default: the source)" : "Reference model, full precision",
            textInput(v, "reference_path", { mono: true, id: "fg-ref", placeholder: draft.mode === "quantize" ? "Leave empty to use the source model" : "Path of the BF16 or F16 .gguf" }),
            "Without a full-precision file, use the best quant you have; the numbers then say how far the result is from that."),
          h("div", { class: "row wrap" },
            h("div", { class: "grow", style: { minWidth: "260px" } }, field("Text for the comparison", textInput(v, "text", { mono: true, id: "fg-eval", placeholder: "Path of a text file, different from the calibration text" }))),
            h("button", { class: "btn", id: "fg-eval-bundled", style: { alignSelf: "end" }, onclick: (e) => busy(e.currentTarget, async () => {
              const made = await post("/api/forge/corpus", { bundled: true, part: "evaluation" });
              v.text = made.path;
              saveDraft();
              renderPanel();
            }) }, "Use the held-out part of the bundled text")),
          h("div", { class: "grid three" }, field("Chunks", numInput(v, "chunks", { min: 1 }), "The stored probabilities grow with chunks times vocabulary."),
            field("Context", numInput(v, "context", { min: 64 })))] : null,
        v.enabled ? referenceTable() : null),
      nav("prune", "build"));
  }

  // ------------------------------------------------------------------------------------------- build

  async function check(btn, box) {
    const info = await busy(btn, () => post("/api/forge/check", { recipe: toRecipe() }));
    if (!info) return null;
    clear(box,
      h("ol", { class: "fg-plan" }, info.steps.map((s) => h("li", null, s.title))),
      info.problems.map((p) => callout("bad", p)),
      info.missing_tools.map((t) => callout("bad", `${t} was not found. Set the tools folder on the Start step.`)),
      info.ok ? callout("good", "Everything is in place.") : null);
    return info;
  }

  function panelBuild() {
    const r = toRecipe();
    const box = h("div", { class: "stack", id: "fg-check-box" });
    const importer = h("input", { type: "file", accept: ".json,application/json", style: { display: "none" }, id: "fg-import", onchange: async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        const data = JSON.parse(await file.text());
        const res = await post("/api/forge/recipe/import", { recipe: data });
        draft = fromRecipe(res.recipe, res.needs);
        saveDraft();
        toast(`Loaded recipe ${res.recipe.name}. Point it at your files on the Models step.`, "good", 7000);
        go("models");
      } catch (err) {
        toast(err.message || "Not a recipe file", "bad", 8000);
      }
    } });
    const share = async () => (await post("/api/forge/recipe/share", { recipe: toRecipe() })).recipe;
    const start = h("button", { class: "btn primary", id: "fg-start", onclick: (e) => busy(e.currentTarget, async () => {
      const info = await check(null, box);
      if (!info?.ok) return;
      job = await post("/api/forge/jobs", { recipe: toRecipe() });
      logSeq = 0;
      follow();
    }) }, icon("play", "sm"), "Build it");
    return h("div", { class: "stack" },
      h("div", { class: "card stack" },
        h("div", { class: "card-head" }, icon("check"), h("div", { class: "grow" }, h("h3", null, "Build"),
          h("p", null, "Check the plan, then start. The job keeps running if you leave this page, and it can be resumed if it stops."))),
        h("dl", { class: "kv" },
          h("dt", null, "Name"), h("dd", null, r.name), h("dt", null, "Way"), h("dd", null, r.mode === "hybrid" ? "Hybrid of existing quants" : "Quantize a full-precision model"),
          h("dt", null, "Sources"), h("dd", null, Object.entries(r.sources).map(([id, s]) => `${id}: ${s.file || s.path.split(/[\\/]/).pop() || "?"}`).join(", ") || "none yet"),
          r.mode === "hybrid" ? [h("dt", null, "Donor layers"), h("dd", null, `${Object.keys(r.hybrid.layers).length}`)] : [h("dt", null, "Type"), h("dd", null, r.quantize.type)],
          h("dt", null, "Pruning"), h("dd", null, r.reap.enabled ? `keep ${r.reap.keep} experts per layer (${r.reap.score})` : "off"),
          h("dt", null, "Check"), h("dd", null, r.verify.enabled ? `${r.verify.chunks} chunks of ${r.verify.context}` : "off"),
          h("dt", null, "Output"), h("dd", { class: "mono" }, overview?.output_dir || "")),
        box,
        h("div", { class: "row wrap" }, h("button", { class: "btn", id: "fg-check", onclick: (e) => check(e.currentTarget, box) }, "Check the plan"), start)),
      h("div", { class: "card stack" },
        h("div", { class: "card-head" }, icon("link"), h("div", { class: "grow" }, h("h3", null, "Recipe"),
          h("p", null, "A recipe is a small JSON file: sources by file name (no folders), the layer choice, the pruning and the check. Share it and others can build the same model from their own copies."))),
        h("div", { class: "row wrap" },
          h("button", { class: "btn", id: "fg-save", onclick: (e) => busy(e.currentTarget, async () => download(`${r.name}.recipe.json`, JSON.stringify(await share(), null, 2) + "\n")) },
            icon("download", "sm"), "Save recipe"),
          h("button", { class: "btn", onclick: (e) => busy(e.currentTarget, async () => { await copyText(JSON.stringify(await share(), null, 2)); toast("Recipe copied", "good"); }) },
            icon("copy", "sm"), "Copy"),
          h("button", { class: "btn", onclick: () => importer.click() }, icon("folder", "sm"), "Load a recipe"), importer)),
      nav("verify", null));
  }

  // ------------------------------------------------------------------------------------------- job

  function stepRow(s) {
    const mark = { done: "ok", running: "play", failed: "alert", cancelled: "pause", interrupted: "pause" }[s.state];
    const pct = s.state === "done" ? 100 : Math.round((s.progress || 0) * 100);
    return h("div", { class: `fg-job-step ${s.state}`, "data-step-id": s.id },
      h("span", { class: "ic" }, mark ? icon(mark, "sm") : h("span", { class: "dotp" })),
      h("div", { class: "grow" }, h("div", { class: "row" }, h("b", null, s.title), h("span", { class: "grow" }), h("span", { class: "small muted num" }, s.state === "running" ? `${pct} %` : s.state)),
        s.state === "running" || s.state === "done" ? h("div", { class: "bar" }, h("i", { style: { width: `${pct}%` } })) : null,
        s.message && s.state === "running" ? h("div", { class: "small muted" }, s.message) : null,
        s.error ? h("div", { class: "small bad-text" }, s.error) : null));
  }

  function resultCard(j) {
    const res = j.result || {};
    const card = res.card;
    if (j.state !== "done" || !res.model) return null;
    const facts = [fact("Size", gb(res.size))];
    if (card) {
      if (card.bpw) facts.push(fact("Bits per weight", fmt.two(card.bpw)));
      if (card.kld !== null && card.kld !== undefined) facts.push(fact("KL divergence", card.kld.toFixed(4) + (card.kld_se ? ` ± ${card.kld_se.toFixed(4)}` : "")));
      if (card.top1 !== null && card.top1 !== undefined) facts.push(fact("Same top token", fmt.pct(card.top1, 1)));
      if (card.ppl_ratio) facts.push(fact("Perplexity ratio", card.ppl_ratio.toFixed(4)));
    }
    return h("div", { class: "card stack", id: "fg-result" },
      h("div", { class: "card-head" }, icon("ok"), h("div", { class: "grow" }, h("h3", null, "Done"), h("p", { class: "mono", style: { overflowWrap: "anywhere" } }, res.model))),
      h("div", { class: "grid four" }, facts),
      card ? h("p", { class: "small faint", style: { margin: 0 } }, `Measured against ${card.reference || "the reference"} on ${card.text || "your text"}, ${card.chunks} chunks of ${card.context}. ${card.note}`) : null,
      h("dl", { class: "kv" }, h("dt", null, "SHA-256"), h("dd", { class: "mono" }, res.sha256 || "")),
      h("div", { class: "row wrap" },
        h("button", { class: "btn primary", id: "fg-publish", onclick: (e) => busy(e.currentTarget, async () => {
          const pub = await post(`/api/forge/jobs/${j.id}/publish`, {});
          toast(`Added to your models, with the start profile “${pub.profile.name}”`, "good", 7000);
        }) }, icon("models", "sm"), "Add to my models"),
        h("button", { class: "btn", onclick: (e) => busy(e.currentTarget, async () => {
          download(`${j.name}.recipe.json`, JSON.stringify(await get(`/api/forge/jobs/${j.id}/recipe`), null, 2) + "\n");
        }) }, icon("download", "sm"), "Save recipe"),
        h("button", { class: "btn", onclick: () => copyText(res.sha256 || "") }, icon("copy", "sm"), "Copy SHA-256")));
  }

  function renderJob() {
    if (!job) { clear(jobBox); return; }
    const j = job;
    const running = j.running || j.state === "running" || j.state === "queued";
    const tone = { done: "good", failed: "bad", cancelled: "warn", interrupted: "warn" }[j.state] || "info";
    clear(jobBox,
      h("div", { class: "card stack", id: "fg-job" },
        h("div", { class: "card-head" }, icon("layers"), h("div", { class: "grow" }, h("h3", null, j.name), h("p", null, `Job ${j.id}`)),
          h("span", { class: `pill ${tone}${running ? " live" : ""}`, id: "fg-job-state" }, h("span", { class: "dot" }), j.state)),
        j.error ? callout("bad", j.error) : null,
        h("div", { class: "stack tight" }, j.steps.map(stepRow)),
        h("div", { class: "row wrap" },
          running ? h("button", { class: "btn danger", id: "fg-cancel", onclick: (e) => busy(e.currentTarget, () => post(`/api/forge/jobs/${j.id}/cancel`, {})) }, icon("stop", "sm"), "Cancel") : null,
          !running && j.state !== "done" ? h("button", { class: "btn primary", id: "fg-resume", onclick: (e) => busy(e.currentTarget, async () => { job = await post(`/api/forge/jobs/${j.id}/resume`, {}); follow(); }) },
            icon("play", "sm"), "Resume") : null,
          !running ? h("button", { class: "btn ghost", onclick: () => removeJob(j) }, icon("trash", "sm"), "Remove from the list") : null,
          h("span", { class: "grow" }), h("button", { class: "btn ghost sm", onclick: () => { job = null; clearTimeout(timer); renderJob(); } }, "Hide")),
        logView),
      resultCard(j));
  }

  async function removeJob(j) {
    const files = j.state === "done";
    const ok = await confirmDialog({ title: "Remove this job?", body: files ? "The model file stays where it is; only the job and its log are removed." : "Its log and unfinished work are removed.", ok: "Remove", danger: true });
    if (!ok) return;
    await busy(null, () => post(`/api/forge/jobs/${j.id}/delete`, {}));
    job = null;
    clearTimeout(timer);
    renderJob();
    loadOverview();
  }

  function addLog(lines) {
    const stick = logView.scrollTop + logView.clientHeight >= logView.scrollHeight - 30;
    for (const l of lines) logView.append(h("div", { class: `ln ${l.kind || ""}` }, l.text));
    while (logView.childElementCount > 1500) logView.firstChild.remove();
    if (stick) logView.scrollTop = logView.scrollHeight;
  }

  async function follow() {
    clearTimeout(timer);
    logView.replaceChildren();
    logSeq = 0;
    renderJob();
    const loop = async () => {
      if (gone || !job) return;
      try {
        const next = await get(`/api/forge/jobs/${job.id}`);
        const log = await get(`/api/forge/jobs/${job.id}/log?since=${logSeq}`);
        const finished = DONE_STATES.includes(next.state) && !next.running;
        const changed = JSON.stringify(next.steps) !== JSON.stringify(job.steps) || next.state !== job.state;
        job = next;
        logSeq = log.next;
        if (changed || finished) renderJob();
        addLog(log.lines);
        if (finished) { loadOverview(); return; }
      } catch (e) {
        if (e.status === 404) { job = null; renderJob(); return; }
      }
      timer = setTimeout(loop, 700);
    };
    await loop();
  }

  // ------------------------------------------------------------------------------------------- earlier jobs

  function renderEarlier() {
    const jobs = overview?.jobs || [];
    if (!jobs.length) { clear(earlier, h("div", { class: "empty" }, "Your builds show up here.")); return; }
    clear(earlier, h("div", { class: "card-head" }, icon("refresh"), h("div", { class: "grow" }, h("h3", null, "Earlier builds"))),
      h("div", { class: "table-wrap" }, h("table", { class: "t" }, h("tbody", null, jobs.map((j) => h("tr", { class: "click", "data-job": j.id,
        onclick: async () => { job = await get(`/api/forge/jobs/${j.id}`); follow(); main.querySelector(".page")?.scrollTo(0, 0); } },
      h("td", null, h("b", null, j.name)), h("td", { class: "faint" }, fmt.date(j.created)),
      h("td", { class: "r" }, h("span", { class: `pill ${{ done: "good", failed: "bad", running: "info" }[j.state] || "warn"}` }, h("span", { class: "dot" }), j.state))))))));
  }

  async function loadOverview() {
    try {
      overview = await get("/api/forge");
    } catch (e) {
      toast(e.message, "bad");
      return;
    }
    if (gone) return;
    renderEarlier();
    if (step === "start" || step === "prune" || step === "verify") renderPanel();
  }

  // The text that comes with Studio is the default for both texts, until the person names another.
  async function bundledDefaults() {
    try {
      if (!draft.imatrix.corpus) {
        const made = await post("/api/forge/corpus", { bundled: true });
        if (!draft.imatrix.corpus) draft.imatrix.corpus = made.path; // the person may have typed one meanwhile
      }
      if (!draft.verify.text) {
        const made = await post("/api/forge/corpus", { bundled: true, part: "evaluation" });
        if (!draft.verify.text) draft.verify.text = made.path;
      }
      saveDraft();
      if (!gone && (step === "verify" || step === "build")) renderPanel();
    } catch { /* the page works without the defaults */ }
  }

  renderStepper();
  renderPanel();
  bundledDefaults();
  loadOverview().then(() => {
    const active = (overview?.jobs || []).find((j) => j.running || j.state === "running");
    if (active && !job) get(`/api/forge/jobs/${active.id}`).then((j) => { job = j; follow(); }).catch(() => {});
  });

  return { unmount() { gone = true; clearTimeout(timer); } };
}
