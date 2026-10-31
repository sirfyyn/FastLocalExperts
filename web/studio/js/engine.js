// Engine: start and stop llama-server, the start settings, launch profiles (complete commands), the installed
// engines, connecting to a server that already runs, and the live log.

import { get, post } from "./api.js";
import { emit, on, refreshState, store, topbar } from "./app.js";
import { explainBlock, flagTitle, loadFlagTable } from "./flags.js";
import { loadStyle, mountBest } from "./tuner.js";
import { clear, confirmDialog, copyButton, copyText, debounce, fmt, h, icon, stateBadge, toast } from "./ui.js";

const PLACEHOLDERS = [
  ["{engine}", "The llama-server of the engine picked for this profile"],
  ["{model}", "The model file chosen when starting"],
  ["{mtp_head}", "The MTP head file (Models page)"],
  ["{mtp_vocab}", "The MTP vocabulary file"],
  ["{expert_profile}", "Where the FLE engine keeps the expert-use profile of this model"],
  ["{context}", "Context length from the start settings"],
  ["{threads}", "Threads from the start settings (0 there = one per physical core)"],
  ["{port}", "A free port FLE Studio picks"],
  ["{host}", "127.0.0.1"],
  ["{data}", "FLE Studio's data folder"],
];
const CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
const NEW_COMMAND = "{engine} -m {model} -c {context} -t {threads} --jinja";

let savedMode = "simple";
try { savedMode = localStorage.getItem("fle-engine-mode") === "advanced" ? "advanced" : "simple"; } catch { /* storage blocked */ }

export function mount(main) {
  const unsub = [];
  loadStyle();
  loadFlagTable().then(() => { if (preview) renderPreview(); });
  let mode = savedMode; // simple: model, Best for my machine, Start; advanced: every setting and profile
  const modeSwitch = h("div", { class: "mode-switch", role: "group", "aria-label": "Settings view" });
  function renderMode() {
    clear(modeSwitch, [["simple", "Simple"], ["advanced", "Advanced"]].map(([m, label]) => h("button", {
      class: mode === m ? "on" : "", "aria-pressed": String(mode === m), "data-mode": m, onclick: () => setMode(m) }, label)));
  }
  function setMode(m) {
    mode = savedMode = m;
    try { localStorage.setItem("fle-engine-mode", m); } catch { /* storage blocked */ }
    renderMode();
    applyMode();
    renderLaunch(true);
    if (preview) renderPreview();
  }
  function applyMode() {
    const adv = mode === "advanced";
    for (const k of ["profiles", "engines", "attach"]) boxes[k].hidden = !adv;
  }
  const boxes = {
    status: h("div", { class: "card pad-lg" }),
    best: h("div", { class: "card" }),
    launch: h("div", { class: "card" }),
    preview: h("div", { class: "card" }),
    profiles: h("div", { class: "card" }),
    engines: h("div", { class: "card" }),
    attach: h("div", { class: "card" }),
    log: h("div", { class: "card" }),
  };
  main.replaceChildren(topbar("Engine"), h("div", { class: "page" }, h("div", { class: "page-inner" },
    h("div", { class: "page-head" }, h("div", null, h("h2", null, "Engine"),
      h("p", null, "FLE Studio runs llama-server for you. Every start is a plain command that you can read, copy and change; "
        + "a profile keeps a command of your own, placeholders and all.")), modeSwitch),
    boxes.status,
    boxes.best,
    h("div", { class: "grid two start" }, boxes.launch, boxes.preview),
    boxes.profiles,
    h("div", { class: "grid two start" }, boxes.engines, boxes.attach),
    boxes.log)));

  const settings = () => store.state?.settings || {};
  const sel = { model: "", profile: "auto" };
  let editing = null; // the profile in the editor: {id, name, engine, command, notes, builtin}
  let preview = null;
  let previewError = "";
  let checkResult = null;
  let pending = {};

  // ------------------------------------------------------------------------------------------- data flow

  function syncSelection() {
    const st = settings();
    const models = usableModels();
    if (!sel.model || !models.some((m) => m.id === sel.model)) {
      sel.model = models.some((m) => m.id === st.last_model) ? st.last_model : models[0]?.id || "";
    }
    const profiles = store.state?.profiles || [];
    const want = st.launch?.profile || "auto";
    sel.profile = want === "auto" || profiles.some((p) => p.id === want) ? want : "auto";
  }

  function usableModels() {
    return (store.state?.models || []).filter((m) => m.kind === "model" && m.complete && !m.error);
  }

  const flush = debounce(async () => {
    const patch = pending;
    pending = {};
    try {
      store.state.settings = await post("/api/settings", patch);
      refreshPreview();
    } catch (e) {
      toast(e.message, "bad", 7000);
    }
  }, 250);

  function save(patch) {
    for (const [k, v] of Object.entries(patch)) {
      pending[k] = v && typeof v === "object" && !Array.isArray(v) ? { ...(pending[k] || {}), ...v } : v;
      const target = store.state.settings;
      if (v && typeof v === "object" && !Array.isArray(v)) Object.assign(target[k], v); else target[k] = v;
    }
    flush();
  }

  const refreshPreview = debounce(async () => {
    if (!sel.model && sel.profile === "auto") {
      preview = null;
      previewError = usableModels().length ? "Choose a model." : "Download or add a model first (Models page).";
      renderPreview();
      return;
    }
    try {
      preview = await post("/api/engine/preview", { model: sel.model, profile: sel.profile });
      previewError = "";
    } catch (e) {
      preview = null;
      previewError = e.message;
    }
    renderPreview();
  }, 200);

  // ------------------------------------------------------------------------------------------- status

  function renderStatus() {
    const e = store.engine;
    if (!e) return;
    const svc = e.service;
    const name = e.model?.name || "";
    const running = ["starting", "ready"].includes(e.state);
    const actions = [];
    if (svc) {
      actions.push(...serviceButtons(e));
    } else if (e.state === "attached") {
      actions.push(h("button", { class: "btn", onclick: stop }, icon("x", "sm"), "Disconnect"));
    } else if (running) {
      actions.push(h("button", { class: "btn", onclick: restart, disabled: e.state !== "ready" }, icon("refresh", "sm"), "Restart"),
        h("button", { class: "btn danger", onclick: stop }, icon("power", "sm"), e.state === "starting" ? "Cancel" : "Stop"));
    } else if (e.state === "stopping") {
      actions.push(h("button", { class: "btn", disabled: true }, "Stopping…"));
    } else {
      actions.push(h("button", { class: "btn primary", onclick: start, disabled: !sel.model && sel.profile === "auto" },
        icon("play", "sm"), "Start"));
    }

    let detail = null;
    if (e.state === "starting") {
      const pct = e.load?.progress ? Math.round(e.load.progress * 100) : null;
      detail = h("div", { class: "stack tight" },
        h("div", { class: pct === null ? "bar indet" : "bar" }, h("i", { style: { width: `${pct || 0}%` } })),
        h("div", { class: "row small muted num wrap" }, h("span", null, e.phase || "Loading"),
          e.load?.total ? h("span", null, `${fmt.bytes(e.load.read)} of ${fmt.bytes(e.load.total)}`) : null,
          e.load?.elapsed ? h("span", null, fmt.dur(e.load.elapsed)) : null));
    } else if (e.state === "crashed") {
      detail = h("div", { class: "callout bad" }, icon("alert", "sm"), h("div", { class: "stack tight" },
        h("b", null, e.error || "The engine stopped"),
        e.hint ? h("span", null, e.hint) : null,
        e.exit_code !== null && e.exit_code !== undefined ? h("span", { class: "small muted" }, `Exit code ${e.exit_code}. The log below has the details.`) : null));
    } else if (e.state === "ready" || e.state === "attached") {
      const plan = e.plan || {};
      detail = h("dl", { class: "kv" },
        kv("Context", e.n_ctx ? `${fmt.int(e.n_ctx)} tokens${e.slots > 1 ? ` shared by ${e.slots} slots` : ""}` : "–"),
        e.attached ? kv("Server", e.base_url + (e.reachable ? "" : " (not answering)")) : kv("Engine", `${plan.engine || "–"} · ${plan.binary || ""}`),
        plan.profile ? kv("Start", plan.profile === "auto" ? "Automatic settings" : profileName(plan.profile)) : null,
        e.build ? kv("Build", e.build) : null,
        e.rss ? kv("Engine memory", fmt.mem(e.rss)) : null,
        e.ready_at && !e.attached ? kv("Up", `${fmt.dur(Date.now() / 1000 - e.ready_at)} · loaded in ${fmt.dur(e.ready_at - e.started)}`) : null);
    } else if (e.state === "stopped" && svc) {
      detail = e.error ? h("div", { class: "callout warn" }, icon("alert", "sm"), h("span", null, e.error))
        : h("div", { class: "muted small" }, `${svc.unit} is not running. Start runs the service's own command; FLE Studio does not change it.`);
    } else if (e.state === "stopped") {
      detail = h("div", { class: "muted small" }, "Not running. Choose a model and how to start it below, then press Start.");
    }
    const unitInfo = svc ? h("dl", { class: "kv" },
      kv("Service", `${svc.unit}${svc.user ? " (user service)" : ""}${svc.description ? ` · ${svc.description}` : ""}`),
      kv("systemd", serviceLine(svc)),
      svc.pid ? kv("Main process", `PID ${svc.pid}${svc.runs_as ? ` · runs as ${svc.runs_as}` : ""}`) : null,
      svc.command ? kv("Command", h("div", { class: "row", style: { alignItems: "flex-start", gap: "6px" } },
        h("code", { class: "mono grow", style: { minWidth: 0 } }, svc.command), copyButton(svc.command, "Copy the command"))) : null)
      : null;

    const endpoint = e.state === "ready" || e.state === "attached"
      ? h("div", { class: "callout info" }, icon("link", "sm"), h("div", { class: "stack tight", style: { minWidth: 0 } },
        h("div", { class: "row wrap" }, h("b", null, "OpenAI-compatible API"), h("code", { class: "mono" }, `${location.origin}/v1`),
          copyButton(`${location.origin}/v1`, "Copy the API address")),
        h("span", { class: "small muted" }, store.state?.lan
          ? "Apps on this computer need no key. Other machines use the access key from Settings as their API key."
          : "Any app on this computer that speaks the OpenAI API can use the loaded model; no key needed.")))
      : null;

    clear(boxes.status,
      h("div", { class: "status-hero" },
        h("div", { style: { minWidth: 0 } }, h("div", { class: "big" }, stateBadge(e.state), h("span", null, name || "No model loaded"))),
        h("div", { class: "row wrap" }, actions)),
      detail ? h("div", { style: { marginTop: "14px" } }, detail) : null,
      unitInfo ? h("div", { style: { marginTop: "14px" } }, unitInfo) : null,
      endpoint ? h("div", { style: { marginTop: "14px" } }, endpoint) : null);
  }

  // "active (running) for 2 h 5 min · enabled"
  function serviceLine(svc) {
    const ago = svc.since ? Math.max(0, Date.now() / 1000 - svc.since) : null;
    const when = ago === null ? "" : svc.active === "active" || svc.active === "activating" ? ` for ${fmt.dur(ago)}` : ` since ${fmt.date(svc.since)}`;
    return `${svc.active || "unknown"}${svc.sub ? ` (${svc.sub})` : ""}${when}${svc.enabled ? ` · ${svc.enabled}` : ""}`;
  }

  function serviceButtons(e) {
    const svc = e.service;
    const out = [];
    if (e.state === "stopped" || e.state === "crashed") {
      out.push(h("button", { class: "btn primary", onclick: () => serviceAct("start") }, icon("play", "sm"), "Start the service"));
    } else {
      const stopping = e.state === "stopping";
      out.push(h("button", { class: "btn", onclick: () => serviceAct("restart"), disabled: stopping }, icon("refresh", "sm"), "Restart"),
        h("button", { class: "btn danger", onclick: () => serviceAct("stop"), disabled: stopping }, icon("power", "sm"),
          stopping ? "Stopping…" : "Stop the service"));
    }
    out.push(h("button", { class: "btn ghost", onclick: stop, title: `FLE Studio stops watching ${svc.unit}; the service is not touched` },
      icon("x", "sm"), "Disconnect"));
    return out;
  }

  async function serviceAct(action) {
    const svc = store.engine?.service;
    if (!svc) return;
    if (action !== "start") {
      const ok = await confirmDialog({
        title: action === "stop" ? `Stop ${svc.unit}?` : `Restart ${svc.unit}?`,
        body: action === "stop" ? "Every app that uses this server loses it until the service runs again."
          : "The server is away while it loads its model again; apps that use it have to wait.",
        ok: action === "stop" ? "Stop" : "Restart", danger: action === "stop" });
      if (!ok) return;
    }
    try {
      store.engine = await post(`/api/engine/service/${action}`);
      emit("engine");
      renderStatus();
    } catch (err) {
      toast(err.message, "bad", 15000);
    }
  }

  function kv(k, v) {
    return [h("dt", null, k), h("dd", null, v)];
  }

  function profileName(id) {
    return (store.state?.profiles || []).find((p) => p.id === id)?.name || id;
  }

  async function start() {
    try {
      store.engine = await post("/api/engine/start", { model: sel.model, profile: sel.profile });
      emit("engine");
      renderStatus();
    } catch (e) {
      toast(e.message, "bad", 9000);
    }
  }

  async function stop() {
    try {
      store.engine = await post("/api/engine/stop");
      emit("engine");
      renderStatus();
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  async function restart() {
    await stop();
    for (let i = 0; i < 60; i += 1) {
      const s = await get("/api/engine");
      if (!["stopping", "ready", "starting"].includes(s.state)) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    await start();
  }

  // ------------------------------------------------------------------------------------------- start settings

  function renderLaunch(force = false) {
    if (!force && typingIn(boxes.launch)) return; // do not rebuild the form under the cursor
    const st = settings();
    const L = st.launch || {};
    const models = usableModels();
    const profiles = store.state?.profiles || [];
    if (mode === "simple") { renderSimpleLaunch(L, models); return; }
    const nvidia = (store.state?.system?.gpus || []).filter((g) => g.vendor === "nvidia");

    const modelSel = h("select", { class: "select", disabled: !models.length,
      onchange: (e) => { sel.model = e.target.value; save({ last_model: sel.model }); renderStatus(); } },
      models.length ? models.map((m) => h("option", { value: m.id, selected: m.id === sel.model }, `${m.name} · ${fmt.bytes(m.size)}`))
        : h("option", { value: "" }, "No model yet"));
    const profSel = h("select", { class: "select",
      onchange: (e) => { sel.profile = e.target.value; save({ launch: { profile: sel.profile } }); renderLaunch(true); renderProfiles(true); refreshPreview(); } },
      h("option", { value: "auto", selected: sel.profile === "auto" }, "Automatic: sized for this machine"),
      profiles.map((p) => h("option", { value: p.id, selected: p.id === sel.profile }, `Profile: ${p.name}`)));

    const auto = sel.profile === "auto";
    const ctxValue = Number(L.context) || 32768;
    const ctxOptions = CONTEXTS.includes(ctxValue) ? CONTEXTS : [...CONTEXTS, ctxValue].sort((a, b) => a - b);
    const gpuOptions = [["auto", nvidia.length > 1 ? "Automatic: the largest GPU" : "Automatic"], ["all", "All NVIDIA GPUs"],
      ...nvidia.map((g) => [String(g.index), `GPU ${g.index} · ${g.name} · ${fmt.gib(g.memory_total, 0)}`])];
    if (!gpuOptions.some(([v]) => v === L.gpus)) gpuOptions.push([L.gpus, `GPUs ${L.gpus}`]);
    const coresHint = h("span", { class: "hint" }, maskHint(L.cpu_mask));
    const eng = store.engine;
    const boundTo = eng?.attached ? eng.service?.unit || "a running server" : "";

    clear(boxes.launch,
      h("div", { class: "card-head" }, icon("sliders"), h("div", null, h("h3", null, "Start settings"),
        h("p", null, "Saved right away. The command on the right updates as you change them."))),
      boundTo ? h("div", { class: "callout", style: { marginBottom: "14px" } }, icon("info", "sm"), h("div", { class: "small" },
        `FLE Studio is connected to ${boundTo}, which runs its own command. These settings are for engines FLE Studio `
        + "starts itself; they apply once you disconnect.")) : null,
      h("div", { class: "stack" },
        field("Model", modelSel),
        field("How to start", profSel, auto ? "" : "The profile's command decides most things. Context, threads, GPUs and cores below fill its placeholders or apply when the command leaves them out."),
        h("div", { class: "grid two" },
          field("Context", h("select", { class: "select", onchange: (e) => save({ launch: { context: Number(e.target.value) } }) },
            ctxOptions.map((c) => h("option", { value: c, selected: c === ctxValue }, `${fmt.ctx(c)} tokens`))),
          auto ? "Tokens the model can see at once; more needs more memory." : "Fills {context}."),
          auto ? field("Requests at once", numberInput(L.parallel, 1, 16, (v) => save({ launch: { parallel: v } })),
            "They share one unified KV cache; the context is total, not per slot.") : field("Threads", numberInput(L.threads, 0, 1024, (v) => save({ launch: { threads: v } })), "0 = one per physical core. Fills {threads}.")),
        nvidia.length ? h("div", { class: "grid two" },
          field("GPUs", h("select", { class: "select", onchange: (e) => save({ launch: { gpus: e.target.value } }) },
            gpuOptions.map(([v, label]) => h("option", { value: v, selected: v === L.gpus }, label))),
          auto ? "Other GPUs stay free for other work." : "Applies when the profile sets no CUDA_VISIBLE_DEVICES."),
          field("CPU cores", coresInput(L.cpu_mask, coresHint), coresHint)) : field("CPU cores", coresInput(L.cpu_mask, coresHint), coresHint),
        auto ? h("details", { class: "stack" },
          h("summary", { class: "small muted", style: { cursor: "pointer" } }, "More: expert cache, threads, memory, port"),
          h("div", { class: "stack", style: { marginTop: "10px" } },
            h("div", { class: "grid two" },
              field("GPU expert cache (MiB)", numberInput(L.gpu_cache_mib, 0, 1 << 20, (v) => save({ launch: { gpu_cache_mib: v } })),
                "FLE engine only. 0 = sized from free GPU memory."),
              field("Threads", numberInput(L.threads, 0, 1024, (v) => save({ launch: { threads: v } })),
                "0 = one per physical core. The FLE engine keeps one core free on processors with one thread per core.")),
            h("div", { class: "grid two" },
              field("Prompt cache in RAM (MiB)", numberInput(L.prompt_cache_mib, -1, 1 << 22, (v) => save({ launch: { prompt_cache_mib: v } })),
                "-1 = automatic, 0 = off. Keeps earlier prompts so follow-ups start faster."),
              field("Engine port", numberInput(L.port, 0, 65535, (v) => save({ launch: { port: v } })), "0 = any free port.")),
            field("Experts in pinned RAM", seg([["auto", "Automatic"], ["on", "On"], ["off", "Off"]], L.pinned_load,
              (v) => save({ launch: { pinned_load: v } })), "FLE engine: faster expert loading on Linux and Windows, costs RAM that cannot be swapped."),
            h("label", { class: "row" }, toggle(L.mtp, (v) => save({ launch: { mtp: v } })),
              h("span", null, "MTP drafting when the MTP head is there ", h("span", { class: "muted small" }, "(FLE engine)"))),
            field("Speculation", h("select", { class: "select", onchange: (e) => save({ launch: { spec: e.target.value } }) },
              [["auto", "Auto (MTP when the sidecars exist)"], ["none", "Off"], ["mtp", "MTP only"], ["ngram-mod", "Ngram (modified)"], ["ngram-simple", "Ngram (simple)"]].map(([v, label]) => h("option", { value: v, selected: v === (L.spec || "auto") }, label))),
              "How the engine drafts ahead. auto only turns on MTP when its head and vocabulary file are present; ngram-mod/ngram-simple need an engine that supports --spec-type and are prompt-dependent."),
            field("Extra arguments", h("input", { class: "input mono", value: L.extra_args || "", placeholder: "--flash-attn on",
              onchange: (e) => save({ launch: { extra_args: e.target.value } }) })),
            field("Extra environment", h("textarea", { class: "textarea mono", rows: 2, placeholder: "KEY=value, one per line",
              value: L.extra_env || "", onchange: (e) => save({ launch: { extra_env: e.target.value } }) })))) : null,
        h("div", { class: "divider" }),
        h("label", { class: "row" }, toggle(L.restart_on_crash, (v) => save({ launch: { restart_on_crash: v } })),
          h("span", null, "Start again if it crashes ", h("span", { class: "muted small" }, "(for a server that should stay up)"))),
        h("label", { class: "row" }, toggle(L.autostart, (v) => save({ launch: { autostart: v } })),
          h("span", null, "Load this model when FLE Studio starts"))));
  }

  function renderSimpleLaunch(L, models) {
    const modelSel = h("select", { class: "select", disabled: !models.length,
      onchange: (e) => { sel.model = e.target.value; save({ last_model: sel.model }); renderStatus(); refreshPreview(); bestCard?.reload(); } },
      models.length ? models.map((m) => h("option", { value: m.id, selected: m.id === sel.model }, `${m.name} · ${fmt.bytes(m.size)}`))
        : h("option", { value: "" }, "No model yet"));
    const ctxValue = Number(L.context) || 32768;
    const ctxOptions = CONTEXTS.includes(ctxValue) ? CONTEXTS : [...CONTEXTS, ctxValue].sort((a, b) => a - b);
    clear(boxes.launch,
      h("div", { class: "card-head" }, icon("sliders"), h("div", null, h("h3", null, "Model"),
        h("p", null, sel.profile === "auto" ? "Starts with automatic settings, sized for this machine."
          : `Starts with the profile “${profileName(sel.profile)}”.`))),
      h("div", { class: "stack" },
        field("Model", modelSel),
        field("Context", h("select", { class: "select", onchange: (e) => save({ launch: { context: Number(e.target.value) } }) },
          ctxOptions.map((c) => h("option", { value: c, selected: c === ctxValue }, `${fmt.ctx(c)} tokens`))),
        "Tokens the model can see at once; more needs more memory."),
        sel.profile !== "auto" ? h("button", { class: "btn", onclick: () => { sel.profile = "auto"; save({ launch: { profile: "auto" } }); renderLaunch(true); refreshPreview(); } },
          "Use automatic settings") : null,
        h("p", { class: "small faint", style: { margin: 0 } }, "Switch to Advanced above for every setting, launch profiles, engines and servers that already run.")));
  }

  function field(label, control, hint = "") {
    return h("label", { class: "field" }, h("span", null, label), control,
      hint instanceof Node ? hint : hint ? h("span", { class: "hint" }, hint) : null);
  }

  function numberInput(value, min, max, onSave) {
    return h("input", { class: "input num", type: "number", min, max, step: 1, value: value ?? 0,
      onchange: (e) => {
        const v = Math.max(min, Math.min(max, Math.round(Number(e.target.value) || 0)));
        e.target.value = v;
        onSave(v);
      } });
  }

  function coresInput(mask, hintEl) {
    return h("input", { class: "input mono", value: maskToRanges(mask) || "", placeholder: "all cores, or e.g. 0-15",
      onchange: (e) => {
        const v = e.target.value.trim();
        let m = "";
        try {
          m = v ? rangesToMask(v) : "";
        } catch (err) {
          toast(err.message, "bad");
          return;
        }
        hintEl.textContent = maskHint(m);
        save({ launch: { cpu_mask: m } });
      } });
  }

  function seg(options, value, onPick) {
    const box = h("div", { class: "seg", role: "radiogroup" });
    for (const [v, label] of options) {
      const b = h("button", { type: "button", class: v === value ? "on" : "", role: "radio", "aria-checked": v === value ? "true" : "false" }, label);
      b.addEventListener("click", () => {
        box.querySelectorAll("button").forEach((x) => { x.classList.remove("on"); x.setAttribute("aria-checked", "false"); });
        b.classList.add("on");
        b.setAttribute("aria-checked", "true");
        onPick(v);
      });
      box.append(b);
    }
    return box;
  }

  function toggle(checked, onFlip) {
    return h("span", { class: "switch" }, h("input", { type: "checkbox", checked, onchange: (e) => onFlip(e.target.checked) }), h("i"));
  }

  // ------------------------------------------------------------------------------------------- command preview

  function renderPreview() {
    const head = h("div", { class: "card-head" }, icon("terminal"), h("div", { class: "grow" }, h("h3", null, "Command"),
      h("p", null, "What Start runs. Paste it into a terminal to run the engine without FLE Studio.")),
    preview ? copyButton(() => preview.command, "Copy the command") : null);
    if (!preview) {
      clear(boxes.preview, head, previewError ? h("div", { class: "callout warn" }, icon("alert", "sm"), h("div", null, previewError)) : null);
      return;
    }
    clear(boxes.preview, head,
      h("div", { class: "cmd" }, highlightCommand(preview.command)),
      explainBlock(preview, { open: mode === "simple" }),
      preview.notes?.length ? h("ul", { class: "small muted", style: { margin: "12px 0 0", paddingLeft: "1.2em", display: "grid", gap: "4px" } },
        preview.notes.map((n) => h("li", null, n))) : null,
      h("div", { class: "row small faint wrap", style: { marginTop: "10px" } },
        h("span", null, `Engine: ${preview.engine}`), h("span", null, `Port ${preview.port}`),
        preview.draft ? h("span", null, "MTP drafting on") : null));
  }

  // ------------------------------------------------------------------------------------------- profiles

  function renderProfiles(force = false) {
    if (!force && typingIn(boxes.profiles)) return;
    const profiles = store.state?.profiles || [];
    if (editing && editing.id && !profiles.some((p) => p.id === editing.id)) editing = null;
    if (!editing && profiles.length) editing = fromProfile(profiles.find((p) => p.id === sel.profile) || profiles[0]);
    const list = h("div", { class: "profile-list" },
      profiles.map((p) => h("button", { class: editing?.id === p.id ? "on" : "", onclick: () => { editing = fromProfile(p); renderProfiles(true); } },
        h("span", null, p.name, p.id === sel.profile ? h("span", { class: "pill info", style: { marginLeft: "8px", height: "18px" } }, "next start") : null),
        h("small", null, p.builtin ? "Built in" : p.engine === "custom" ? "Own engine" : p.engine === "fle" ? "FLE engine" : "llama.cpp"))),
      h("button", { class: editing && !editing.id ? "on" : "", onclick: () => { editing = { id: "", name: "", engine: "fle", command: NEW_COMMAND, notes: "", builtin: false }; renderProfiles(true); } },
        h("span", { class: "row", style: { gap: "6px" } }, icon("plus", "sm"), "New profile"), h("small", null, "Paste any llama-server command")));
    clear(boxes.profiles,
      h("div", { class: "card-head" }, icon("layers"), h("div", null, h("h3", null, "Launch profiles"),
        h("p", null, "A profile is a complete command: environment, engine and flags. Paste one from a script, a systemd unit or your shell history."))),
      h("div", { class: "profiles-grid" }, list, editing ? editor(editing) : h("div", { class: "empty" }, "No profiles yet.")));
  }

  function fromProfile(p) {
    return { id: p.id, name: p.name, engine: p.engine, command: p.command, notes: p.notes || "", builtin: p.builtin };
  }

  function editor(p) {
    if (p.builtin) {
      return h("div", { class: "stack", style: { minWidth: 0 } },
        h("div", { class: "row wrap" }, h("b", null, p.name), h("span", { class: "tag" }, "built in"), h("span", { class: "grow" }),
          h("button", { class: "btn sm", onclick: () => useProfile(p.id) }, icon("play", "sm"), "Use for the next start"),
          h("button", { class: "btn sm", onclick: () => { editing = { ...p, id: "", builtin: false, name: `${p.name} (copy)` }; renderProfiles(); } },
            icon("copy", "sm"), "Copy to edit")),
        h("div", { class: "cmd" }, highlightCommand(p.command), copyButton(p.command)),
        p.notes ? h("p", { class: "small muted", style: { margin: 0 } }, p.notes) : null,
        legend(null));
    }
    const name = h("input", { class: "input", value: p.name, placeholder: "e.g. Rhea production" });
    const cmd = h("textarea", { class: "textarea mono", rows: 7, spellcheck: "false", value: p.command,
      placeholder: "FLE_MOE_CACHE_BUDGET_MIB=16384 {engine} -m {model} -c {context} …" });
    const notes = h("textarea", { class: "textarea", rows: 2, value: p.notes, placeholder: "What this profile is for, where it was measured" });
    const engSel = h("select", { class: "select" },
      h("option", { value: "fle", selected: p.engine === "fle" }, "FLE engine (installed)"),
      h("option", { value: "upstream", selected: p.engine === "upstream" }, "llama.cpp (installed)"));
    // Keep what is typed in `editing`, so a refresh of the page data does not throw away an unsaved command.
    name.addEventListener("input", () => { p.name = name.value; });
    cmd.addEventListener("input", () => { p.command = cmd.value; });
    notes.addEventListener("input", () => { p.notes = notes.value; });
    engSel.addEventListener("change", () => { p.engine = engSel.value; });
    const ownBinary = h("div", { class: "small muted" });
    const engineField = h("label", { class: "field" }, h("span", null, "{engine} is"), engSel);
    const updateBinary = () => {
      const bin = commandBinary(cmd.value);
      const own = bin && bin !== "{engine}" && /[\\/]/.test(bin);
      engineField.hidden = own;
      ownBinary.hidden = !own;
      ownBinary.replaceChildren(icon("info", "sm"), ` This command starts its own engine: `, h("code", { class: "mono" }, bin));
    };
    cmd.addEventListener("input", updateBinary);
    updateBinary();
    const doSave = async (useIt) => {
      const bin = commandBinary(cmd.value);
      const own = bin && bin !== "{engine}" && /[\\/]/.test(bin);
      try {
        const saved = await post("/api/profiles", { id: p.id || undefined, name: name.value.trim(), command: cmd.value,
          notes: notes.value.trim(), engine: own ? "custom" : engSel.value });
        editing = fromProfile(saved);
        await refreshState();
        if (useIt) await useProfile(saved.id);
        toast(`Saved “${saved.name}”`, "good");
      } catch (e) {
        toast(e.message, "bad", 8000);
      }
    };
    return h("div", { class: "stack", style: { minWidth: 0 } },
      h("div", { class: "grid two" }, field("Name", name), engineField),
      h("label", { class: "field" }, h("span", null, "Command"), cmd, ownBinary),
      legend(cmd),
      field("Notes", notes),
      h("div", { class: "row wrap" },
        h("button", { class: "btn primary", onclick: () => doSave(true) }, icon("check", "sm"), p.id ? "Save and use" : "Save and use for the next start"),
        h("button", { class: "btn", onclick: () => doSave(false) }, "Save"),
        h("span", { class: "grow" }),
        p.id ? h("button", { class: "btn ghost danger", onclick: () => removeProfile(p) }, icon("trash", "sm"), "Delete") : null),
      h("p", { class: "small faint", style: { margin: 0 } },
        "FLE Studio adds --host 127.0.0.1, a free --port, --metrics and an alias when the command leaves them out; it needs them to talk to the engine."));
  }

  function legend(target) {
    return h("div", { class: "ph-legend" }, h("span", { class: "small muted" }, "Placeholders:"),
      PLACEHOLDERS.map(([ph, what]) => h("code", { title: what, onclick: target ? () => insertAtCursor(target, ph) : null }, ph)));
  }

  function insertAtCursor(ta, text) {
    const s = ta.selectionStart ?? ta.value.length;
    const e = ta.selectionEnd ?? s;
    ta.value = ta.value.slice(0, s) + text + ta.value.slice(e);
    ta.focus();
    ta.selectionStart = ta.selectionEnd = s + text.length;
    ta.dispatchEvent(new Event("input"));
  }

  async function useProfile(id) {
    sel.profile = id;
    save({ launch: { profile: id } });
    renderLaunch(true);
    renderProfiles(true);
    refreshPreview();
  }

  async function removeProfile(p) {
    if (!(await confirmDialog({ title: `Delete the profile “${p.name}”?`, body: "Only the saved command goes; models and engines stay.", ok: "Delete", danger: true }))) return;
    try {
      await post(`/api/profiles/${p.id}/delete`);
      editing = null;
      await refreshState();
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  // ------------------------------------------------------------------------------------------- engines

  function renderEngines() {
    const st = store.state;
    if (!st || typingIn(boxes.engines)) return;
    const choice = st.settings.engine?.choice || "auto";
    const installed = st.engines?.installed || [];
    const variants = st.engines?.variants || [];
    const jobs = store.downloads.filter((d) => d.kind === "engine" && !["done", "cancelled"].includes(d.state));
    const custom = h("input", { class: "input mono", value: st.settings.engine?.custom_path || "",
      placeholder: navigator.userAgent.includes("Windows") ? "C:\\llama.cpp\\llama-server.exe" : "/path/to/llama-server" });
    const result = h("div");
    const showCheck = () => {
      if (!checkResult) { clear(result); return; }
      const r = checkResult;
      clear(result, r.ok
        ? h("div", { class: "callout good" }, icon("ok", "sm"), h("div", null,
          h("b", null, r.flavor === "fle" ? "FLE engine" : "llama.cpp"), ` · ${r.backend || "cpu"}${r.version ? ` · ${r.version}` : ""}`))
        : h("div", { class: "callout bad" }, icon("alert", "sm"), h("div", { class: "small" }, r.problem || "It does not start")));
    };
    showCheck();
    const check = async () => {
      try {
        checkResult = await post("/api/engines/check", { path: custom.value.trim() });
        showCheck();
        if (checkResult.ok) save({ engine: { custom_path: custom.value.trim() } });
      } catch (e) { toast(e.message, "bad"); }
    };

    clear(boxes.engines,
      h("div", { class: "card-head" }, icon("engine"), h("div", null, h("h3", null, "Engines"),
        h("p", null, "Prebuilt llama.cpp and FLE builds, downloaded from GitHub and checked before use."))),
      h("div", { class: "stack" },
        field("Use", seg([["auto", "Best installed"], ["fle", "FLE"], ["upstream", "llama.cpp"], ["custom", "My own build"]], choice,
          (v) => { save({ engine: { choice: v } }); })),
        installed.length ? h("div", null, installed.map((e) => h("div", { class: "variant" },
          h("div", { style: { minWidth: 0 } }, h("b", null, `${e.kind === "fle" ? "FLE engine" : "llama.cpp"} · ${e.backend || ""}`),
            h("span", null, `${e.tag || ""}${e.version ? ` · ${e.version}` : ""}`),
            h("div", { class: "small faint mono", style: { overflowWrap: "anywhere" } }, e.path),
            e.ok === false ? h("div", { class: "small", style: { color: "var(--bad)" } }, e.problem) : null),
          h("button", { class: "btn ghost icon sm", title: "Remove this engine", "aria-label": "Remove", onclick: () => removeEngine(e) }, icon("trash", "sm")))))
          : choice === "custom" && st.settings.engine?.custom_path
            ? h("div", { class: "callout" }, icon("info", "sm"), h("div", null, "No prebuilt engine installed. FLE Studio uses your own build (below)."))
            : h("div", { class: "callout" }, icon("info", "sm"), h("div", null, "No engine installed yet. Install one below or point to your own llama-server.")),
        jobs.map((j) => h("div", { class: "stack tight" },
          h("div", { class: "row small" }, h("b", null, j.title), h("span", { class: "grow" }), h("span", { class: "muted num" }, `${Math.round((j.progress || 0) * 100)} %`)),
          h("div", { class: "bar" }, h("i", { style: { width: `${Math.round((j.progress || 0) * 100)}%` } })),
          j.error ? h("div", { class: "small", style: { color: "var(--bad)" } }, j.error) : null)),
        h("details", null, h("summary", { class: "small muted", style: { cursor: "pointer" } }, "Install an engine"),
          h("div", { style: { marginTop: "6px" } }, variants.map((v) => h("div", { class: "variant" },
            h("div", null, h("b", null, v.label), h("span", null, v.why)),
            h("button", { class: "btn sm", onclick: () => install(v) }, icon("download", "sm"),
              installed.some((e) => e.variant === v.id) ? "Update" : "Install"))))),
        h("div", { class: "divider" }),
        h("label", { class: "field" }, h("span", null, "Your own llama-server"), h("div", { class: "row" }, custom,
          h("button", { class: "btn", onclick: check }, "Check"))),
        result,
        h("span", { class: "small faint" }, "Any build works, upstream llama.cpp or FLE; FLE Studio reads --help to tell them apart. Choose “My own build” above to use it.")));
  }

  async function install(v) {
    try {
      await post("/api/engines/install", { variant: v.id });
      await refreshState();
      toast(`Downloading ${v.label}`);
    } catch (e) {
      toast(e.message, "bad", 8000);
    }
  }

  async function removeEngine(e) {
    if (!(await confirmDialog({ title: "Remove this engine?", body: e.path, ok: "Remove", danger: true }))) return;
    try {
      await post(`/api/engines/${encodeURIComponent(e.id)}/remove`);
      await refreshState();
    } catch (err) {
      toast(err.message, "bad");
    }
  }

  // ------------------------------------------------------------------------------------------- attach

  function renderAttach() {
    if (typingIn(boxes.attach)) return;
    const st = settings();
    const e = store.engine || {};
    const systemd = !!store.state?.systemd;
    const unit = systemd ? h("input", { class: "input mono", value: st.attach_service || "", placeholder: "rhea.service",
      spellcheck: false }) : null;
    const url = h("input", { class: "input mono", value: st.attach_url || "",
      placeholder: systemd ? "From the service, or http://127.0.0.1:8080" : "http://127.0.0.1:8080" });
    const key = h("input", { class: "input mono", type: "password", autocomplete: "off",
      placeholder: st.has_attach_key ? "Saved; leave empty to keep it" : "Only if the server wants one" });
    const connect = async () => {
      try {
        store.engine = await post("/api/engine/attach", { url: url.value.trim(), ...(unit ? { service: unit.value.trim() } : {}),
          ...(key.value ? { key: key.value } : {}) });
        key.value = "";
        emit("engine");
        renderStatus();
        const s = store.engine;
        toast(s.service ? `Connected to ${s.service.unit}` : s.reachable ? "Connected" : s.error || "Connected, but the server does not answer yet",
          s.service || s.reachable ? "good" : "info");
      } catch (err) {
        toast(err.message, "bad", 8000);
      }
    };
    clear(boxes.attach,
      h("div", { class: "card-head" }, icon("link"), h("div", null, h("h3", null, "Use a server that already runs"),
        h("p", null, systemd
          ? "Chat with any llama.cpp or FLE server and watch its numbers. Name its systemd service as well, and FLE Studio "
            + "follows the service's log and loading progress and can start, stop and restart it."
          : "Chat with any llama.cpp or FLE server and watch its numbers. FLE Studio does not start or stop it."))),
      h("div", { class: "stack" },
        unit ? field("Service", unit, "Optional. A llama-server that runs as a systemd service; the address then comes from its command.") : null,
        field("Address", url),
        field("API key", key),
        h("div", { class: "row" },
          e.attached ? h("button", { class: "btn", onclick: stop }, icon("x", "sm"), "Disconnect")
            : h("button", { class: "btn", onclick: connect, disabled: ["starting", "ready"].includes(e.state) }, icon("link", "sm"), "Connect"),
          ["starting", "ready"].includes(e.state) ? h("span", { class: "small muted" }, "Stop the engine first.") : null)));
  }

  // ------------------------------------------------------------------------------------------- log

  let logLines = [];
  let logSeq = 0;
  let follow = true;
  let filter = "";
  const logView = h("div", { class: "log", role: "log", "aria-live": "off" });
  logView.addEventListener("scroll", () => {
    follow = logView.scrollTop + logView.clientHeight >= logView.scrollHeight - 30;
  });

  function lineClass(t) {
    if (t.startsWith("$ ")) return "ln cmdline";
    if (t.startsWith("FLE Studio:")) return "ln ok";
    if (/\b(error|failed|fatal|abort|exception|out of memory)\b/i.test(t)) return "ln err";
    if (/\bwarn(ing)?\b/i.test(t)) return "ln warn";
    if (/server is listening|model loaded|all slots are idle/i.test(t)) return "ln ok";
    return "ln";
  }

  function lineEl(l) {
    return h("div", { class: lineClass(l.text) }, l.text);
  }

  let placeholderShown = false;
  function renderLogAll() {
    const shown = filter ? logLines.filter((l) => l.text.toLowerCase().includes(filter)) : logLines;
    logView.replaceChildren(...shown.slice(-2000).map(lineEl));
    placeholderShown = !shown.length;
    if (placeholderShown) logView.append(h("div", { class: "ln" }, filter ? "No line matches." : "The engine has not written anything yet."));
    if (follow) logView.scrollTop = logView.scrollHeight;
  }

  let logBusy = false;
  async function fetchLog() {
    if (logBusy) return;
    logBusy = true;
    try {
      const r = await get(`/api/engine/log?since=${logSeq}`);
      logSeq = r.seq;
      if (r.lines.length) {
        logLines.push(...r.lines);
        if (logLines.length > 5000) logLines = logLines.slice(-5000);
        if (filter || placeholderShown) renderLogAll();
        else {
          for (const l of r.lines) logView.append(lineEl(l));
          while (logView.childElementCount > 2000) logView.firstChild.remove();
          if (follow) logView.scrollTop = logView.scrollHeight;
        }
      } else if (!logLines.length) {
        renderLogAll();
      }
    } catch { /* the next engine event tries again */ }
    logBusy = false;
  }

  const logSub = h("p");
  function renderLogSub() {
    const svc = store.engine?.service;
    logSub.textContent = svc
      ? `The journal of ${svc.unit}, live. All of it: journalctl ${svc.user ? "--user-unit" : "-u"} ${svc.unit}`
      : `Everything llama-server prints. Full logs: ${store.state?.data_dir || "the data folder"}/logs`;
  }

  function renderLogCard() {
    const search = h("input", { class: "input", placeholder: "Filter lines", style: { maxWidth: "220px" },
      oninput: debounce((e) => { filter = e.target.value.trim().toLowerCase(); renderLogAll(); }, 150) });
    renderLogSub();
    clear(boxes.log,
      h("div", { class: "card-head" }, icon("terminal"), h("div", { class: "grow" }, h("h3", null, "Engine log"), logSub),
      search,
      h("button", { class: "btn ghost icon sm", title: "Copy the log", "aria-label": "Copy the log",
        onclick: (e) => copyText(logLines.map((l) => l.text).join("\n"), e.currentTarget) }, icon("copy", "sm")),
      h("button", { class: "btn ghost icon sm", title: "Jump to the end", "aria-label": "Jump to the end",
        onclick: () => { follow = true; logView.scrollTop = logView.scrollHeight; } }, icon("down", "sm"))),
      logView);
  }

  // ------------------------------------------------------------------------------------------- wiring

  function renderAll() {
    syncSelection();
    applyMode();
    renderStatus();
    renderLaunch();
    renderProfiles();
    renderEngines();
    renderAttach();
    refreshPreview();
  }

  let lastState = store.engine?.state;
  unsub.push(on("state", renderAll));
  let lastUnit = store.engine?.service?.unit;
  unsub.push(on("engine", () => {
    renderStatus();
    if (store.engine?.log_seq !== logSeq) fetchLog();
    if (store.engine?.state !== lastState || store.engine?.service?.unit !== lastUnit) {
      lastState = store.engine?.state;
      lastUnit = store.engine?.service?.unit;
      renderLaunch();
      renderAttach();
      renderLogSub();
    }
  }));
  unsub.push(on("downloads", () => { if (store.downloads.some((d) => d.kind === "engine")) renderEngines(); }));
  renderMode();
  applyMode();
  const bestCard = mountBest(boxes.best, {
    model: () => sel.model,
    profile: () => sel.profile,
    engineRunning: () => !["stopped", "crashed"].includes(store.engine?.state) || !!store.engine?.service,
    onUse: async (profileId) => {
      sel.profile = profileId;
      save({ launch: { profile: profileId } });
      await start();
      renderLaunch(true);
      refreshPreview();
    },
  });
  unsub.push(on("engine", () => bestCard.reload()));
  renderLogCard();
  if (store.state) renderAll();
  fetchLog();
  refreshState().catch(() => {});

  return { unmount() { unsub.forEach((u) => u()); bestCard.unmount(); } };
}

// --------------------------------------------------------------------------------------------- helpers

// True while the cursor is in a text field or menu inside `box`: rebuilding it then would lose the input.
function typingIn(box) {
  const a = document.activeElement;
  return !!a && box.contains(a) && /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName);
}

// The first word of a command that is not a KEY=value assignment: the program it starts.
function commandBinary(text) {
  const words = (text || "").replace(/\\\r?\n/g, " ").match(/"[^"]*"|'[^']*'|\S+/g) || [];
  const w = words.find((x) => !/^[A-Za-z_][A-Za-z0-9_]*=/.test(x) && x !== "env");
  return w ? w.replace(/^["']|["']$/g, "") : "";
}

function highlightCommand(text) {
  const out = [];
  for (const part of (text || "").split(/(\s+)/)) {
    if (!part) continue;
    if (/^\s+$/.test(part)) out.push(part);
    else if (/^[A-Z_][A-Z0-9_]*=/.test(part)) {
      const i = part.indexOf("=");
      out.push(h("span", { class: "env", title: flagTitle(part) || null }, part.slice(0, i)), part.slice(i));
    } else if (/^--?[A-Za-z]/.test(part)) out.push(h("span", { class: "flag", title: flagTitle(part) || null }, part));
    else if (/\{[a-z_]+\}/.test(part)) out.push(h("span", { class: "ph" }, part));
    else out.push(part);
  }
  return out;
}

// "0-15,32" -> "0x10000FFFF"; a hex mask passes through.
function rangesToMask(text) {
  const t = text.replace(/\s+/g, "");
  if (/^0x[0-9a-f]+$/i.test(t)) return `0x${t.slice(2).toUpperCase()}`;
  if (!/^\d+(-\d+)?(,\d+(-\d+)?)*$/.test(t)) throw new Error("Write cores like 0-15 or 0-7,16-23, or a hex mask like 0x0000FFFF");
  let mask = 0n;
  for (const part of t.split(",")) {
    const [a, b = a] = part.split("-").map(Number);
    if (b < a || b > 1023) throw new Error(`Core range ${part} does not make sense`);
    for (let i = a; i <= b; i += 1) mask |= 1n << BigInt(i);
  }
  return `0x${mask.toString(16).toUpperCase().padStart(8, "0")}`;
}

function maskToRanges(mask) {
  if (!mask || !/^0x[0-9a-f]+$/i.test(mask)) return mask || "";
  let m = BigInt(mask);
  const cores = [];
  for (let i = 0; m > 0n; i += 1, m >>= 1n) if (m & 1n) cores.push(i);
  const out = [];
  for (let i = 0; i < cores.length; i += 1) {
    let j = i;
    while (j + 1 < cores.length && cores[j + 1] === cores[j] + 1) j += 1;
    out.push(i === j ? `${cores[i]}` : `${cores[i]}-${cores[j]}`);
    i = j;
  }
  return out.join(",");
}

function maskHint(mask) {
  return mask ? `Pinned to cores ${maskToRanges(mask)} (${mask}, --cpu-strict 1).` : "Empty: the system schedules the threads.";
}
