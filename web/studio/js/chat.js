// The chat page: conversation (a tree of messages with versions), composer with attachments, the tool loop,
// and the live panel with the engine's numbers.

import { get, hostUrl, post, put, streamChat, upload } from "./api.js";
import { emit, engineModelName, engineUp, modelMenuProviders, navigate, on, refreshChats, store, topbar } from "./app.js";
import { attachmentsEl, createAttach } from "./attach.js";
import { attachmentBlock, buildRequest, estimateRequest, estimateTokens, usedImages } from "./chat-request.js";
import { addChild, ensureTree, indexOf, newId, pathOf, removeBranch, siblingsOf, switchTo } from "./chat-tree.js";
import { createStreamRenderer, renderMarkdown } from "./md.js";
import { host as appHost, replaceMessages, swapNode } from "./app-bridge.js";
import { showAppNow, toolCardEl, waitForApproval } from "./tools-ui.js";
import { clear, confirmDialog, copyText, debounce, fmt, h, icon, mark, meter, openMenu, ring, sparkline, stateBadge, toast } from "./ui.js";

const SUGGESTIONS = [
  ["Explain mixture-of-experts", "and why only a few experts run for each token"],
  ["Write a Python script", "that hashes every file in a folder in parallel"],
  ["Summarize a text for me", "paste an article and get the key points"],
  ["Plan three days in Lisbon", "on a budget, with one day trip"],
];

const imageCache = new Map(); // picture file name → data URL, for requests
const PAINT_MS = 80; // an answer that streams is drawn at most about 12 times a second
const CURSOR = '<span class="cursor"></span>';

async function loadImages(nodes) {
  for (const blob of usedImages(nodes)) {
    if (imageCache.has(blob)) continue;
    const res = await fetch(`/api/attachments/${blob}`);
    if (!res.ok) continue;
    const file = await res.blob();
    const data = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(file);
    });
    if (data) imageCache.set(blob, data);
  }
  return imageCache;
}

export function mount(main, chatId) {
  let chat = newChat();
  let streaming = null; // the message that is being written right now
  let turn = null; // the whole answer: streaming, running tools, streaming again. {ctrl}
  let pinned = true;
  let liveOpen = false;
  let caps = { vision: false, tools: null, n_ctx: null, max_upload_mb: 25, pdf: true };
  let toolInfo = { enabled: false, tools: [], servers: [], max_rounds: 8, connecting: false };
  let toolPolls = 0;
  let ctxHistory = { key: "", used: 0, exact: false }; // the context count, see contextUse (declared here: the first render runs early)
  let ctxAtt = { key: "", tokens: 0 };
  let alive = true;
  const unsub = [];

  // ------------------------------------------------------------------------------------------- layout

  const title = h("h1", { title: "Rename", style: { cursor: "text" } }, "New chat");
  title.addEventListener("click", rename);
  const modelChip = h("button", { class: "chip" });
  modelChip.addEventListener("click", () => modelMenu(modelChip));
  const liveBtn = h("button", { class: "btn ghost icon", title: "Live data", "aria-label": "Live data",
    onclick: () => { liveOpen = !liveOpen; live.classList.toggle("open", liveOpen); } }, icon("panel"));
  const moreBtn = h("button", { class: "btn ghost icon", title: "Export or import chats", "aria-label": "Chat menu",
    onclick: () => chatMenu(moreBtn) }, icon("more"));
  const importInput = h("input", { type: "file", hidden: true, accept: ".json,.md,.markdown,application/json,text/markdown,text/plain",
    "aria-label": "Import a chat file" });
  importInput.addEventListener("change", async () => { const f = importInput.files[0]; importInput.value = ""; if (f) await importChat(f); });

  const inner = h("div", { class: "messages-inner" });
  const messages = h("div", { class: "messages" }, inner);
  messages.addEventListener("scroll", () => {
    pinned = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 90;
  });
  messages.addEventListener("click", onMessagesClick);

  const attach = createAttach({ getCaps: () => caps, onChange: () => renderComposer() });
  const ta = h("textarea", { rows: 1, placeholder: "Message the model", "aria-label": "Message" });
  const sendBtn = h("button", { class: "send", "aria-label": "Send", title: "Send (Enter)" }, icon("send"));
  const thinkChip = h("button", { class: "chip", title: "Let the model think before it answers" });
  const sampChip = h("button", { class: "chip", title: "Sampling settings" });
  const toolsChip = h("button", { class: "chip", title: "Tools the model may call" });
  const ctxMeter = h("span", { class: "ctx-meter num" });
  const ctxWarn = h("div", { class: "ctx-warn", role: "status", hidden: true });
  const composer = h("div", { class: "composer" }, attach.chips, ta,
    h("div", { class: "bar-row" }, attach.button, thinkChip, toolsChip, sampChip, h("span", { class: "grow" }), ctxMeter, sendBtn), attach.input);
  const note = h("div", { class: "composer-note" });
  const live = h("aside", { class: "live-panel", "aria-label": "Live data" });

  main.replaceChildren(
    topbar(title, modelChip, liveBtn, moreBtn),
    h("div", { class: "chat" },
      h("div", { class: "chat-col" }, messages, h("div", { class: "composer-wrap" }, ctxWarn, composer, note)),
      live),
    importInput);

  ta.addEventListener("input", autosize);
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
    if (e.key === "Escape" && turn) stop();
  });
  ta.addEventListener("paste", (e) => attach.onPaste(e));
  const dropTarget = main.querySelector(".chat-col");
  for (const name of ["dragenter", "dragover"]) {
    dropTarget.addEventListener(name, (e) => { if (e.dataTransfer?.types?.includes("Files")) { e.preventDefault(); dropTarget.classList.add("dropping"); } });
  }
  dropTarget.addEventListener("dragleave", (e) => { if (!dropTarget.contains(e.relatedTarget)) dropTarget.classList.remove("dropping"); });
  dropTarget.addEventListener("drop", (e) => { dropTarget.classList.remove("dropping"); attach.onDrop(e); });
  sendBtn.addEventListener("click", () => (turn ? stop() : submit()));
  thinkChip.addEventListener("click", toggleThinking);
  sampChip.addEventListener("click", () => samplingMenu(sampChip));
  toolsChip.addEventListener("click", () => toolsMenu(toolsChip));

  unsub.push(on("engine", () => { renderHeader(); renderComposer(); if (!chat.messages.length) renderMessages(); fetchCaps(); }));
  unsub.push(on("state", () => { renderComposer(); if (!chat.messages.length) renderMessages(); }));
  unsub.push(on("live", renderLive));

  if (chatId) load(chatId); else { renderMessages(); ta.focus(); }
  renderHeader();
  renderComposer();
  renderLive();
  fetchCaps();
  fetchTools();

  // ------------------------------------------------------------------------------------------- data

  function newChat() {
    return { id: null, title: "", created: 0, model: "", system: "", leaf: null, messages: [] };
  }

  const newAssistant = () => ({ role: "assistant", content: "", reasoning: "", model: engineModelName(), t: Date.now() / 1000 });
  const nowSec = () => Date.now() / 1000;

  async function load(id) {
    try {
      chat = ensureTree(await get(`/api/chats/${id}`));
    } catch {
      toast("That chat no longer exists", "bad");
      navigate("/", { replace: true });
      return;
    }
    for (const m of chat.messages) {  // a page reload ended these calls
      for (const c of m.tool_calls || []) {
        if (["streaming", "pending", "waiting", "running"].includes(c.status)) { c.status = "error"; c.result = c.result || "Interrupted before it finished."; }
      }
    }
    renderHeader();
    renderMessages();
    renderComposer();
    scrollBottom(true);
    ta.focus();
  }

  async function fetchCaps() {
    try {
      caps = await get("/api/chat/capabilities");
      renderComposer();
    } catch { /* the next engine change asks again */ }
  }

  async function fetchTools() {
    try {
      toolInfo = await get("/api/tools");
    } catch { return; }
    renderComposer();
    if (toolInfo.connecting && alive && toolPolls < 60) { toolPolls += 1; setTimeout(fetchTools, 1500); } else toolPolls = 0;
  }

  // What an MCP App may ask of the chat (app-bridge.js asks the user first, then calls these).
  appHost.trusted = (server) => (store.state?.settings?.tools?.mcp_trusted || []).includes(server);
  appHost.say = async (text) => {
    if (turn) throw new Error("The assistant is still answering; try again in a moment");
    if (!engineUp()) throw new Error("No model is loaded");
    ta.value = text;
    autosize();
    await submit();
  };

  // Function declarations, not consts: the first render above runs before this point is reached.
  function toolsActive() { return toolInfo.enabled && toolInfo.tools.length > 0 && caps.tools !== false; }
  function infoFor(name) { return toolInfo.tools.find((t) => t.name === name); }

  // Saves run one after another: the first one creates the chat, so a second one must wait for its id.
  let saving = Promise.resolve();
  function save() {
    saving = saving.then(saveNow, saveNow);
    return saving;
  }

  async function saveNow() {
    const body = { ...chat, model: chat.model || engineModelName() };
    try {
      const saved = chat.id ? await put(`/api/chats/${chat.id}`, body) : await post("/api/chats", body);
      // Keep our own message objects: an answer may still be streaming into them.
      Object.assign(chat, { id: saved.id, title: saved.title, created: saved.created, updated: saved.updated });
      if (location.pathname !== `/c/${chat.id}`) history.replaceState(null, "", `/c/${chat.id}`);
      renderHeader();
      refreshChats();
    } catch (e) {
      toast(`Could not save the chat: ${e.message}`, "bad");
    }
  }

  // ------------------------------------------------------------------------------------------- header

  function renderHeader() {
    title.textContent = chat.title || "New chat";
    const e = store.engine;
    const name = engineModelName();
    clear(modelChip, e && (e.state === "ready" || e.state === "attached") ? h("span", { class: "pill good", style: { height: "18px", padding: "0 6px" } }, h("span", { class: "dot" })) :
      e?.state === "starting" ? h("span", { class: "pill info live", style: { height: "18px", padding: "0 6px" } }, h("span", { class: "dot" })) : null,
    h("span", { style: { maxWidth: "260px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } },
      name || "No model loaded"), icon("down", "sm"));
  }

  async function rename() {
    if (!chat.id) return;
    const input = h("input", { class: "input", value: chat.title, style: { height: "30px", maxWidth: "420px" } });
    title.replaceChildren(input);
    input.focus();
    input.select();
    const done = async (keep) => {
      input.removeEventListener("blur", onBlur);
      if (keep && input.value.trim() && input.value.trim() !== chat.title) {
        chat.title = input.value.trim();
        await save();
      }
      renderHeader();
    };
    const onBlur = () => done(true);
    input.addEventListener("blur", onBlur);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") input.blur();
      if (e.key === "Escape") done(false);
    });
  }

  function modelMenu(anchor) {
    const svc = store.engine?.service;
    if (svc) { // the service loads its own model with its own command
      const tail = [{ label: "Start, stop and log: Engine page", icon: "engine", onclick: () => navigate("/engine") }];
      // Plugins add entries, for example a model switcher next to the service (plugin switch).
      Promise.all(modelMenuProviders.map((p) => p(svc).catch(() => null))).then((lists) => {
        const extra = lists.filter(Boolean).flat();
        openMenu(anchor, extra.length ? [...extra, "sep", ...tail] : [{ head: svc.unit }, ...tail]);
      });
      return;
    }
    const models = (store.state?.models || []).filter((m) => m.kind === "model" && m.complete);
    const cur = store.engine?.model?.id;
    const items = [{ head: "Load a model" }];
    for (const m of models) {
      items.push({ label: m.name, hint: m.id === cur && engineUp() ? "loaded" : fmt.bytes(m.size), icon: m.id === cur ? "check" : "models",
        onclick: () => loadModel(m.id) });
    }
    if (!models.length) items.push({ label: "No models yet: get one", icon: "download", onclick: () => navigate("/models") });
    items.push("sep");
    if (engineUp() && store.engine.state === "ready") items.push({ label: "Eject the model", icon: "power", onclick: ejectModel });
    items.push({ label: "Engine and launch settings", icon: "engine", onclick: () => navigate("/engine") });
    openMenu(anchor, items);
  }

  // ------------------------------------------------------------------------------------------- export and import

  function chatMenu(anchor) {
    const has = !!chat.id;
    openMenu(anchor, [
      { head: "This chat" },
      { label: "Export as JSON (all versions)", icon: "download", onclick: () => exportChat("json") },
      { label: "Export as Markdown (the version on screen)", icon: "download", onclick: () => exportChat("md") },
      "sep",
      { head: "Chats" },
      { label: "Import a chat (JSON or Markdown)", icon: "upload", onclick: () => importInput.click() },
    ].filter((it) => has || !/Export/.test(it.label || "")));
  }

  async function exportChat(format) {
    if (!chat.id) { toast("Nothing to export yet"); return; }
    await save();
    const a = h("a", { href: hostUrl(`/api/chats/${chat.id}/export?format=${format}`), download: "" });
    document.body.append(a);
    a.click();
    a.remove();
  }

  async function importChat(file) {
    try {
      const made = await upload("/api/chats/import", file);
      await refreshChats();
      toast(`Imported “${made.title}” (${made.messages} messages)`, "good");
      navigate(`/c/${made.id}`);
    } catch (e) {
      toast(`Could not import ${file.name}: ${e.message}`, "bad", 9000);
    }
  }

  // ------------------------------------------------------------------------------------------- messages

  function view() {
    const idx = indexOf(chat);
    return { idx, path: pathOf(chat, idx) };
  }

  function renderMessages() {
    if (!chat.messages.length) {
      clear(inner, welcome());
      return;
    }
    const v = view();
    replaceMessages(inner, v.path.map((m, i) => msgEl(m, i, v)));
  }

  function welcome() {
    const e = store.engine || {};
    const st = store.state || {};
    const models = (st.models || []).filter((m) => m.kind === "model" && m.complete);
    const box = h("div", { class: "welcome" });
    if (e.state === "ready" || e.state === "attached") {
      box.append(
        h("div", { style: { width: "44px", height: "44px" } }, mark(44)),
        h("h2", null, "What should ", h("span", { class: "grad" }, engineModelName() || "the model"), " work on?"),
        h("p", { class: "lead" }, "Answers stream from the engine on this computer. Speed, MTP acceptance and the expert cache show in the live panel. Attach files with the paperclip or drop them here."),
        h("div", { class: "suggest" }, SUGGESTIONS.map(([a, b]) => h("button", { onclick: () => { ta.value = `${a} ${b}`; autosize(); submit(); } },
          h("b", null, a), h("span", null, b)))));
      return box;
    }
    if (e.state === "starting") {
      const p = e.load?.progress || 0;
      box.append(
        h("h2", null, "Loading ", h("span", { class: "grad" }, engineModelName())),
        h("p", { class: "lead" }, e.phase || "Starting the engine"),
        h("div", { class: "card" }, h("div", { class: "stack" },
          h("div", { class: `bar${p ? "" : " indet"}` }, h("i", { style: { width: `${(p * 100).toFixed(1)}%` } })),
          h("div", { class: "row between small muted num" },
            h("span", null, e.load?.total ? `${fmt.bytes(e.load.read)} of ${fmt.bytes(e.load.total)} read` : "Preparing"),
            h("span", null, e.load?.elapsed ? fmt.dur(e.load.elapsed) : "")),
          e.hint ? h("div", { class: "callout info" }, icon("info"), h("div", null, e.hint)) : null)));
      return box;
    }
    if (e.service) {
      const svc = e.service;
      box.append(
        h("div", { style: { width: "44px", height: "44px" } }, mark(44)),
        h("h2", null, h("span", { class: "grad" }, svc.unit), e.state === "stopping" ? " is stopping" : e.state === "crashed" ? " stopped with an error" : " is not running"),
        h("p", { class: "lead" }, "FLE Studio is connected to this service. Starting it runs the service's own command, which loads its model."),
        e.error ? h("div", { class: "callout bad" }, icon("alert"), h("div", null, e.error)) : null,
        h("div", { class: "row wrap", style: { justifyContent: "center" } },
          e.state === "stopping" ? null : h("button", { class: "btn primary", onclick: startService }, icon("play"), "Start the service"),
          h("a", { class: "btn", href: "/engine", "data-link": true }, icon("engine"), "Engine page and log")));
      return box;
    }
    const hasModel = models.length > 0;
    box.append(
      h("div", { style: { width: "44px", height: "44px" } }, mark(44)),
      h("h2", null, "Run ", h("span", { class: "grad" }, "Qwen3.8-Flash-Next"), " on your own machine"),
      h("p", { class: "lead" }, "FLE Studio downloads a model, starts the FLE engine with settings that fit your hardware, and shows what the engine is doing while you chat."));
    if (e.state === "crashed") {
      box.append(h("div", { class: "callout bad" }, icon("alert"), h("div", null,
        h("b", null, "The engine stopped. "), e.error || "", e.hint ? h("div", { class: "small", style: { marginTop: "4px" } }, e.hint) : null,
        h("div", { style: { marginTop: "8px" } }, h("a", { href: "/engine", "data-link": true }, "Open the engine log")))));
    }
    const last = models.find((m) => m.id === st.settings?.last_model) || models[0];
    box.append(h("div", { class: "steps" },
      step(1, hasModel ? "done" : "now", "Get a model", hasModel ? `${models.length} in your library` : "Pick an FLE quant that fits your memory, or any GGUF from Hugging Face.",
        hasModel ? null : h("a", { class: "btn primary", href: "/models", "data-link": true }, icon("download"), "Choose a model")),
      step(2, hasModel ? "now" : "", "Load it", hasModel ? last.name : "FLE Studio picks the launch settings for your GPU and RAM.",
        hasModel ? h("button", { class: "btn primary", onclick: () => loadModel(last.id) }, icon("play"), "Load") : null),
      step(3, "", "Chat", "Ask anything. Nothing leaves this computer.", null)));
    if (!hasModel) {
      box.append(h("p", { class: "small muted" }, "Already running a llama-server somewhere? ",
        h("a", { href: "/engine", "data-link": true }, "Connect to it"), " instead."));
    }
    return box;
  }

  function step(n, cls, titleText, sub, action) {
    return h("div", { class: `step ${cls}` }, h("div", { class: "n" }, cls === "done" ? icon("check", "sm") : n),
      h("div", null, h("b", null, titleText), h("span", null, sub)), action || h("span"));
  }

  // "2 / 3" with arrows, for a message that exists in more than one version.
  function switcherEl(m, idx) {
    const sibs = siblingsOf(m, idx);
    if (sibs.length < 2) return null;
    const at = sibs.indexOf(m);
    const go = (to) => () => { if (turn) return; switchTo(chat, to); renderMessages(); scrollBottom(); if (chat.id) save(); };
    return h("div", { class: "switcher", role: "group", "aria-label": "Versions of this message" },
      h("button", { class: "btn ghost icon sm", title: "Previous version", "aria-label": "Previous version", disabled: at === 0 || !!turn,
        onclick: go(sibs[at - 1]) }, icon("chevron", "sm flip")),
      h("span", { class: "num" }, `${at + 1} / ${sibs.length}`),
      h("button", { class: "btn ghost icon sm", title: "Next version", "aria-label": "Next version", disabled: at === sibs.length - 1 || !!turn,
        onclick: go(sibs[at + 1]) }, icon("chevron", "sm")));
  }

  function msgEl(m, i, v) {
    const last = i === v.path.length - 1;
    const switcher = switcherEl(m, v.idx);
    if (m.role === "user") {
      const bubble = m.content ? h("div", { class: "bubble" }, m.content) : null;
      return h("div", { class: `msg user${last ? " last" : ""}`, "data-id": m.id, "data-i": i },
        attachmentsEl(m.attachments), bubble,
        h("div", { class: "foot" }, switcher,
          h("div", { class: "actions" },
            h("button", { class: "btn ghost icon sm", title: "Edit and send again", "aria-label": "Edit", disabled: !!turn, onclick: () => editUser(m) }, icon("edit", "sm")),
            h("button", { class: "btn ghost icon sm", title: "Copy", "aria-label": "Copy", onclick: (e) => copyText(m.content, e.currentTarget) }, icon("copy", "sm")),
            h("button", { class: "btn ghost icon sm", title: "Delete this message and what follows", "aria-label": "Delete", disabled: !!turn, onclick: () => removeMsg(m) }, icon("trash", "sm")))));
    }
    const isLive = streaming && streaming.msg === m;
    if (isLive) { streaming.els = {}; streaming.shape = shapeOf(streaming); }
    const cont = v.idx.byId.get(m.parent)?.role === "assistant"; // the answer goes on after a tool call
    const body = h("div", { class: "body", style: { display: "grid", gap: "8px", minWidth: 0 } });
    if (!cont) body.append(h("div", { class: "who" }, m.model || "Assistant", m.t ? h("span", { class: "faint", style: { fontWeight: 400 } }, fmt.ago(m.t)) : null));
    if (m.reasoning || (isLive && streaming.thinking)) body.append(thinkEl(m, isLive));
    const hasCalls = (m.tool_calls || []).length > 0;
    if (m.content || isLive || !hasCalls) { // a step that only calls tools shows just its tool cards
      const md = h("div", { class: `md${isLive ? " streaming" : ""}` });
      if (isLive) {
        streamInto(md, streaming.rc, m.content, !streaming.thinking && !hasCalls);
        streaming.els.md = md;
      } else if (!m.content && !m.error) md.innerHTML = '<p class="faint">(no text)</p>';
      else md.innerHTML = renderMarkdown(m.content);
      body.append(md);
    }
    for (const call of m.tool_calls || []) body.append(toolCardEl(call, infoFor(call.name)));
    if (m.error) body.append(h("div", { class: "callout bad" }, icon("alert"), h("div", null, m.error)));
    if (m.stopped) body.append(h("div", { class: "small faint" }, "Stopped."));
    const stats = statsEl(m);
    if (stats) body.append(stats);
    if (!isLive) {
      body.append(h("div", { class: "foot" }, switcher, h("div", { class: "actions" },
        h("button", { class: "btn ghost icon sm", title: "Copy", "aria-label": "Copy", onclick: (e) => copyText(m.content, e.currentTarget) }, icon("copy", "sm")),
        last ? h("button", { class: "btn ghost icon sm", title: "Answer again (keeps this answer as another version)", "aria-label": "Regenerate",
          disabled: !!turn, onclick: () => regenerate() }, icon("refresh", "sm")) : null,
        h("button", { class: "btn ghost icon sm", title: "Delete this message and what follows", "aria-label": "Delete", disabled: !!turn, onclick: () => removeMsg(m) }, icon("trash", "sm")))));
    }
    return h("div", { class: `msg assistant${last ? " last" : ""}${cont ? " cont" : ""}`, "data-id": m.id, "data-i": i },
      h("div", { class: "avatar" }, cont ? null : mark(20)), body);
  }

  function thinkEl(m, isLive) {
    const secs = m.think_s ?? (isLive && streaming.thinkStart ? (performance.now() - streaming.thinkStart) / 1000 : null);
    const label = isLive && streaming.thinking ? "Thinking" : secs ? `Thought for ${fmt.dur(secs)}` : "Thoughts";
    const bodyEl = h("div", { class: "think-body md" });
    if (isLive) { streamInto(bodyEl, streaming.rr, m.reasoning || "", false); streaming.els.think = bodyEl; }
    else bodyEl.innerHTML = renderMarkdown(m.reasoning || "");
    const d = h("details", { class: `think${isLive && streaming.thinking ? " live" : ""}` },
      h("summary", null, icon("chevron", "sm chev"), icon("brain", "sm"), h("span", { class: "lbl" }, label)),
      bodyEl);
    if (isLive && streaming.thinking) d.open = true;
    else if (streaming?.openThink?.has(m)) d.open = true;
    d.addEventListener("toggle", () => {
      if (!streaming) return;
      if (d.open) streaming.openThink.add(m); else streaming.openThink.delete(m);
    });
    return d;
  }

  function statsEl(m) {
    const s = m.stats;
    if (!s || !s.predicted_n) return null;
    const parts = [
      h("span", { class: "hot" }, h("b", null, fmt.rate(s.decode)), " tok/s"),
      h("span", null, h("b", null, fmt.int(s.predicted_n)), " tokens"),
    ];
    if (s.prefill && s.prompt_n) parts.push(h("span", null, "prefill ", h("b", null, fmt.rate(s.prefill)), " tok/s"));
    if (s.accept !== null && s.accept !== undefined) parts.push(h("span", null, "MTP ", h("b", null, fmt.pct(s.accept))));
    if (s.ttft_ms) parts.push(h("span", null, "first token ", h("b", null, fmt.dur(s.ttft_ms / 1000))));
    if (s.total_s) parts.push(h("span", null, h("b", null, fmt.dur(s.total_s))));
    return h("div", { class: "stats", title: "Measured by the engine for this answer" }, parts);
  }

  function onMessagesClick(e) {
    const b = e.target.closest("[data-copy]");
    if (!b) return;
    const code = b.closest(".codeblock")?.querySelector("code");
    if (code) copyText(code.textContent, b);
  }

  function scrollBottom(force = false) {
    if (force || pinned) messages.scrollTop = messages.scrollHeight;
  }

  function repaint(node) { // one message again, for example a tool card that changed
    const v = view();
    const i = v.path.indexOf(node);
    const old = inner.querySelector(`.msg[data-id="${node.id}"]`);
    if (i < 0 || !old) { renderMessages(); return; }
    swapNode(old, msgEl(node, i, v));
    scrollBottom();
  }

  // ------------------------------------------------------------------------------------------- composer

  function autosize() {
    ta.style.height = "auto";
    ta.style.height = `${Math.min(260, ta.scrollHeight)}px`;
    renderComposer();
  }

  // How full the context window is: the engine's own count up to the last answer, an estimate for the rest.
  // The count is made once per change of the history (a turn), not per keystroke: the history part is kept under a key
  // made of cheap numbers (lengths), and the draft in the box is added to it. The estimate is additive over messages.
  function historyKey(nodes, c) {
    let n = 0;
    for (const m of nodes) {
      n += (m.content || "").length + (m.reasoning || "").length + (m.stats?.predicted_n || 0) + (m.attachments || []).length;
      for (const call of m.tool_calls || []) n += (call.arguments || "").length + (typeof call.result === "string" ? call.result.length : 0);
    }
    const tools = toolsActive() ? toolInfo.tools.map((t) => t.name).join(",") : "";
    return [chat.id, chat.leaf, nodes.length, n, (chat.system || c.system_prompt || "").length, tools].join("|");
  }

  function draftTokens(draft) {
    const key = attach.items.map((a) => `${a.kind}:${a.blob || a.name}:${a.text?.length || 0}`).join(",");
    if (ctxAtt.key !== key) {
      ctxAtt = { key, tokens: attach.items.reduce((sum, a) => sum + (a.kind === "text" ? estimateTokens(attachmentBlock(a)) : 600), 0) };
    }
    return draft || attach.items.length ? estimateTokens(draft) + ctxAtt.tokens + 4 : 0;
  }

  function contextUse() {
    const c = store.state?.settings?.chat || {};
    const nctx = caps.n_ctx || store.engine?.n_ctx;
    const nodes = pathOf(chat);
    const key = historyKey(nodes, c);
    if (ctxHistory.key !== key) {
      const images = new Map(usedImages(nodes).map((b) => [b, "x"]));
      const opts = { system: chat.system || c.system_prompt, settings: c, tools: toolsActive() ? toolInfo.tools.map((t) => t.schema) : [], images };
      let used = estimateRequest(buildRequest({ nodes, ...opts }));
      let exact = false;
      for (let k = nodes.length - 1; k >= 0; k -= 1) {
        const s = nodes[k].stats;
        if (s?.predicted_n) {
          used = used - estimateRequest(buildRequest({ nodes: nodes.slice(0, k + 1), ...opts })) + (s.cache_n || 0) + (s.prompt_n || 0) + (s.predicted_n || 0);
          exact = k === nodes.length - 1;
          break;
        }
      }
      ctxHistory = { key, used, exact };
    }
    const draft = ta.value.trim();
    const extra = draftTokens(draft);
    return { used: Math.max(0, Math.round(ctxHistory.used + extra)), nctx, exact: ctxHistory.exact && !extra };
  }

  function renderComposer() {
    const up = engineUp();
    const c = store.state?.settings?.chat || {};
    const busy = !!turn;
    sendBtn.classList.toggle("stop", busy);
    sendBtn.disabled = !busy && (!up || (!ta.value.trim() && !attach.items.length) || attach.busy());
    clear(sendBtn, icon(busy ? "stop" : "send"));
    sendBtn.title = busy ? "Stop (Esc)" : "Send (Enter)";
    ta.placeholder = up ? `Message ${engineModelName() || "the model"}` : "Load a model to start chatting";
    thinkChip.classList.toggle("on", c.thinking !== false);
    clear(thinkChip, icon("brain", "sm"), c.thinking !== false ? "Thinking on" : "Thinking off");
    clear(sampChip, icon("sliders", "sm"), c.sampling === "custom" ? `T ${c.temperature} · top-p ${c.top_p}` : "Model defaults");
    const count = toolInfo.tools.length;
    toolsChip.classList.toggle("on", toolsActive());
    clear(toolsChip, icon("wrench", "sm"), caps.tools === false ? "Tools: not in this model" : !toolInfo.enabled ? "Tools off" :
      toolInfo.connecting && !count ? "Tools: connecting…" : `Tools · ${count}`);
    attach.button.disabled = !up;

    const { used, nctx, exact } = contextUse();
    if (nctx && (used || exact)) {
      const share = used / nctx;
      ctxMeter.textContent = `${exact ? "" : "~"}${fmt.tokens(used)} / ${fmt.ctx(nctx)} tokens`;
      ctxMeter.title = exact ? "Context used by this conversation, of the engine's context window"
        : "An estimate (about 3.6 characters per token) of what the next request needs, of the engine's context window";
      ctxMeter.classList.toggle("warn", share >= 0.8 && share <= 1);
      ctxMeter.classList.toggle("bad", share > 1);
      ctxWarn.hidden = share < 0.8;
      if (share >= 0.8) {
        ctxWarn.className = `ctx-warn ${share > 1 ? "bad" : "warn"}`;
        ctxWarn.textContent = share > 1
          ? `This is about ${fmt.tokens(used)} tokens and the context window holds ${fmt.ctx(nctx)}. The engine will refuse it. Remove an attachment, start a new chat, or raise the context on the Engine page.`
          : `About ${fmt.tokens(used)} of ${fmt.ctx(nctx)} tokens are used. Long answers may not fit; consider a shorter attachment or a new chat.`;
      }
    } else {
      ctxMeter.textContent = nctx ? `${fmt.ctx(nctx)} context` : "";
      ctxMeter.classList.remove("warn", "bad");
      ctxWarn.hidden = true;
    }
    note.textContent = up ? "Runs on this computer. Answers can be wrong; check what matters." : "";
  }

  async function toggleThinking() {
    const cur = store.state?.settings?.chat?.thinking !== false;
    await saveChatSettings({ thinking: !cur });
  }

  async function saveChatSettings(patch) {
    try {
      const s = await post("/api/settings", { chat: patch });
      store.state.settings = s;
      emit("state");
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  async function saveToolSettings(patch) {
    try {
      store.state.settings = await post("/api/settings", { tools: patch });
      emit("state");
      await fetchTools();
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  function toolsMenu(anchor) {
    const s = store.state?.settings?.tools || {};
    const items = [{ head: toolInfo.enabled ? "Tools the model can call" : "Tools are off" }];
    if (caps.tools === false) items.push({ head: "The loaded model's chat template has no tool support" });
    for (const t of toolInfo.tools) {
      items.push({ label: t.name, hint: t.confirm ? "asks first" : "", icon: t.source === "mcp" ? "link" : "wrench", onclick: () => navigate("/settings") });
    }
    if (!toolInfo.tools.length && toolInfo.enabled) items.push({ head: "No tool is switched on" });
    for (const sv of toolInfo.servers || []) {
      if (sv.status === "error") items.push({ head: `${sv.name}: ${sv.error}`.slice(0, 120) });
    }
    items.push("sep");
    items.push({ label: s.enabled === false ? "Turn tools on" : "Turn tools off", icon: "power", onclick: () => saveToolSettings({ enabled: s.enabled === false }) });
    items.push({ label: "Choose tools and MCP servers", icon: "settings", onclick: () => navigate("/settings") });
    openMenu(anchor, items);
  }

  function samplingMenu(anchor) {
    const c = store.state?.settings?.chat || {};
    openMenu(anchor, [
      { head: "Sampling" },
      { label: "Model defaults", hint: c.sampling !== "custom" ? "in use" : "", icon: c.sampling !== "custom" ? "check" : "models",
        onclick: () => saveChatSettings({ sampling: "model" }) },
      { label: `My settings (T ${c.temperature}, top-p ${c.top_p}, top-k ${c.top_k})`, hint: c.sampling === "custom" ? "in use" : "",
        icon: c.sampling === "custom" ? "check" : "sliders", onclick: () => saveChatSettings({ sampling: "custom" }) },
      "sep",
      { label: "Change values and system prompt", icon: "settings", onclick: () => navigate("/settings") },
    ]);
  }

  // ------------------------------------------------------------------------------------------- sending

  function leafNode() {
    return indexOf(chat).byId.get(chat.leaf) || null;
  }

  async function submit() {
    const text = ta.value.trim();
    if ((!text && !attach.items.length) || turn) return;
    if (attach.busy()) { toast("Wait until the files are read"); return; }
    if (!engineUp()) { toast("Load a model first", "bad"); return; }
    const atts = attach.take();
    ta.value = "";
    autosize();
    const msg = { role: "user", content: text, t: nowSec() };
    if (atts.length) msg.attachments = atts;
    const user = addChild(chat, leafNode(), msg);
    const asst = addChild(chat, user, newAssistant());
    pinned = true;
    renderMessages();
    scrollBottom(true);
    save();
    await runTurn(asst);
  }

  // Ask again: the new answer is another version next to the old one.
  async function regenerate() {
    if (turn) return;
    if (!engineUp()) { toast("Load a model first", "bad"); return; }
    const v = view();
    let k = v.path.length - 1;
    while (k > 0 && v.path[k].role !== "user") k -= 1;
    const user = v.path[k];
    if (!user || user.role !== "user") return;
    const asst = addChild(chat, user, newAssistant());
    renderMessages();
    scrollBottom(true);
    await runTurn(asst);
  }

  // Edit your own message: the edited text becomes another version of it, with a fresh answer; the old branch stays.
  function editUser(node) {
    if (turn) return;
    const el = inner.querySelector(`.msg[data-id="${node.id}"]`);
    const box = h("textarea", { class: "textarea", rows: 3, value: node.content, "aria-label": "Edit your message" });
    const form = h("div", { class: "edit-box" }, box, h("div", { class: "row end" },
      h("button", { class: "btn ghost", onclick: () => renderMessages() }, "Cancel"),
      h("button", { class: "btn primary", "data-act": "send-edit", onclick: async () => {
        const text = box.value.trim();
        if (!text && !(node.attachments || []).length) return;
        if (!engineUp()) { toast("Load a model first", "bad"); return; }
        const parent = node.parent ? indexOf(chat).byId.get(node.parent) : null;
        const msg = { role: "user", content: text, t: nowSec() };
        if (node.attachments?.length) msg.attachments = node.attachments;
        const user = addChild(chat, parent, msg);
        const asst = addChild(chat, user, newAssistant());
        renderMessages();
        scrollBottom(true);
        save();
        await runTurn(asst);
      } }, "Send as new version")));
    el.replaceChildren(form);
    box.focus();
  }

  async function removeMsg(node) {
    if (turn) return;
    const idx = indexOf(chat);
    let below = -1;
    const stack = [node];
    while (stack.length) { below += 1; stack.push(...(idx.kids.get(stack.pop().id) || [])); }
    if (below > 0) {
      const ok = await confirmDialog({ title: "Delete this message?", ok: "Delete", danger: true,
        body: `It and the ${below} message${below === 1 ? "" : "s"} after it (in every version) will be removed from this chat.` });
      if (!ok) return;
    }
    removeBranch(chat, node);
    renderMessages();
    renderComposer();
    if (chat.id) await save();
  }

  // lastRound: no more tool calls. The tools stay in the request (same prompt, cache kept); tool_choice "none" stops the call.
  function requestFor(node, useTools, images, lastRound = false) {
    const c = store.state?.settings?.chat || {};
    const idx = indexOf(chat);
    const nodes = [];
    for (let cur = node.parent ? idx.byId.get(node.parent) : null; cur; cur = cur.parent ? idx.byId.get(cur.parent) : null) nodes.unshift(cur);
    return buildRequest({ nodes, system: chat.system || c.system_prompt, settings: c, noMoreTools: lastRound,
      tools: useTools ? toolInfo.tools.map((t) => t.schema) : [], images: caps.vision ? images : new Map() });
  }

  // One answer, which may take several steps: the model asks for tools, the tools run, the model goes on.
  async function runTurn(first) {
    const ctrl = new AbortController();
    turn = { ctrl };
    renderComposer();
    let node = first;
    try {
      for (let round = 0; ; round += 1) {
        const lastRound = round >= (toolInfo.max_rounds || 8);
        await generateInto(node, toolsActive(), ctrl, lastRound);
        if (lastRound) { delete node.tool_calls; break; } // the limit is reached; a call that came anyway does not run
        if (ctrl.signal.aborted || node.error || !(node.tool_calls || []).length) break;
        await save();
        await executeCalls(node, ctrl);
        if (ctrl.signal.aborted) break;
        node = addChild(chat, node, newAssistant());
        renderMessages();
        scrollBottom();
      }
    } finally {
      turn = null;
      streaming = null;
      renderMessages();
      renderComposer();
      scrollBottom();
      await save();
      ta.focus();
    }
  }

  async function generateInto(asst, useTools, ctrl, lastRound = false) {
    const images = await loadImages(pathOf(chat).filter((m) => m !== asst));
    streaming = { msg: asst, ctrl, t0: performance.now(), tFirst: 0, thinking: false, thinkStart: 0, raw: "", split: false,
      openThink: new Set(), frame: 0, lastPaint: 0, shape: "", els: {}, rc: createStreamRenderer(), rr: createStreamRenderer() };
    renderMessages();
    scrollBottom();
    renderComposer();
    try {
      await streamChat(requestFor(asst, useTools, images, lastRound), { signal: ctrl.signal, onChunk });
    } catch (e) {
      if (e.name === "AbortError") asst.stopped = true;
      else {
        asst.error = e.message;
        if (useTools && e.status === 400 && /tool|jinja/i.test(e.message)) {
          asst.error += " If this model cannot call tools, turn them off with the Tools button in the message box.";
        }
      }
    }
    const s = streaming;
    if (s.thinking) asst.think_s = (performance.now() - s.thinkStart) / 1000;
    const total = (performance.now() - s.t0) / 1000;
    if (asst.stats) {
      asst.stats.total_s = total;
      asst.stats.ttft_ms = s.tFirst ? s.tFirst - s.t0 : null;
    }
    const calls = (asst.tool_calls || []).filter((c) => c && c.name);
    for (const c of calls) { if (!c.id) c.id = `call_${newId()}`; c.status = "pending"; }
    if (calls.length) asst.tool_calls = calls; else delete asst.tool_calls;
    streaming = null;
    clearTimeout(s.frame);
    renderMessages();
    renderComposer();
    scrollBottom();
  }

  // Run the tools one after another. Calls that need approval wait for a click on their card.
  async function executeCalls(node, ctrl) {
    for (const call of node.tool_calls) {
      if (ctrl.signal.aborted) { call.status = "error"; call.result = call.result || "Stopped before it ran."; continue; }
      const info = infoFor(call.name);
      let args;
      try { args = call.arguments ? JSON.parse(call.arguments) : {}; } catch (e) {
        call.status = "error";
        call.result = `The arguments are not valid JSON (${e.message}). Call the tool again with a valid JSON object.`;
        continue;
      }
      if (!info) {
        call.status = "error";
        call.result = `There is no tool named "${call.name}". Available: ${toolInfo.tools.map((t) => t.name).join(", ") || "none"}.`;
        continue;
      }
      let approved = false;
      if (info.confirm) {
        call.status = "waiting";
        repaint(node);
        const answer = await waitForApproval(call.id, ctrl.signal);
        if (answer === "deny") {
          call.status = "denied";
          call.result = "The user declined to run this tool.";
          repaint(node);
          continue;
        }
        if (answer === "trust") await trustServer(info.server);
        approved = true;
      }
      call.status = "running";
      repaint(node);
      try {
        const r = await post("/api/tools/call", { name: call.name, arguments: args, approved });
        call.result = r.content;
        call.ms = r.ms;
        call.status = r.is_error ? "error" : "done";
        if (r.app) { // an MCP App shows this result (app-bridge.js); kept with the chat, small results only
          const big = JSON.stringify(r.app.result).length > 200000;
          call.app = { server: r.app.server, uri: r.app.uri, tool: r.app.tool, input: args,
            result: big ? { content: [{ type: "text", text: call.result }] } : r.app.result };
          showAppNow(call.id);
        }
      } catch (e) {
        call.status = "error";
        call.result = e.message;
      }
      repaint(node);
    }
  }

  async function trustServer(name) {
    const have = store.state?.settings?.tools?.mcp_trusted || [];
    await saveToolSettings({ mcp_trusted: [...new Set([...have, name])] });
    toast(`Tools of “${name}” run without asking from now on. Settings → Tools can take that back.`, "info", 7000);
  }

  function onChunk(chunk) {
    const s = streaming;
    if (!s) return;
    const m = s.msg;
    const delta = chunk.choices?.[0]?.delta || {};
    const now = performance.now();
    if (delta.reasoning_content) {
      s.split = true;
      if (!s.tFirst) s.tFirst = now;
      if (!s.thinking && !m.content) { s.thinking = true; s.thinkStart = now; }
      m.reasoning += delta.reasoning_content;
    }
    if (delta.content) {
      if (!s.tFirst) s.tFirst = now;
      s.raw += delta.content;
      if (s.split) {
        m.content = s.raw;
      } else {
        const t = s.raw.match(/^\s*<think>([\s\S]*?)(<\/think>\s*|$)([\s\S]*)$/);
        if (t) {
          m.reasoning = t[1];
          m.content = t[3];
          if (!t[2] && !s.thinking) { s.thinking = true; s.thinkStart = s.thinkStart || now; }
        } else {
          m.content = s.raw;
        }
      }
      if (s.thinking && m.content.trim()) {
        s.thinking = false;
        m.think_s = (now - s.thinkStart) / 1000;
      }
    }
    for (const tc of delta.tool_calls || []) { // tool calls arrive in pieces: the name once, the arguments bit by bit
      if (!s.tFirst) s.tFirst = now;
      if (s.thinking) { s.thinking = false; m.think_s = (now - s.thinkStart) / 1000; }
      const list = m.tool_calls || (m.tool_calls = []);
      const call = list[tc.index ?? 0] || (list[tc.index ?? 0] = { id: "", name: "", arguments: "", status: "streaming" });
      if (tc.id) call.id = tc.id;
      if (tc.function?.name && !call.name) call.name = tc.function.name;
      if (tc.function?.arguments) call.arguments += tc.function.arguments;
    }
    if (chunk.timings) {
      const t = chunk.timings;
      m.stats = {
        decode: t.predicted_per_second, predicted_n: t.predicted_n, prefill: t.prompt_per_second, prompt_n: t.prompt_n,
        cache_n: t.cache_n, accept: t.draft_n ? (t.draft_n_accepted || 0) / t.draft_n : null, draft_n: t.draft_n || 0,
      };
    }
    if (!s.frame) { // at most one drawing per PAINT_MS, however fast the chunks come
      const wait = Math.max(0, PAINT_MS - (performance.now() - s.lastPaint));
      s.frame = setTimeout(() => {
        if (streaming !== s) return;
        s.frame = 0;
        s.lastPaint = performance.now();
        paintStreaming();
      }, wait);
    }
  }

  // What the structure of the streaming message depends on. While it stays the same, only the text is added to the
  // two markdown bodies; when it changes (a thinking block appears, a tool call comes) the message is built again.
  function shapeOf(s) {
    const m = s.msg;
    if ((m.tool_calls || []).length) return ""; // a card with growing arguments: always built again
    return [!!m.reasoning, s.thinking, !!m.content, m.think_s ?? "", m.stats?.predicted_n ?? ""].join();
  }

  // Write `text` into a markdown body made of two parts: `.md-stable` (finished blocks, only ever added to) and
  // `.md-tail` (the open rest, written again). See createStreamRenderer in md.js.
  function streamInto(box, renderer, text, cursor) {
    const { tail } = renderer.update(text);
    box.replaceChildren(h("div", { class: "md-stable", html: renderer.html }), h("div", { class: "md-tail", html: tail + (cursor ? CURSOR : "") }));
  }

  function patchStream(box, renderer, text, cursor) {
    const { reset, fresh, tail } = renderer.update(text);
    const [stable, rest] = box.children;
    if (reset) stable.innerHTML = fresh;
    else if (fresh) stable.insertAdjacentHTML("beforeend", fresh);
    rest.innerHTML = tail + (cursor ? CURSOR : "");
  }

  function paintStreaming() {
    const s = streaming;
    const m = s.msg;
    const shape = shapeOf(s);
    if (shape && shape === s.shape && s.els.md?.isConnected && (!m.reasoning || s.els.think?.isConnected)) {
      patchStream(s.els.md, s.rc, m.content, !s.thinking);
      if (s.els.think) {
        patchStream(s.els.think, s.rr, m.reasoning, false);
        if (s.thinking) s.els.think.scrollTop = s.els.think.scrollHeight;
      }
      scrollBottom();
      return;
    }
    const v = view();
    const i = v.path.indexOf(m);
    const old = inner.querySelector(`.msg[data-id="${m.id}"]`);
    const fresh = msgEl(m, i, v);
    if (old) {
      const thinkBody = old.querySelector(".think-body");
      const keepScroll = thinkBody ? thinkBody.scrollTop : 0;
      swapNode(old, fresh);
      const nb = fresh.querySelector(".think-body");
      if (nb) nb.scrollTop = s.thinking ? nb.scrollHeight : keepScroll;
    } else inner.append(fresh);
    scrollBottom();
  }

  function stop() {
    turn?.ctrl.abort();
  }

  // ------------------------------------------------------------------------------------------- engine actions

  async function loadModel(id) {
    try {
      const st = await post("/api/engine/start", { model: id });
      store.engine = st;
      emit("engine");
      toast("Loading the model. This can take a few minutes for large files.");
    } catch (e) {
      toast(e.message, "bad", 8000);
      if (e.status === 400 || e.status === 409) navigate("/engine");
    }
  }

  async function startService() {
    try {
      store.engine = await post("/api/engine/service/start");
      emit("engine");
      toast("Starting the service. Loading a large model takes a few minutes.");
    } catch (e) {
      toast(e.message, "bad", 15000);
    }
  }

  async function ejectModel() {
    try {
      store.engine = await post("/api/engine/stop");
      emit("engine");
    } catch (e) { toast(e.message, "bad"); }
  }

  // ------------------------------------------------------------------------------------------- live panel

  function renderLive() {
    const hist = store.live.slice(-120);
    const lastS = hist[hist.length - 1] || {};
    const meta = store.liveMeta || {};
    const e = store.engine || {};
    const recent = (key, secs) => {
      const cut = (lastS.t || 0) - secs;
      for (let k = hist.length - 1; k >= 0; k -= 1) {
        if (hist[k].t < cut) break;
        if (hist[k][key] !== undefined && hist[k][key] !== null) return hist[k][key];
      }
      return null;
    };
    const lr = meta.last_response;
    const perAnswer = !!lastS.last_answer;  // a server without --metrics: the numbers are each finished answer's
    const decodeNow = perAnswer ? null : recent("decode", 3);
    const decode = decodeNow ?? lr?.predicted_per_second ?? null;
    const prefill = recent("prefill", 15) ?? lr?.prompt_per_second ?? null;
    let accept = recent("accept", 20);
    if (accept === null && meta.accept_logged !== null && meta.accept_logged !== undefined) accept = meta.accept_logged;
    if (accept === null && lr?.draft_n) accept = (lr.draft_n_accepted || 0) / lr.draft_n;
    const cacheHit = meta.cache?.hit_rate !== undefined && meta.cache?.hit_rate !== null ? meta.cache.hit_rate / 100 : null;
    const svc = e.service || null;
    const fle = (e.plan?.args || []).includes("--moe-expert-cache-inserts") || cacheHit !== null || !!svc?.expert_cache;

    const tiles = [];
    tiles.push(h("div", { class: "row between" }, h("h4", null, "Live"), stateBadge(e.state)));
    tiles.push(h("div", { class: "tile" },
      h("div", { class: "k" }, h("span", null, "Generation"), h("span", { class: "faint" }, decodeNow !== null ? "now" : lr ? "last answer" : "")),
      h("div", { class: "v" }, fmt.rate(decode), h("small", null, "tok/s")),
      sparkline(hist.map((x) => x.decode ?? null), { color: "var(--accent)", id: "dec" }),
      h("div", { class: "sub" }, perAnswer ? "Average of each finished answer, last two minutes" : "Tokens per second while answering, last two minutes")));
    tiles.push(h("div", { class: "tile" },
      h("div", { class: "k" }, h("span", null, "Prompt reading (prefill)")),
      h("div", { class: "v" }, fmt.rate(prefill), h("small", null, "tok/s")),
      sparkline(hist.map((x) => x.prefill ?? null), { color: "var(--violet)", id: "pre" })));
    tiles.push(h("div", { class: "tile" }, h("div", { class: "rings" },
      ring(accept, { color: "var(--cyan)", label: "MTP drafts accepted" }),
      ring(fle ? cacheHit : null, { color: "var(--fuchsia)", label: fle ? "Expert cache hits" : "Expert cache (FLE only)" }))));

    const gpus = lastS.gpus || [];
    if (gpus.length) {
      tiles.push(h("h4", null, "GPU"));
      for (const g of gpus) {
        tiles.push(h("div", { class: "tile gpu" },
          h("div", { class: "gpu-name" }, g.name, h("span", null, [g.temperature_c ? `${g.temperature_c} °C` : "", g.power_w ? `${Math.round(g.power_w)} W` : ""].filter(Boolean).join(" · "))),
          meter("Busy", g.utilization || 0, 100, "cyan", `${g.utilization ?? "–"} %`),
          meter("Memory", (g.memory_used || 0), (g.memory_total || 0), "violet")));
      }
    }
    tiles.push(h("h4", null, "Memory"));
    const memTile = h("div", { class: "tile gpu" });
    if (lastS.ram_total) memTile.append(meter("System RAM", lastS.ram_used, lastS.ram_total));
    if (lastS.engine_mem) memTile.append(h("div", { class: "row between small" }, h("span", null, "Engine process"), h("span", { class: "muted num" }, fmt.mem(lastS.engine_mem))));
    if (lastS.slots) memTile.append(h("div", { class: "row between small" }, h("span", null, "Slots busy"), h("span", { class: "muted num" }, `${lastS.slots_busy || 0} of ${lastS.slots}`)));
    if (meta.totals?.generated) memTile.append(h("div", { class: "row between small" }, h("span", null, "Generated since start"), h("span", { class: "muted num" }, `${fmt.int(meta.totals.generated)} tokens`)));
    tiles.push(memTile);
    if (meta.metrics_available === false) {
      tiles.push(h("div", { class: "callout" }, icon("info", "sm"), h("div", { class: "small" }, "This server runs without --metrics, so the numbers come from each finished answer.")));
    }
    if (svc?.expert_cache && !svc.cache_in_log && cacheHit === null) {
      tiles.push(h("div", { class: "callout" }, icon("info", "sm"), h("div", { class: "small" },
        `The expert-cache hit rate is a log line at level 4; ${svc.unit} logs up to level ${svc.log_level}. Add -lv 4 to its command to see it here.`)));
    }
    clear(live, tiles);
  }

  const resizeOnce = debounce(() => renderLive(), 150);
  window.addEventListener("resize", resizeOnce);

  return {
    unmount() {
      alive = false;
      unsub.forEach((u) => u());
      window.removeEventListener("resize", resizeOnce);
      if (turn) turn.ctrl.abort();
    },
  };
}
