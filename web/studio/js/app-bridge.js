// MCP Apps in the chat (extension "MCP Apps", SEP-1865, stable 2026-01-26).
//
// A tool that points at a ui:// resource gets its result shown by that page. The page is HTML from the MCP server, so
// it is never trusted: it runs in <iframe sandbox="allow-scripts"> (no allow-same-origin, so its origin is opaque and
// it cannot reach Studio's pages, cookies or storage), served by Studio with a Content-Security-Policy built from the
// resource's metadata. All it can do is send messages to this file, which answers the ones listed below and refuses
// the rest. Messages are JSON-RPC 2.0 over postMessage.
//
//   app → host   ui/initialize (request), ui/notifications/initialized, ui/notifications/size-changed,
//                tools/call (only the app's own server, only tools meant for apps; asks like a model's call),
//                ui/message (asks first), ui/open-link (asks first, http/https only), ui/request-display-mode
//                (always "inline"), ui/update-model-context (kept, not sent to the model), notifications/message,
//                ping
//   host → app   ui/notifications/tool-input, ui/notifications/tool-result, ui/notifications/host-context-changed,
//                ui/resource-teardown (request, when the user hides the app)
//
// Not offered: fullscreen and picture-in-picture, resources/read from the app, permissions (camera, microphone, …),
// streaming of partial tool input.

import { hostUrl, post } from "./api.js";
import { confirmDialog, h, icon, toast } from "./ui.js";

const PROTOCOL = "2026-01-26";
const MIN_HEIGHT = 60;
const MAX_HEIGHT = 800;
const MAX_TEXT = 8000;

// Set by the chat page: what the bridge needs from it.
export const host = {
  say: async () => { throw new Error("The chat is not open"); }, // (text) → put a user message into the conversation
  trusted: () => false, // (server) → true when its tools run without asking
};

// Running apps by tool-call id. The chat rebuilds its message elements often (a new answer, a changed card), and an
// iframe that is taken out of the page and put back loads again, so the app would start over and lose what the user
// did in it. Where the browser has Node.moveBefore (Chrome 133+, Firefox 144+), a running panel is moved from the old
// element into the new one and keeps running; elsewhere the app loads again, which is harmless but visible.
const running = new Map();
const canMove = () => typeof Element.prototype.moveBefore === "function";

export function liveSlot(callId) {
  const panel = running.get(callId);
  if (!panel || !panel.isConnected || !canMove()) return null;
  return h("div", { class: "app-slot", "data-app": callId });
}

function adopt(root) { // put the running panels into the slots of freshly built elements
  for (const slot of root.querySelectorAll?.(".app-slot[data-app]") || []) {
    const panel = running.get(slot.dataset.app);
    if (panel && slot.parentNode) {
      try { slot.parentNode.moveBefore(panel, slot); slot.remove(); } catch { slot.replaceWith(panel); }
    } else slot.remove();
  }
}

export function swapNode(oldEl, newEl) {
  oldEl.after(newEl);
  adopt(newEl);
  oldEl.remove();
}

export function replaceMessages(container, nodes) {
  const old = [...container.childNodes];
  container.append(...nodes);
  adopt(container);
  for (const n of old) n.remove();
}

export function effectiveTheme() {
  const set = document.documentElement.getAttribute("data-theme");
  if (set === "light" || set === "dark") return set;
  return matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function cssVariables() {
  const css = getComputedStyle(document.documentElement);
  const pick = (name) => css.getPropertyValue(name).trim();
  return {
    "--color-background-primary": pick("--bg"),
    "--color-background-secondary": pick("--soft"),
    "--color-text-primary": pick("--text"),
    "--color-text-secondary": pick("--muted"),
    "--color-border-primary": pick("--line-2"),
    "--color-accent": pick("--accent"),
    "--font-sans": pick("--font"),
    "--font-mono": pick("--mono"),
  };
}

const clip = (text, n = 300) => (text.length > n ? `${text.slice(0, n)}…` : text);

// The panel under a tool card: loads the frame, runs the bridge. `call` is the chat's tool call record with .app.
export function appPanel(call) {
  const status = h("div", { class: "app-status small faint" }, icon("link", "sm"), " Loading the app…");
  const panel = h("div", { class: "app-panel", "data-app": call.id }, status);
  running.set(call.id, panel);
  start(call, panel, status).catch((e) => {
    status.className = "app-status small bad-text";
    status.textContent = `The app could not be shown: ${e.message}`;
  });
  return panel;
}

async function start(call, panel, status) {
  const prep = await post("/api/apps/prepare", { name: call.name });
  const frame = h("iframe", {
    class: "mcp-app", src: hostUrl(prep.url), sandbox: "allow-scripts", title: `${call.name} (MCP app)`,
    referrerpolicy: "no-referrer", loading: "eager", style: { height: "120px" },
  });
  if (prep.prefers_border === false) frame.classList.add("borderless");
  const hide = h("button", { class: "btn ghost sm", title: "Close the app (the tool's result stays in the card)" }, "Hide app");
  status.replaceWith(h("div", { class: "app-head row between small faint" },
    h("span", null, icon("link", "sm"), ` App from the MCP server “${prep.server}”`), hide));
  panel.append(frame);

  let ready = false;
  let everConnected = false;
  let dead = false;
  const win = () => frame.contentWindow;
  const reply = (id, result) => win()?.postMessage({ jsonrpc: "2.0", id, result }, "*");
  const fail = (id, message, code = -32000) => win()?.postMessage({ jsonrpc: "2.0", id, error: { code, message } }, "*");
  const notify = (method, params) => win()?.postMessage({ jsonrpc: "2.0", method, params }, "*");
  const context = () => ({
    theme: effectiveTheme(),
    styles: { variables: cssVariables() },
    displayMode: "inline",
    availableDisplayModes: ["inline"],
    containerDimensions: { maxHeight: MAX_HEIGHT, width: Math.round(panel.clientWidth || 600) },
    locale: navigator.language || "en-US",
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    platform: "web",
    toolInfo: { id: call.id, tool: { name: call.app.tool, inputSchema: { type: "object" } } },
  });

  async function onRequest(m) {
    const p = m.params || {};
    switch (m.method) {
      case "ui/initialize":
        return reply(m.id, {
          protocolVersion: PROTOCOL,
          hostInfo: { name: "fle-studio", version: document.getElementById("brand-ver")?.textContent?.replace(/^v/, "") || "" },
          hostCapabilities: { serverTools: {}, message: { text: {} }, openLinks: {}, updateModelContext: { text: {} }, logging: {} },
          hostContext: context(),
        });
      case "ping":
        return reply(m.id, {});
      case "ui/request-display-mode":
        return reply(m.id, { mode: "inline" });
      case "ui/update-model-context":
        call.app.context = { content: p.content, structuredContent: p.structuredContent }; // kept; the model does not see it
        return reply(m.id, {});
      case "tools/call": {
        const name = typeof p.name === "string" ? p.name : "";
        if (!name) return fail(m.id, "A tool name is needed", -32602);
        const args = p.arguments && typeof p.arguments === "object" && !Array.isArray(p.arguments) ? p.arguments : {};
        let approved = false;
        if (!host.trusted(prep.server)) {
          approved = await confirmDialog({
            title: "Let the app run a tool?",
            body: `The app from “${prep.server}” wants to run “${name}” with ${clip(JSON.stringify(args))}`,
            ok: "Allow once",
          });
          if (!approved) return fail(m.id, "The user declined to run this tool");
        }
        try {
          const r = await post("/api/apps/call", { frame: prep.frame, name, arguments: args, approved });
          return reply(m.id, r.result);
        } catch (e) { return fail(m.id, e.message); }
      }
      case "ui/message": {
        const text = (Array.isArray(p.content) ? p.content : []).filter((c) => c?.type === "text").map((c) => String(c.text || "")).join("\n").trim();
        if (p.role !== "user" || !text) return fail(m.id, "Only a user message with text is supported", -32602);
        if (text.length > MAX_TEXT) return fail(m.id, "The message is too long", -32602);
        const ok = await confirmDialog({ title: "Add a message to the chat?",
          body: `The app from “${prep.server}” wants to say, as you: “${clip(text, 500)}”`, ok: "Send" });
        if (!ok) return fail(m.id, "The user declined");
        try { await host.say(text); return reply(m.id, {}); } catch (e) { return fail(m.id, e.message); }
      }
      case "ui/open-link": {
        let url;
        try { url = new URL(String(p.url || "")); } catch { return fail(m.id, "Not a web address", -32602); }
        if (url.protocol !== "https:" && url.protocol !== "http:") return fail(m.id, "Only http and https links can be opened", -32602);
        const ok = await confirmDialog({ title: "Open a link?", body: `The app from “${prep.server}” wants to open ${clip(url.href, 400)}`, ok: "Open" });
        if (!ok) return fail(m.id, "The user declined");
        window.open(url.href, "_blank", "noopener,noreferrer");
        return reply(m.id, {});
      }
      default:
        return fail(m.id, `FLE Studio does not offer ${m.method}`, -32601);
    }
  }

  function onNotification(m) {
    const p = m.params || {};
    if (m.method === "ui/notifications/initialized") {
      ready = true; // nothing is sent to the app before this
      notify("ui/notifications/tool-input", { arguments: call.app.input || {} });
      notify("ui/notifications/tool-result", call.app.result || {});
    } else if (m.method === "ui/notifications/size-changed") {
      const height = Number(p.height);
      if (Number.isFinite(height)) frame.style.height = `${Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, Math.ceil(height) + 2))}px`;
    } else if (m.method === "notifications/message") {
      console.debug("[mcp app]", p.level, p.data);
    }
  }

  function onMessage(e) {
    if (dead) return;
    if (everConnected && !frame.isConnected) { dispose(); return; }
    if (e.source !== win() || !e.data || e.data.jsonrpc !== "2.0") return; // only this frame, only JSON-RPC
    const m = e.data;
    if (typeof m.method !== "string") return; // answers to our requests need no handling
    if (m.id !== undefined && m.id !== null) onRequest(m).catch((err) => fail(m.id, err.message));
    else onNotification(m);
  }

  const dark = matchMedia("(prefers-color-scheme: light)");
  const sendContext = () => { if (ready && frame.isConnected) notify("ui/notifications/host-context-changed", context()); };
  const watcher = new MutationObserver(sendContext);
  watcher.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  dark.addEventListener("change", sendContext);
  window.addEventListener("message", onMessage);
  const poll = setInterval(() => { if (frame.isConnected) everConnected = true; else if (everConnected) dispose(); }, 1500);
  frame.addEventListener("load", () => { everConnected = true; });
  everConnected = frame.isConnected;

  function dispose() {
    if (dead) return;
    dead = true;
    if (running.get(call.id) === panel) running.delete(call.id);
    window.removeEventListener("message", onMessage);
    dark.removeEventListener("change", sendContext);
    watcher.disconnect();
    clearInterval(poll);
  }

  hide.addEventListener("click", async () => {
    // ui/resource-teardown is a request: give the app a moment to answer, then take it down either way.
    const id = `teardown-${Date.now()}`;
    win()?.postMessage({ jsonrpc: "2.0", id, method: "ui/resource-teardown", params: { reason: "The user closed the app" } }, "*");
    await new Promise((r) => setTimeout(r, 300));
    dispose();
    panel.remove();
    call.app.hidden = true;
    toast("App hidden. Open the tool card to see its result as text.", "info", 3000);
  });
}
