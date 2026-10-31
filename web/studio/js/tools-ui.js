// Tool cards in the chat: what the model called, with which arguments, what came back, and the Allow / Decline
// buttons for calls that need the user's approval. Everything shown is set as text, never as HTML.

import { appPanel, liveSlot } from "./app-bridge.js";
import { fmt, h, icon } from "./ui.js";

const approvals = new Map(); // call id → function that answers the waiting loop: "allow" | "deny" | "trust"
const opened = new Set(); // call ids whose card the user opened
const liveApps = new Set(); // call ids whose app runs now; apps of older chats start with a click

export function waitForApproval(callId, signal) {
  return new Promise((resolve) => {
    const done = (answer) => { approvals.delete(callId); resolve(answer); };
    approvals.set(callId, done);
    signal?.addEventListener("abort", () => done("deny"), { once: true });
    if (signal?.aborted) done("deny");
  });
}

export const showAppNow = (callId) => liveApps.add(callId);

export const answerApproval = (callId, answer) => approvals.get(callId)?.(answer);

function pretty(text) {
  if (!text) return "{}";
  try { return JSON.stringify(JSON.parse(text), null, 2); } catch { return text; }
}

function oneLine(text) {
  try { return JSON.stringify(JSON.parse(text || "{}")); } catch { return String(text || ""); }
}

const STATUS = {
  streaming: ["info", "Calling…"],
  waiting: ["warn", "Waiting for you"],
  running: ["info", "Running…"],
  done: ["good", "Done"],
  error: ["bad", "Error"],
  denied: ["", "Declined"],
};

export function toolCardEl(call, info) {
  const [cls, label] = STATUS[call.status] || STATUS.done;
  const waiting = call.status === "waiting";
  const d = h("details", { class: `toolcard ${call.status || "done"}`, "data-call": call.id });
  d.open = waiting || opened.has(call.id);
  d.addEventListener("toggle", () => { if (d.open) opened.add(call.id); else opened.delete(call.id); });
  const took = call.ms !== undefined && call.status === "done" ? ` · ${fmt.dur(call.ms / 1000)}` : "";
  d.append(h("summary", null, icon("chevron", "sm chev"), icon("wrench", "sm"),
    h("span", { class: "tname" }, call.name || "…"),
    h("span", { class: "targs" }, oneLine(call.arguments)),
    h("span", { class: `pill ${cls}` }, label + took)));
  const body = h("div", { class: "tbody" });
  if (waiting) {
    const where = info?.server ? `the tool of the MCP server “${info.server}”` : "this tool";
    body.append(h("div", { class: "approve" },
      h("span", null, `Let ${where} run with these arguments?`),
      h("div", { class: "row wrap" },
        h("button", { class: "btn primary sm", onclick: () => answerApproval(call.id, "allow") }, "Allow once"),
        h("button", { class: "btn sm", onclick: () => answerApproval(call.id, "deny") }, "Decline"),
        info?.server ? h("button", { class: "btn ghost sm", title: "Settings → Tools lists the servers you allowed, and can take this back",
          onclick: () => answerApproval(call.id, "trust") }, `Always allow “${info.server}”`) : null)));
  }
  body.append(h("div", { class: "tlabel" }, "Arguments"), h("pre", null, pretty(call.arguments)));
  if (call.result) body.append(h("div", { class: "tlabel" }, call.status === "error" ? "Error" : "Result"), h("pre", null, call.result));
  d.append(body);
  return call.app && !waiting ? withApp(d, call) : d;
}

// An MCP App shows the result under the card, in a sandboxed frame (app-bridge.js). Apps of earlier sessions start
// with a click: showing one runs the server's page again, not the tool.
function withApp(card, call) {
  const wrap = h("div", { class: "toolwrap" }, card);
  if (liveApps.has(call.id) && !call.app.hidden) wrap.append(liveSlot(call.id) || appPanel(call));
  else {
    wrap.append(h("button", { class: "btn sm app-open", onclick: (e) => {
      liveApps.add(call.id);
      call.app.hidden = false;
      e.currentTarget.replaceWith(appPanel(call));
    } }, icon("link", "sm"), `Show the app of “${call.app.server}”`));
  }
  return wrap;
}
