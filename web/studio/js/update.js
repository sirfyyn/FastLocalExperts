// Updates: a quiet card in the sidebar when a newer FLE Studio (or FLE engine) exists, the details with the release
// notes, the one-click update where FLE Studio can do it, and the Settings section. The server asks GitHub at most
// once a day; this page only shows what the server knows.

import { get, post } from "./api.js";
import { clear, copyButton, h, icon, toast } from "./ui.js";

let view = null;
let card = null;
let modal = null;
let poll = null;
const listeners = new Set();

const POLL_IDLE_MS = 5 * 60 * 1000;

// ----------------------------------------------------------------------------------------------- state

async function refresh() {
  try {
    view = await get("/api/update");
  } catch {
    return view;
  }
  changed();
  return view;
}

function changed() {
  renderCard();
  if (modal) modal.render();
  listeners.forEach((fn) => fn(view));
  const busy = view && (view.run.state === "running" || view.run.state === "restarting");
  clearTimeout(poll);
  poll = setTimeout(refresh, busy ? 1500 : POLL_IDLE_MS);
}

function visible(v) {
  if (!v || !v.enabled) return { app: false, engines: false, run: false, result: false };
  return {
    app: !!(v.app && !v.app.dismissed),
    engines: v.engines.some((e) => !e.dismissed),
    run: v.run.state !== "idle",
    result: !!v.result,
  };
}

// ----------------------------------------------------------------------------------------------- notes as text

// The release text comes from the network. It is shown as text only: headings, bullets, `code`; nothing becomes HTML.
function inline(text) {
  const out = [];
  text.split(/(`[^`]+`)/).forEach((part) => {
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) out.push(h("code", null, part.slice(1, -1)));
    else if (part) out.push(part.replace(/\*\*/g, "").replace(/\[([^\]]*)\]\([^)]*\)/g, "$1"));
  });
  return out;
}

export function renderNotes(text) {
  const box = h("div", { class: "upd-notes" });
  let list = null;
  for (const raw of String(text || "").split("\n")) {
    const line = raw.trimEnd();
    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
    if (bullet) {
      if (!list) { list = h("ul"); box.append(list); }
      list.append(h("li", null, inline(bullet[1])));
      continue;
    }
    list = null;
    const head = /^#{1,6}\s+(.*)$/.exec(line);
    if (head) box.append(h("h4", null, inline(head[1])));
    else if (line.trim()) box.append(h("p", null, inline(line.trim())));
  }
  if (!box.childNodes.length) box.append(h("p", { class: "muted" }, "No release notes."));
  return box;
}

// ----------------------------------------------------------------------------------------------- the card

function cardText(v) {
  const run = v.run;
  if (run.state === "running") return ["Updating FLE Studio …", run.tag || ""];
  if (run.state === "restarting") return ["Restarting …", "The page reloads by itself"];
  if (run.state === "failed") return ["Update did not work", "The old version keeps running"];
  if (v.result) return [v.result.status === "failed" ? "Update did not work" : "FLE Studio was updated", v.result.tag || ""];
  const show = visible(v);
  if (show.app) return [`FLE Studio ${v.app.version} is available`, `You have ${v.current}`];
  const e = v.engines.find((x) => !x.dismissed);
  return [`${e.label} update`, `${e.installed} → ${e.latest}`];
}

function renderCard() {
  if (!card) return;
  const show = view ? visible(view) : null;
  const any = show && (show.app || show.engines || show.run || show.result);
  card.hidden = !any;
  if (!any) { card.replaceChildren(); return; }
  const [title, sub] = cardText(view);
  const bad = view.run.state === "failed" || view.result?.status === "failed";
  card.className = `update-card${bad ? " bad" : ""}`;
  clear(card, h("span", { class: "upd-dot", "aria-hidden": "true" }),
    h("div", { class: "grow" }, h("div", { class: "upd-title" }, title), h("div", { class: "small muted" }, sub)));
}

// ----------------------------------------------------------------------------------------------- details

async function dismissApp() {
  try {
    view = await post("/api/update/dismiss", { version: view.app.version });
    toast("We will not mention this version again", "info", 2400);
  } catch (e) {
    toast(e.message, "bad");
  }
  closeModal();
  changed();
}

async function dismissEngines() {
  try {
    view = await post("/api/update/dismiss", { engines: true });
  } catch (e) {
    toast(e.message, "bad");
  }
  closeModal();
  changed();
}

async function applyNow() {
  try {
    view = await post("/api/update/apply", { tag: view.app.tag });
  } catch (e) {
    toast(e.message, "bad", 8000);
    return;
  }
  changed();
}

async function updateEngine(e, btn) {
  btn.disabled = true;
  try {
    await post("/api/engines/install", { variant: e.variant });
    toast(`Downloading ${e.label}. Progress is on the Engine page.`, "good", 6000);
  } catch (err) {
    toast(err.message, "bad", 8000);
    btn.disabled = false;
  }
}

function closeModal() {
  if (!modal) return;
  document.removeEventListener("keydown", modal.onKey);
  modal.back.remove();
  modal = null;
}

function openDetails() {
  if (modal) return;
  const body = h("div", { class: "stack" });
  const back = h("div", { class: "modal-back", onclick: (e) => { if (e.target === back) closeModal(); } },
    h("div", { class: "modal upd-modal", role: "dialog", "aria-modal": "true", "aria-label": "Updates" }, body));
  const onKey = (e) => { if (e.key === "Escape") closeModal(); };
  document.addEventListener("keydown", onKey);
  document.body.append(back);
  modal = { back, onKey, render: () => clear(body, ...details()) };
  modal.render();
}

function commandBlock(command) {
  return h("div", { class: "upd-cmd" }, h("code", { class: "mono" }, command), copyButton(command, "Copy the command"));
}

function details() {
  const v = view;
  const parts = [];
  const run = v.run;
  if (run.state === "running" || run.state === "restarting") {
    parts.push(h("h3", null, run.state === "running" ? "Updating FLE Studio" : "Restarting"),
      h("p", null, run.state === "running" ? `Installing ${run.tag}. FLE Studio restarts when it is done.`
        : "The new version is installed. This page reloads in a moment."),
      h("div", { class: "upd-progress", role: "progressbar", "aria-label": "Updating" }, h("i")));
    return parts;
  }
  if (run.state === "failed") {
    parts.push(h("h3", null, "The update did not work"), h("p", null, run.message || "The old version keeps running."));
    if (run.tail?.length) parts.push(h("pre", { class: "upd-log mono" }, run.tail.join("\n")));
    if (run.log) parts.push(h("p", { class: "small faint" }, "Full log: ", h("span", { class: "mono" }, run.log)));
    parts.push(h("div", { class: "row end" }, h("button", { class: "btn primary", onclick: async () => {
      try { view = await post("/api/update/ack"); } catch { /* shown again by the next refresh */ }
      closeModal();
      changed();
    } }, "OK")));
    return parts;
  }
  if (v.result) {
    const bad = v.result.status === "failed";
    parts.push(h("h3", null, bad ? "The update did not work" : "FLE Studio was updated"),
      h("p", null, bad ? "The old version keeps running." : `Now running ${v.current}.`));
    if (bad && v.result.log) parts.push(h("p", { class: "small faint" }, "Log: ", h("span", { class: "mono" }, v.result.log)));
    parts.push(h("div", { class: "row end" }, h("button", { class: "btn primary", onclick: async () => {
      try { view = await post("/api/update/ack"); } catch { /* ignore */ }
      closeModal();
      changed();
    } }, "OK")));
    return parts;
  }
  const show = visible(v);
  if (v.app && (show.app || !show.engines)) {
    const a = v.app;
    parts.push(h("div", null, h("h3", null, `FLE Studio ${a.version} is available`),
      h("p", null, `You have ${v.current}.${a.prerelease ? " This is a pre-release." : ""}`)),
      renderNotes(a.notes),
      h("a", { class: "small", href: a.url, target: "_blank", rel: "noopener" }, "Release page on GitHub ", icon("external", "sm")));
    if (a.can_auto) {
      parts.push(h("p", { class: "small muted" }, "FLE Studio installs the new version and starts again with the same settings. "
        + "If it does not work, the old version keeps running."),
      h("div", { class: "row end wrap" },
        h("button", { class: "btn ghost", onclick: dismissApp }, "Later"),
        h("button", { class: "btn", onclick: closeModal }, "Close"),
        h("button", { class: "btn primary", onclick: applyNow }, icon("download", "sm"), "Update now")));
    } else {
      parts.push(h("p", { class: "small muted" }, a.reason || "Update it with this command:"), a.command ? commandBlock(a.command) : null,
        h("div", { class: "row end wrap" },
          h("button", { class: "btn ghost", onclick: dismissApp }, "Later"),
          h("button", { class: "btn primary", onclick: closeModal }, "Close")));
    }
  }
  const engines = v.engines.filter((e) => !e.dismissed);
  if (engines.length) {
    parts.push(h("div", { class: "upd-engines" }, h("h3", null, "FLE engine"),
      engines.map((e) => {
        const btn = h("button", { class: "btn sm primary" }, "Update engine");
        btn.addEventListener("click", () => updateEngine(e, btn));
        return h("div", { class: "row upd-engine" }, h("div", { class: "grow" }, h("div", null, e.label),
          h("div", { class: "small muted" }, `${e.installed} → ${e.latest}`),
          h("a", { class: "small", href: e.url, target: "_blank", rel: "noopener" }, "Release page ", icon("external", "sm"))), btn);
      }),
      h("div", { class: "row end wrap" },
        h("button", { class: "btn ghost", onclick: dismissEngines }, "Later"),
        !v.app || v.app.dismissed ? h("button", { class: "btn", onclick: closeModal }, "Close") : null)));
  }
  return parts;
}

// ----------------------------------------------------------------------------------------------- restart

// While FLE Studio restarts, the page waits until the server answers again, then loads the new version.
function watchRestart() {
  let down = false;
  const started = Date.now();
  const tick = async () => {
    try {
      const res = await fetch("/api/ping", { cache: "no-store" });
      if (res.ok && (down || Date.now() - started > 6000)) { location.reload(); return; }
    } catch {
      down = true;
    }
    if (Date.now() - started < 180000) setTimeout(tick, 1000);
  };
  setTimeout(tick, 1000);
}

let watching = false;
listeners.add((v) => {
  if (v && v.run.state === "restarting" && !watching) { watching = true; watchRestart(); }
});

// ----------------------------------------------------------------------------------------------- start

export function initUpdates() {
  if (!document.getElementById("update-css")) {
    document.head.append(h("link", { id: "update-css", rel: "stylesheet", href: "/update.css" }));
  }
  const engineCard = document.getElementById("engine-card");
  if (!engineCard || card) return;
  card = h("div", { class: "update-card", id: "update-card", role: "button", tabindex: "0", hidden: true });
  card.addEventListener("click", openDetails);
  card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openDetails(); } });
  engineCard.before(card);
  refresh().then(() => {
    if (view?.result || view?.run.state === "failed") openDetails();
  });
}

// ----------------------------------------------------------------------------------------------- Settings

function when(ts) {
  if (!ts) return "never";
  const d = new Date(ts * 1000);
  return d.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

// `section(icon, title, sub, ...body)`, `toggle(checked, onFlip)` and `save(patch, quiet)` come from the Settings page.
export function updatesSection({ section, toggle, save }) {
  const info = h("div", { class: "stack" });

  async function check() {
    try {
      view = await post("/api/update/check");
      if (view.error) toast(view.error, "info", 4000);
      else if (view.app) toast(`FLE Studio ${view.app.version} is available`, "good", 3000);
      else toast("FLE Studio is up to date", "good", 2400);
    } catch (e) {
      toast(e.message, "bad", 6000);
    }
    changed();
  }

  async function flip(patch, thenCheck) {
    if (await save({ updates: patch }, true)) {
      await refresh();
      if (thenCheck) await check();
    }
  }

  function paint() {
    const v = view;
    if (!v) { clear(info, h("span", { class: "small faint" }, "Loading …")); return; }
    const where = { "uv-tool": "uv tool", pip: "pip", source: "a source folder" }[v.install.kind] || v.install.kind;
    const status = !v.enabled ? "Checking is off." : v.checked ? `Last checked: ${when(v.checked)}.` : "Not checked yet.";
    const found = v.enabled && v.app ? `FLE Studio ${v.app.version} is available.` : v.enabled && v.checked && !v.error ? "FLE Studio is up to date." : "";
    clear(info,
      h("label", { class: "row" }, toggle(v.enabled, (on) => flip({ check: on }, false)),
        h("span", null, "Check for updates (about once a day)")),
      h("label", { class: "row" }, toggle(v.prereleases, (on) => flip({ prereleases: on }, true)),
        h("span", null, "Include pre-releases")),
      h("div", { class: "row wrap" },
        h("button", { class: "btn", id: "update-check-now", disabled: !v.enabled, onclick: check }, icon("refresh", "sm"), "Check now"),
        h("span", { class: "small muted" }, [status, found].filter(Boolean).join(" "))),
      v.error && v.enabled ? h("span", { class: "small faint" }, `Could not check: ${v.error}`) : null,
      h("span", { class: "small faint" }, `Installed with ${where}${v.install.service ? " (as a service)" : ""}. `
        + "The check asks api.github.com for the latest release and sends nothing about you or this machine."));
  }

  const node = section("download", "Updates", "FLE Studio tells you when a new version is out.", info);
  const sync = () => paint();
  listeners.add(sync);
  paint();
  if (!view) refresh();
  // The section is rebuilt whenever Settings redraws; drop the old listener when its node is gone.
  const drop = new MutationObserver(() => { if (!document.body.contains(node)) { listeners.delete(sync); drop.disconnect(); } });
  drop.observe(document.body, { childList: true, subtree: true });
  return node;
}
