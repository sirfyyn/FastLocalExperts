// Hosts: the switcher "This machine / <host> / Add host…" at the top of the sidebar, the notice when the host cannot be
// reached, the dialogs to add and pair a host, and the Settings cards (the hosts of this Studio, and the clients that
// are paired with it). The page itself needs nothing else: api.js sends every request to the chosen host.

import { activeHostId, get, post } from "./api.js";
import { clear, confirmDialog, copyButton, h, icon, openMenu, toast } from "./ui.js";

let hosts = [];
let sshOk = true;
let current = null;
let banner = null;
let problem = null;
const listeners = new Set();

const CHECK_EVERY_MS = 10000;

// ----------------------------------------------------------------------------------------------- state

function remember(id, name) {
  try {
    if (id) { localStorage.setItem("fle-host", id); localStorage.setItem("fle-host-name", name || ""); }
    else { localStorage.removeItem("fle-host"); localStorage.removeItem("fle-host-name"); }
  } catch { /* storage blocked: the choice lasts until the next reload only */ }
}

export function switchTo(id) {
  const host = hosts.find((x) => x.id === id);
  remember(id, host?.name || "");
  location.href = "/";
}

async function load() {
  const data = await get("/api/hosts");
  hosts = data.hosts;
  sshOk = data.ssh;
  current = hosts.find((x) => x.id === activeHostId()) || null;
  listeners.forEach((fn) => fn());
  return data;
}

// ----------------------------------------------------------------------------------------------- the switcher

function paintBadge() {
  const dot = document.getElementById("host-dot");
  const name = document.getElementById("host-name");
  if (!dot || !name) return;
  dot.style.background = current ? current.color : "var(--good)";
  name.textContent = current ? current.name : "This machine";
  document.body.classList.toggle("host-remote", !!current);
  if (current) {
    document.documentElement.style.setProperty("--host", current.color);
    document.title = `${current.name} · FLE Studio`;
  } else {
    document.documentElement.style.removeProperty("--host");
  }
}

function openSwitcher(anchor) {
  const items = [{ head: "Compute on" }];
  items.push({ label: "This machine", icon: current ? undefined : "check", onclick: () => current && switchTo("") });
  for (const x of hosts) {
    items.push({ label: x.name, hint: x.kind === "ssh" ? "ssh" : "", icon: current?.id === x.id ? "check" : undefined,
      onclick: () => current?.id !== x.id && switchTo(x.id) });
  }
  items.push("sep", { label: "Add host…", icon: "plus", onclick: () => openAdd() });
  openMenu(anchor, items);
}

// ----------------------------------------------------------------------------------------------- notice

function hideBanner() {
  problem = null;
  banner?.remove();
  banner = null;
}

function showProblem(detail) {
  if (!current) return;
  problem = detail;
  const code = detail.code;
  const title = code === "fingerprint_changed" ? `The certificate of “${current.name}” has changed`
    : code === "unpaired" ? `“${current.name}” no longer accepts this computer`
      : `Host “${current.name}” is not reachable`;
  const lines = [];
  if (code === "fingerprint_changed") {
    lines.push("It may be a new installation of the host, or somebody in between. Nothing was sent.",
      h("div", { class: "mono small" }, `Pinned: ${detail.pinned}`), h("div", { class: "mono small" }, `Now:    ${detail.seen}`));
  } else if (code === "unpaired") {
    lines.push("The access may have been revoked on the host. Pair again with a new code.");
  } else {
    lines.push(detail.message || "No answer.");
    if (current.mirror) lines.push("Chats are shown from the copy on this computer (read only).");
  }
  const reconnect = h("button", { class: "btn sm", onclick: async (e) => {
    e.target.disabled = true;
    try {
      const r = await post(`/api/hosts/${current.id}/connect`, {});
      if (r.ok) { toast("Connected", "good", 1800); location.reload(); return; }
      showProblem({ code: r.code || "unreachable", message: r.message, seen: r.seen, pinned: r.pinned });
    } catch (err) { toast(err.message, "bad"); }
    e.target.disabled = false;
  } }, icon("refresh", "sm"), "Reconnect");
  const repair = h("button", { class: "btn sm primary", onclick: () => openRepair(current) }, "Pair again…");
  const local = h("button", { class: "btn sm ghost", onclick: () => switchTo("") }, "Use this machine");
  banner?.remove();
  banner = h("div", { class: `host-banner ${code === "fingerprint_changed" ? "warn" : ""}`, id: "host-banner", role: "alert" },
    icon("alert"), h("div", { class: "grow" }, h("strong", null, title), ...lines.map((l) => h("div", { class: "small" }, l))),
    code === "unreachable" ? reconnect : repair, local);
  document.body.append(banner);
}

async function healthCheck() {
  if (!current) return;
  try {
    const r = await post(`/api/hosts/${current.id}/check`, {});
    if (r.ok) {
      if (problem) { hideBanner(); location.reload(); }
    } else showProblem({ code: r.code || "unreachable", message: r.message, seen: r.seen, pinned: r.pinned });
  } catch { /* this Studio itself is away: api.js reports that */ }
}

// ----------------------------------------------------------------------------------------------- dialogs

function modal(title, ...body) {
  const close = () => { back.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  const back = h("div", { class: "modal-back", onclick: (e) => { if (e.target === back) close(); } },
    h("div", { class: "modal host-modal", role: "dialog", "aria-modal": "true", "aria-label": title }, h("h3", null, title), ...body));
  document.addEventListener("keydown", onKey);
  document.body.append(back);
  return { close, back };
}

function field(label, control, hint = "") {
  return h("label", { class: "field" }, h("span", null, label), control, hint ? h("span", { class: "hint" }, hint) : null);
}

function openAdd() {
  let kind = "direct";
  const name = h("input", { class: "input", id: "host-name-in", placeholder: "Titan", autocomplete: "off" });
  const address = h("input", { class: "input mono", id: "host-address", placeholder: "titan:8642", autocomplete: "off", spellcheck: "false" });
  const target = h("input", { class: "input mono", id: "host-ssh-target", placeholder: "user@titan", autocomplete: "off", spellcheck: "false" });
  const sshPort = h("input", { class: "input mono", id: "host-ssh-port", value: "22", inputmode: "numeric" });
  const remotePort = h("input", { class: "input mono", id: "host-remote-port", value: "8642", inputmode: "numeric" });
  const code = h("input", { class: "input mono", id: "host-code", placeholder: "ABCD-EFGH", autocomplete: "off", spellcheck: "false",
    style: { textTransform: "uppercase" } });
  const fp = h("input", { class: "input mono", id: "host-fingerprint", placeholder: "AA:BB:CC:…", autocomplete: "off", spellcheck: "false" });
  const mirror = h("input", { type: "checkbox", id: "host-mirror" });
  const error = h("div", { class: "host-error small", role: "alert" });
  const directBox = h("div", { class: "stack" }, field("Address of the host", address,
    "Name or IP address, and the port if it is not 8642. The host must run with --host."));
  const sshBox = h("div", { class: "stack" }, field("SSH target", target,
    sshOk ? "FLE Studio starts ssh itself, with your keys. No passwords." : "The ssh program was not found on this computer."),
  h("div", { class: "row" }, field("SSH port", sshPort), field("Studio port on the host", remotePort)));
  const seg = h("div", { class: "seg", role: "radiogroup" });
  const set = (k) => {
    kind = k;
    directBox.hidden = k !== "direct";
    sshBox.hidden = k !== "ssh";
    for (const b of seg.children) { b.classList.toggle("on", b.dataset.kind === k); b.setAttribute("aria-checked", String(b.dataset.kind === k)); }
  };
  for (const [k, label] of [["direct", "Address + code"], ["ssh", "SSH"]]) {
    seg.append(h("button", { type: "button", role: "radio", "data-kind": k, onclick: () => set(k) }, label));
  }
  const submit = h("button", { class: "btn primary", id: "host-add-go", type: "submit" }, "Add host");
  const dlg = modal("Add host",
    h("p", { class: "small muted" }, "On the host: Settings → Remote access → New code, or run ",
      h("code", null, "fle-studio --pair-code"), " there."),
    seg,
    h("form", { class: "stack", onsubmit: async (e) => {
      e.preventDefault();
      error.textContent = "";
      submit.disabled = true;
      const body = { kind, name: name.value.trim(), code: code.value.trim(), mirror: mirror.checked, fingerprint: fp.value.trim() };
      if (kind === "direct") body.address = address.value.trim();
      else Object.assign(body, { target: target.value.trim(), port: sshPort.value.trim(), remote_port: remotePort.value.trim() });
      try {
        const r = await post("/api/hosts", body);
        dlg.close();
        toast(`${r.host.name} added`, "good", 2000);
        switchTo(r.host.id);
      } catch (err) {
        error.textContent = err.message;
        submit.disabled = false;
      }
    } },
    directBox, sshBox, field("Name (optional)", name), field("Pairing code", code),
    h("details", null, h("summary", { class: "small" }, "Check the certificate fingerprint (optional)"),
      field("Fingerprint shown on the host", fp, "If you type it in, it must match what the host shows now.")),
    h("label", { class: "row small" }, mirror, h("span", null, "Keep a read-only copy of this host's chats here, to read them offline")),
    error,
    h("div", { class: "row end" }, h("button", { class: "btn ghost", type: "button", onclick: () => dlg.close() }, "Cancel"), submit)));
  set("direct");
  address.focus();
}

function openRepair(host) {
  const code = h("input", { class: "input mono", id: "repair-code", placeholder: "ABCD-EFGH", autocomplete: "off",
    style: { textTransform: "uppercase" } });
  const fp = h("input", { class: "input mono", id: "repair-fingerprint", placeholder: "AA:BB:CC:… (optional)", autocomplete: "off" });
  const error = h("div", { class: "host-error small", role: "alert" });
  const go = h("button", { class: "btn primary", type: "submit" }, "Pair again");
  const dlg = modal(`Pair again with ${host.name}`,
    h("p", { class: "small muted" }, "Make a new code on the host (Settings → Remote access). It also pins the certificate "
      + "the host shows now; compare it with the fingerprint on the host if you want to be sure."),
    h("form", { class: "stack", onsubmit: async (e) => {
      e.preventDefault();
      go.disabled = true;
      error.textContent = "";
      try {
        await post(`/api/hosts/${host.id}/repair`, { code: code.value.trim(), fingerprint: fp.value.trim() });
        dlg.close();
        toast("Paired again", "good", 1800);
        location.reload();
      } catch (err) { error.textContent = err.message; go.disabled = false; }
    } },
    field("Pairing code", code), field("Fingerprint (optional)", fp), error,
    h("div", { class: "row end" }, h("button", { class: "btn ghost", type: "button", onclick: () => dlg.close() }, "Cancel"), go)));
  code.focus();
}

// ----------------------------------------------------------------------------------------------- start

export async function initHosts() {
  if (!document.getElementById("hosts-css")) {
    document.head.append(h("link", { id: "hosts-css", rel: "stylesheet", href: "/hosts.css" }));
  }
  try {
    await load();
  } catch { return; }
  if (activeHostId() && !current) { remember("", ""); location.href = "/"; return; }
  const anchor = document.getElementById("new-chat");
  if (anchor && !document.getElementById("host-switch")) {
    const btn = h("button", { class: "host-switch", id: "host-switch", type: "button", "aria-haspopup": "menu",
      title: "Choose the machine that computes", onclick: () => openSwitcher(btn) },
    h("span", { class: "host-dot", id: "host-dot" }), h("span", { class: "grow host-name", id: "host-name" }), icon("down", "sm"));
    anchor.before(btn);
  }
  paintBadge();
  listeners.add(paintBadge);
  window.addEventListener("fle-host-problem", (e) => showProblem(e.detail));
  if (current) setInterval(healthCheck, CHECK_EVERY_MS);
}

// ----------------------------------------------------------------------------------------------- Settings

function ago(ts) {
  if (!ts) return "never";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return "just now";
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 129600) return `${Math.round(s / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

// `section(icon, title, sub, ...body)` and `toggle(checked, onFlip)` come from the Settings page.
export function remoteSections({ section, toggle }) {
  return [hostsCard({ section, toggle }), accessCard({ section, toggle })];
}

function hostsCard({ section, toggle }) {
  const list = h("div", { class: "stack", id: "hosts-list" });
  const node = section("link", "Hosts", "Other machines that compute for this one: pick one at the top of the sidebar.", list,
    h("div", { class: "row" }, h("button", { class: "btn", id: "hosts-add", onclick: () => openAdd() }, icon("plus", "sm"), "Add host…")));
  const paint = () => {
    if (!node.isConnected && node.dataset.seen) { listeners.delete(paint); return; }  // the page was drawn again
    node.dataset.seen = node.isConnected ? "1" : "";
    if (!hosts.length) { clear(list, h("span", { class: "small faint" }, "No hosts yet. This machine does the work.")); return; }
    clear(list, hosts.map((x) => {
      const status = h("span", { class: "small muted" });
      const check = async () => {
        status.textContent = "checking …";
        try {
          const r = await post(`/api/hosts/${x.id}/check`, {});
          status.textContent = r.ok ? `reachable${r.admin ? ", admin" : ""}` : r.message;
          status.className = `small ${r.ok ? "good-text" : "bad-text"}`;
        } catch (e) { status.textContent = e.message; }
      };
      const name = h("input", { class: "input", value: x.name, "aria-label": "Name", onchange: async (e) => {
        try { await post(`/api/hosts/${x.id}/update`, { name: e.target.value }); await load(); } catch (err) { toast(err.message, "bad"); }
      } });
      return h("div", { class: "host-row", "data-host": x.id },
        h("span", { class: "host-dot", style: { background: x.color } }),
        h("div", { class: "grow stack tight" },
          h("div", { class: "row wrap" }, name,
            h("span", { class: "pill info" }, x.kind === "ssh" ? "ssh" : x.scheme === "http" ? "http" : "tls"),
            current?.id === x.id ? h("span", { class: "pill good" }, "in use") : null),
          h("div", { class: "small muted mono" }, x.kind === "ssh" ? `ssh ${x.ssh.target} → port ${x.ssh.remote_port}` : x.address),
          x.fingerprint ? h("div", { class: "small faint mono", title: "SHA-256 of the certificate, pinned when you paired" },
            x.fingerprint_pretty) : null,
          h("label", { class: "row small" }, toggle(x.mirror, async (on) => {
            try { await post(`/api/hosts/${x.id}/update`, { mirror: on }); await load(); } catch (err) { toast(err.message, "bad"); }
          }), h("span", null, "Keep a read-only copy of its chats here")),
          status),
        h("div", { class: "row wrap end" },
          h("button", { class: "btn sm", onclick: check }, "Check"),
          current?.id === x.id ? null : h("button", { class: "btn sm", onclick: () => switchTo(x.id) }, "Use"),
          h("button", { class: "btn sm ghost", onclick: () => openRepair(x) }, "Pair again…"),
          h("button", { class: "btn sm danger", onclick: async () => {
            if (!(await confirmDialog({ title: `Remove ${x.name}?`, body: "The pairing is forgotten here. On the host you can also revoke this computer.",
              ok: "Remove", danger: true }))) return;
            if (current?.id === x.id) remember("", "");
            await post(`/api/hosts/${x.id}/remove`, {});
            if (current?.id === x.id) { location.href = "/"; return; }
            await load();
          } }, "Remove")));
    }));
  };
  listeners.add(paint);
  paint();
  if (!hosts.length) load().catch(() => {});
  return node;
}

function accessCard({ section, toggle }) {
  const body = h("div", { class: "stack", id: "remote-access" });
  const node = section("panel", "Remote access", "Let other FLE Studios use this machine's engine.", body);
  if (activeHostId()) {
    clear(body, h("span", { class: "small muted" },
      "You are looking at a host. Paired computers are managed on the host itself, not through another Studio."));
    return node;
  }
  // The Settings page is drawn again whenever the state changes; what was fetched is kept for a few seconds.
  let timerEl = null;
  const cache = accessCache;
  let info = cache.info;
  let made = cache.made;

  const paint = () => {
    if (!info) { clear(body, h("span", { class: "small faint" }, "Loading …")); return; }
    const listen = info.listening;
    const local = ["127.0.0.1", "localhost", "::1"].includes(listen.host);
    const rows = [];
    if (local) {
      rows.push(h("div", { class: "callout" }, "This Studio listens on this computer only. To let other computers connect, start it "
        + "with --host 0.0.0.0 (or the address of your network or tailnet). An SSH host needs no change here."));
    } else {
      rows.push(h("div", { class: "small muted" }, `Listening on ${listen.host}:${listen.port}, ${listen.tls ? "with TLS" : "without TLS (browsers only)"}.`));
    }
    if (info.fingerprint) {
      rows.push(h("div", { class: "stack tight" }, h("span", { class: "label" }, "Certificate fingerprint (SHA-256)"),
        h("div", { class: "row" }, h("code", { class: "mono small fp", id: "host-fp" }, info.fingerprint_pretty), copyButton(info.fingerprint_pretty))));
    }
    if (info.addresses.length && !local) {
      rows.push(h("div", { class: "small muted" }, "Addresses to enter on the client: ", h("span", { class: "mono" }, info.addresses.join(", "))));
    }
    const codeBox = h("div", { class: "stack tight", id: "pair-code-box" });
    if (made) {
      timerEl = h("div", { class: "small muted", id: "pair-code-left" }, countdown(made.expires));
      codeBox.append(h("div", { class: "pair-code mono", id: "pair-code" }, made.pretty), timerEl);
    }
    rows.push(h("div", { class: "row wrap" },
      h("button", { class: "btn", id: "pair-new", onclick: async () => {
        try { made = cache.made = await post("/api/remote/code", {}); await refresh(); } catch (e) { toast(e.message, "bad"); }
      } }, "New pairing code"), codeBox));
    const clients = info.clients;
    rows.push(h("div", { class: "stack", id: "client-list" }, h("span", { class: "label" }, "Paired computers"),
      clients.length ? clients.map((c) => h("div", { class: "host-row", "data-client": c.id },
        h("div", { class: "grow stack tight" }, h("div", null, h("strong", null, c.name)),
          h("div", { class: "small muted" }, `last seen ${ago(c.last_seen)}${c.last_address ? ` from ${c.last_address}` : ""}`)),
        h("label", { class: "row small", title: "Admin may also change settings, install engines and load or delete models" },
          toggle(c.admin, async (on) => {
            try { await post(`/api/remote/clients/${c.id}/admin`, { admin: on }); await refresh(); } catch (e) { toast(e.message, "bad"); }
          }), h("span", null, "Admin")),
        h("button", { class: "btn sm danger", onclick: async () => {
          if (!(await confirmDialog({ title: `Revoke ${c.name}?`, body: "It cannot use this machine any more until it is paired again.",
            ok: "Revoke", danger: true }))) return;
          try { await post(`/api/remote/clients/${c.id}/revoke`, {}); await refresh(); } catch (e) { toast(e.message, "bad"); }
        } }, "Revoke")))
        : h("span", { class: "small faint" }, "None yet."),
      h("span", { class: "small faint" }, "A paired computer may chat, look and measure. Admin adds settings, engine installs and "
        + "updates, and loading or deleting models; installs and updates still ask for a confirmation on that computer.")));
    clear(body, rows);
  };

  async function refresh() {
    try { info = await get("/api/remote"); } catch (e) { clear(body, h("span", { class: "small muted" }, e.message)); return; }
    cache.info = info;
    cache.at = Date.now();
    paint();
  }
  paint();
  if (!info || Date.now() - cache.at > 15000) refresh();
  const ticker = setInterval(() => {
    if (!node.isConnected) { clearInterval(ticker); return; }
    if (timerEl && made) timerEl.textContent = countdown(made.expires);
  }, 1000);
  return node;
}

const accessCache = { info: null, at: 0, made: null };

function countdown(expires) {
  const left = Math.max(0, Math.round(expires - Date.now() / 1000));
  return left > 0 ? `Valid for ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")} more minutes, one use.` : "Expired.";
}
