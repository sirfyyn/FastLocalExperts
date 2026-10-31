// One short sentence for every flag and environment variable of the engine command (the words come from the server,
// flaghelp.py). The command itself stays as it is; this only explains it.

import { get, post } from "./api.js";
import { h } from "./ui.js";

let table = null;
let loading = null;

export function loadFlagTable() {
  if (table) return Promise.resolve(table);
  loading ||= get("/api/flags").then((t) => { table = t; return t; }).catch(() => { loading = null; return null; });
  return loading;
}

// The sentence for one token of a command ("-t", "--jinja", "FLE_MOE_CACHE_BUDGET_MIB=1"), or "" when none is known
// (or the table has not arrived yet).
export function flagTitle(token) {
  if (!table) return "";
  const env = token.match(/^([A-Z][A-Z0-9_]*)=/);
  if (env) return table.env[env[1]] || (env[1].startsWith("FLE_") ? table.fle_env_generic : "");
  const name = token.startsWith("--") ? token.split("=")[0] : token;
  return table.flags[name]?.text || "";
}

// A list under the command: every flag and variable with its value and what it does. `open` shows it unfolded.
export function explainBlock(plan, { open = false } = {}) {
  const box = h("details", { class: "explain", open });
  box.append(h("summary", null, "What each part of the command does"));
  const body = h("div", { class: "explain-list small faint" }, "Loading…");
  box.append(body);
  post("/api/flags/explain", { args: plan.args || [], env: plan.env || {} }).then((r) => {
    const rows = r.items.map((it) => h("div", { class: `explain-row${it.known ? "" : " unknown"}` },
      h("code", { class: it.kind === "env" ? "env" : "flag" }, it.name),
      it.value ? h("code", { class: "val" }, it.value) : h("span"),
      h("span", { class: "txt" }, it.known ? it.text : "No description for this one; it is passed to the engine as it is.")));
    body.replaceChildren(...rows);
    body.classList.remove("small", "faint");
  }).catch((e) => { body.textContent = e.message; });
  return box;
}
