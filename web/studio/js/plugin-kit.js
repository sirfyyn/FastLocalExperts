// The kit for plugin pages (plugins.py, docs/PLUGINS.md). A plugin's page module gets it as the second argument of
// mount(main, kit) and should use only what is here: this list stays stable within one KIT_VERSION.

import { del, get, post, put } from "./api.js";
import { navigate, on, settingsSections, store, topbar } from "./app.js";
import { clear, confirmDialog, fmt, h, icon, toast } from "./ui.js";

export const KIT_VERSION = 1;

// Calls to the plugin's own routes under /api/p/<id>/…; they always go to this machine, never to a chosen host.
function pluginApi(id) {
  const base = `/api/p/${id}`;
  return {
    get: (p) => get(base + p),
    post: (p, b) => post(base + p, b),
    put: (p, b) => put(base + p, b),
    del: (p) => del(base + p),
  };
}

export function makeKit(plugin) {
  return Object.freeze({
    version: KIT_VERSION,
    id: plugin.id,
    api: pluginApi(plugin.id),
    asset: (name) => `/plugins/${plugin.id}/${name}`,
    h, clear, icon, fmt, toast, confirmDialog, topbar, navigate,
    theme: () => store.state?.settings?.theme || "system",
    on, // on("state", fn) and friends; returns the function that unsubscribes
    // For a boot script (ctx.boot): a card in Settings. build(settings, ui) returns an element; ui is {section, field,
    // toggle, save, render}: section(icon, title, sub, ...body) makes the card, save(patch) stores settings. order puts it
    // between the core sections (10 Appearance, 20 Models, 30 Chat, 40 Tools, 60 Hugging Face, 70 Access, 100 Plugins).
    settingsSection: (order, build) => { settingsSections.push({ id: plugin.id, order, build }); },
  });
}
