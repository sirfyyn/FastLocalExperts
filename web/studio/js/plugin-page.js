// A page that a plugin adds (plugins.py): /p/<id> loads /plugins/<id>/<script> and calls its mount(main, kit).

import { store, topbar } from "./app.js";
import { h, icon } from "./ui.js";
import { makeKit } from "./plugin-kit.js";

export function mount(main, id) {
  const plugin = store.plugins.find((p) => p.id === id && p.loaded && p.page);
  if (!plugin) {
    main.replaceChildren(topbar("Plugin"), h("div", { class: "page" }, h("div", { class: "page-inner" },
      h("div", { class: "card" }, h("div", { class: "card-head" }, icon("info"),
        h("div", null, h("h3", null, `The plugin “${id}” is not on`),
          h("p", null, "Plugins are turned on one by one in ", h("a", { href: "/settings", "data-link": true }, "Settings"), ".")))))));
    return null;
  }
  let inner = null;
  let gone = false;
  main.replaceChildren(topbar(plugin.page.title), h("div", { class: "page" }, h("div", { class: "page-inner muted" }, "Loading …")));
  import(`/plugins/${plugin.id}/${plugin.page.script}`)
    .then((mod) => {
      if (gone) return;
      inner = mod.mount(main, makeKit(plugin)) || null;
    })
    .catch((e) => {
      if (gone) return;
      main.replaceChildren(topbar(plugin.page.title), h("div", { class: "page" }, h("div", { class: "page-inner" },
        h("p", { class: "bad" }, `The plugin's page did not load: ${e.message}`))));
    });
  return {
    unmount() {
      gone = true;
      if (inner?.unmount) inner.unmount();
    },
  };
}
