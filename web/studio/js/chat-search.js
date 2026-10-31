// Search in the sidebar: looks through the titles and the text of every chat (all branches, attachment names and text)
// on the server and replaces the chat list with the hits while there is something in the box.

import { get } from "./api.js";
import { clear, debounce, h } from "./ui.js";

let active = false;
export const searchActive = () => active;

function highlight(text, needle) {
  const at = text.toLowerCase().indexOf(needle.toLowerCase());
  if (at < 0) return [text];
  return [text.slice(0, at), h("mark", null, text.slice(at, at + needle.length)), text.slice(at + needle.length)];
}

export function initChatSearch({ onClear }) {
  const input = document.getElementById("chat-search");
  const list = document.getElementById("chat-list");
  if (!input) return;

  async function run() {
    const q = input.value.trim();
    if (!q) {
      active = false;
      onClear();
      return;
    }
    active = true;
    let results = [];
    try {
      results = (await get(`/api/chats/search?q=${encodeURIComponent(q)}`)).results;
    } catch {
      results = [];
    }
    if (input.value.trim() !== q) return; // typed on meanwhile
    const current = (location.pathname.match(/^\/c\/([0-9a-f]{16})$/) || [])[1];
    if (!results.length) {
      clear(list, h("div", { class: "chat-empty" }, `No chat mentions “${q}”.`));
      return;
    }
    clear(list, results.map((c) => h("a", { class: `chat-item hit${c.id === current ? " active" : ""}`, href: `/c/${c.id}`, "data-link": true, title: c.title },
      h("span", { class: "t" }, highlight(c.title, q)),
      c.snippet ? h("span", { class: "snip" }, highlight(c.snippet, q)) : null)));
  }

  input.addEventListener("input", debounce(run, 180));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && input.value) { input.value = ""; run(); }
  });
}
