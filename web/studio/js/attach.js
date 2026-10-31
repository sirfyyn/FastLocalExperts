// Files attached to the message being written: choose, drop or paste them, upload them for reading
// (text, code, PDF text, pictures) and show them as chips above the text box.

import { hostUrl, upload } from "./api.js";
import { fmt, h, icon, toast } from "./ui.js";

const MB = 1024 * 1024;

export function createAttach({ getCaps, onChange }) {
  const items = []; // ready: the objects the server returned {kind, name, text | blob, tokens, ...}
  let uploading = 0;
  const chips = h("div", { class: "att-row" });
  const input = h("input", { type: "file", multiple: true, hidden: true, "aria-label": "Attach files" });
  const button = h("button", { class: "chip", type: "button", title: "Attach files: text, code, PDF or pictures (or drop them here)",
    onclick: () => input.click() }, icon("clip", "sm"), "Attach");
  input.addEventListener("change", () => { addFiles([...input.files]); input.value = ""; });

  function render() {
    chips.replaceChildren(...items.map((a, i) => chip(a, () => { items.splice(i, 1); render(); onChange(); })),
      ...(uploading ? [h("span", { class: "att-chip busy" }, h("span", { class: "spin" }), "Reading…")] : []));
    chips.hidden = !items.length && !uploading;
  }

  function chip(a, remove) {
    const meta = a.kind === "image" ? "picture" : `${fmt.int(a.chars)} characters · ~${fmt.tokens(a.tokens)} tokens`;
    return h("span", { class: `att-chip ${a.kind}`, title: a.truncated ? "Cut off: the file is longer than the limit for one attachment" : a.name },
      a.kind === "image" ? h("img", { src: hostUrl(`/api/attachments/${a.blob}`), alt: "" }) : icon("doc", "sm"),
      h("span", { class: "n" }, a.name), h("small", null, meta + (a.truncated ? " · cut off" : "")),
      h("button", { type: "button", class: "x", title: "Remove", "aria-label": `Remove ${a.name}`, onclick: remove }, icon("x", "sm")));
  }

  async function addFiles(files) {
    const caps = getCaps();
    const limit = (caps.max_upload_mb || 25) * MB;
    for (const file of files) {
      if (file.size > limit) {
        toast(`${file.name} is ${(file.size / MB).toFixed(1)} MB; the limit is ${caps.max_upload_mb} MB. You can raise it in Settings → Chat.`, "bad", 9000);
        continue;
      }
      if (/^image\//.test(file.type) && !caps.vision) {
        toast("The loaded model cannot look at pictures. Load a vision model (a GGUF with its mmproj file) to attach them.", "bad", 9000);
        continue;
      }
      uploading += 1;
      render();
      onChange();
      try {
        items.push(await upload(`/api/attachments?name=${encodeURIComponent(file.name)}`, file));
      } catch (e) {
        toast(`${file.name}: ${e.message}`, "bad", 9000);
      } finally {
        uploading -= 1;
        render();
        onChange();
      }
    }
  }

  return {
    chips, button, input,
    get items() { return items; },
    busy: () => uploading > 0,
    tokens: () => items.reduce((n, a) => n + (a.tokens || 0), 0),
    take() { const out = items.splice(0); render(); return out; },
    clear() { items.splice(0); render(); },
    addFiles,
    onDrop(e) {
      if (!e.dataTransfer?.files?.length) return;
      e.preventDefault();
      addFiles([...e.dataTransfer.files]);
    },
    onPaste(e) {
      const files = [...(e.clipboardData?.files || [])];
      if (!files.length) return;
      e.preventDefault();
      addFiles(files);
    },
  };
}

// The attachments of a sent message: pictures as thumbnails, text files folded, with a preview on click.
export function attachmentsEl(atts) {
  if (!atts?.length) return null;
  return h("div", { class: "att-list" }, atts.map((a) => {
    if (a.kind === "image") {
      return h("a", { class: "att-thumb", href: hostUrl(`/api/attachments/${a.blob}`), target: "_blank", rel: "noopener", title: a.name },
        h("img", { src: hostUrl(`/api/attachments/${a.blob}`), alt: a.name }));
    }
    return h("details", { class: "att-file" },
      h("summary", null, icon("doc", "sm"), h("span", { class: "n" }, a.name),
        h("small", null, `${fmt.int(a.chars)} characters · ~${fmt.tokens(a.tokens)} tokens${a.truncated ? " · cut off" : ""}`)),
      h("pre", null, a.text.length > 6000 ? `${a.text.slice(0, 6000)}\n… (${fmt.int(a.text.length - 6000)} more characters)` : a.text));
  }));
}
