// Settings: looks, where models live, how the chat samples, the Hugging Face token and access from other machines.

import { post } from "./api.js";
import { applyTheme, on, refreshState, store, topbar } from "./app.js";
import { toolsSection, stopToolsPolling } from "./settings-tools.js";
import { remoteSections } from "./hosts.js";
import { clear, fmt, h, icon, toast } from "./ui.js";
import { updatesSection } from "./update.js";

export function mount(main) {
  const unsub = [];
  const page = h("div", { class: "page-inner", style: { maxWidth: "860px" } });
  main.replaceChildren(topbar("Settings"), h("div", { class: "page" }, page));

  async function save(patch, quiet = false) {
    try {
      store.state.settings = await post("/api/settings", patch);
      applyTheme();
      if (!quiet) toast("Saved", "good", 1600);
      return true;
    } catch (e) {
      toast(e.message, "bad", 7000);
      return false;
    }
  }

  function render(force = false) {
    const s = store.state?.settings;
    if (!s) return;
    if (!force && page.contains(document.activeElement) && /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) return;
    clear(page,
      h("div", { class: "page-head" }, h("div", null, h("h2", null, "Settings"),
        h("p", null, `Stored in ${store.state.data_dir}. Start settings for the engine live on the Engine page.`))),
      look(s), models(s), chat(s), toolsSection(s, { section, field, toggle, save, render: () => render(true) }),
      hub(s), network(s), updatesSection({ section, toggle, save }), ...remoteSections({ section, toggle }), about());
  }

  // ------------------------------------------------------------------------------------------- sections

  function section(iconName, title, sub, ...body) {
    return h("div", { class: "card" }, h("div", { class: "card-head" }, icon(iconName), h("div", null, h("h3", null, title), sub ? h("p", null, sub) : null)),
      h("div", { class: "stack" }, ...body));
  }

  function look(s) {
    return section("spark", "Appearance", "",
      seg([["system", "Like the system"], ["dark", "Dark"], ["light", "Light"]], s.theme, (v) => save({ theme: v }, true)));
  }

  function models(s) {
    const dir = h("input", { class: "input mono", value: s.models_dir });
    return section("folder", "Models folder", "Where downloads go and where FLE Studio looks for GGUF files.",
      h("div", { class: "row" }, dir, h("button", { class: "btn", onclick: async () => { if (await save({ models_dir: dir.value.trim() })) refreshState(); } }, "Save")),
      h("span", { class: "small faint" }, `Changing it moves nothing; models already in the new folder show up in the library. ${fmt.bytes(store.state.system?.disk_free)} free there now.`));
  }

  function chat(s) {
    const c = s.chat;
    const custom = c.sampling === "custom";
    const num = (key, step, min, max) => h("input", { class: "input num", type: "number", step, min, max, value: c[key], disabled: !custom,
      onchange: (e) => save({ chat: { [key]: Number(e.target.value) } }, true) });
    return section("chat", "Chat", "Defaults for new answers. A chat can still turn thinking on or off from the message box.",
      h("label", { class: "field" }, h("span", null, "Sampling"),
        seg([["model", "The model's recommended values"], ["custom", "My own values"]], c.sampling, async (v) => { await save({ chat: { sampling: v } }, true); render(); })),
      h("div", { class: "grid three" },
        field("Temperature", num("temperature", 0.05, 0, 2)),
        field("Top P", num("top_p", 0.01, 0, 1)),
        field("Top K", num("top_k", 1, 0, 1000))),
      h("div", { class: "grid three" },
        field("Min P", num("min_p", 0.01, 0, 1)),
        field("Presence penalty", num("presence_penalty", 0.1, -2, 2)),
        field("Answer length limit", h("input", { class: "input num", type: "number", min: 0, step: 256, value: c.max_tokens,
          onchange: (e) => save({ chat: { max_tokens: Math.max(0, Math.round(Number(e.target.value) || 0)) } }, true) }), "0 = no limit")),
      h("label", { class: "row" }, toggle(c.thinking, (v) => save({ chat: { thinking: v } }, true)),
        h("span", null, "Let the model think before it answers ", h("span", { class: "muted small" }, "(for models that can)"))),
      field("System prompt", h("textarea", { class: "textarea", rows: 3, value: c.system_prompt,
        placeholder: "Optional. Sent at the start of every new chat.", onchange: (e) => save({ chat: { system_prompt: e.target.value } }, true) })),
      field("Largest upload", h("input", { class: "input num", type: "number", min: 1, max: 512, step: 1, value: c.max_upload_mb,
        "aria-label": "Largest upload in MB", onchange: (e) => save({ chat: { max_upload_mb: Math.max(1, Math.min(512, Math.round(Number(e.target.value) || 25))) } }, true) }),
      "MB per attached file (1 to 512). Larger files are refused with a message that names this limit."));
  }

  function hub(s) {
    const token = h("input", { class: "input mono", type: "password", autocomplete: "off",
      placeholder: s.has_hf_token ? "A token is saved" : "hf_…" });
    return section("models", "Hugging Face", "Only needed for gated or private repos. Public models download without it.",
      h("div", { class: "row" }, token,
        h("button", { class: "btn", onclick: async () => { if (token.value.trim() && await save({ hf_token: token.value.trim() })) render(); } }, "Save"),
        s.has_hf_token ? h("button", { class: "btn ghost danger", onclick: async () => { if (await save({ hf_token: "" })) render(); } }, "Remove") : null),
      h("span", { class: "small faint" }, "Kept in settings.json on this computer and only sent to huggingface.co. FLE Studio never shows it again."));
  }

  function network(s) {
    const srv = s.server;
    const lan = srv.host !== "127.0.0.1" && srv.host !== "localhost" && srv.host !== "::1";
    const key = h("input", { class: "input mono", type: "password", autocomplete: "off",
      placeholder: s.has_access_key ? "A key is saved" : "Choose a key for other machines" });
    const hosts = h("input", { class: "input mono", value: srv.allowed_hosts, placeholder: "workstation, studio.home.arpa" });
    const port = h("input", { class: "input num", type: "number", min: 1, max: 65535, value: srv.port });
    return section("link", "Access", "Who can open FLE Studio. Changes here apply the next time FLE Studio starts.",
      seg([["127.0.0.1", "This computer only"], ["0.0.0.0", "Other machines too"]], lan ? "0.0.0.0" : "127.0.0.1",
        async (v) => { await save({ server: { host: v } }, true); render(); }),
      lan ? h("div", { class: "callout warn" }, icon("alert", "sm"), h("div", null,
        "Anyone on your network who knows the access key can chat and start or stop the engine. Use it on networks you trust.",
        s.has_access_key ? null : h("div", null, h("b", null, "Set an access key below; without one, FLE Studio makes one up and prints it at start.")))) : null,
      h("div", { class: "grid two" },
        field("Port", port, "Default 8642."),
        field("Access key", h("div", { class: "row" }, key,
          h("button", { class: "btn", onclick: async () => { if (key.value.trim() && await save({ access_key: key.value.trim() })) render(); } }, "Save")),
          "Other machines sign in with it; API clients send it as the Bearer key.")),
      lan ? field("Extra host names", hosts, "Names this machine is reached under, comma separated. IP addresses always work.") : null,
      h("div", { class: "row" },
        h("button", { class: "btn", onclick: async () => {
          const patch = { server: { port: Math.max(1, Math.min(65535, Number(port.value) || 8642)) } };
          if (lan) patch.server.allowed_hosts = hosts.value.trim();
          if (await save(patch)) render();
        } }, lan ? "Save port and names" : "Save port")),
      h("label", { class: "row" }, toggle(srv.open_browser, (v) => save({ server: { open_browser: v } }, true)),
        h("span", null, "Open the browser when FLE Studio starts")));
  }

  function about() {
    const st = store.state;
    const sys = st.system || {};
    return section("info", "About", "",
      h("dl", { class: "kv" },
        h("dt", null, "FLE Studio"), h("dd", null, `${st.version} · ${st.platform}`),
        h("dt", null, "Python"), h("dd", null, sys.python || "–"),
        h("dt", null, "Data folder"), h("dd", { class: "mono" }, st.data_dir),
        h("dt", null, "Reference machine"), h("dd", null, st.reference?.machine || "–")),
      h("div", { class: "row wrap small" },
        h("a", { href: "https://github.com/sirfyyn/FastLocalExperts", target: "_blank", rel: "noopener" }, "FLE engine on GitHub ", icon("external", "sm")),
        h("a", { href: st.reference?.report || "https://sirfyyn.github.io/FastLocalExperts/", target: "_blank", rel: "noopener" }, "How FLE works ", icon("external", "sm")),
        h("a", { href: "https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF", target: "_blank", rel: "noopener" }, "FLE models ", icon("external", "sm"))),
      h("span", { class: "small faint" }, "FLE Studio talks to the internet only when you download a model or an engine, search Hugging Face, look for updates (once a day, switchable above), or turn on web search or a web MCP server for the chat. No telemetry."));
  }

  // ------------------------------------------------------------------------------------------- controls

  function field(label, control, hint = "") {
    return h("label", { class: "field" }, h("span", null, label), control, hint ? h("span", { class: "hint" }, hint) : null);
  }

  function seg(options, value, onPick) {
    const box = h("div", { class: "seg", role: "radiogroup" });
    for (const [v, label] of options) {
      const b = h("button", { type: "button", class: v === value ? "on" : "", role: "radio", "aria-checked": v === value ? "true" : "false" }, label);
      b.addEventListener("click", () => {
        box.querySelectorAll("button").forEach((x) => { x.classList.remove("on"); x.setAttribute("aria-checked", "false"); });
        b.classList.add("on");
        b.setAttribute("aria-checked", "true");
        onPick(v);
      });
      box.append(b);
    }
    return box;
  }

  function toggle(checked, onFlip) {
    return h("span", { class: "switch" }, h("input", { type: "checkbox", checked, onchange: (e) => onFlip(e.target.checked) }), h("i"));
  }

  unsub.push(on("state", render));
  render();
  return { unmount() { unsub.forEach((u) => u()); stopToolsPolling(); } };
}
