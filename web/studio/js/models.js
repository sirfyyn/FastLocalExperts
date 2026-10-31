// Models: what this machine can run, the FLE catalog, Hugging Face, downloads and the local library.

import { get, post } from "./api.js";
import { emit, navigate, on, refreshState, store, topbar } from "./app.js";
import { clear, confirmDialog, debounce, fmt, h, icon, toast } from "./ui.js";

const FIT = {
  good: ["good", "Fits this machine"],
  tight: ["warn", "Tight fit"],
  too_big: ["bad", "Too big for this machine"],
};

export function mount(main) {
  const unsub = [];
  const hw = h("div", { class: "hw" });
  const catGrid = h("div", { class: "cat-grid" });
  const dlBox = h("div");
  const libBox = h("div");
  const hfBox = h("div", { class: "card" });
  const page = h("div", { class: "page-inner" },
    h("div", { class: "page-head" }, h("div", null, h("h2", null, "Models"),
      h("p", null, "Pick a model that fits this computer. Downloads resume after an interruption and every file is checked against its SHA-256 before use."))),
    hw,
    section("FLE quants of Qwen3.8-Flash-Next", "Made for the FLE engine: experts in RAM, the busy ones cached on the GPU, MTP drafting for faster answers.", catGrid),
    dlBox,
    libBox,
    hfBox);
  main.replaceChildren(topbar("Models"), h("div", { class: "page" }, page));

  const mtpChoice = new Map(); // catalog id -> download the MTP head too
  let hfState = { q: "", results: [], repo: null, loading: false };

  unsub.push(on("state", renderAll));
  unsub.push(on("downloads", () => { renderCatalog(); renderDownloads(); }));
  renderAll();
  renderHF();
  refreshState().catch(() => {});

  function renderAll() {
    renderHardware();
    renderCatalog();
    renderDownloads();
    renderLibrary();
  }

  // ------------------------------------------------------------------------------------------- hardware

  function renderHardware() {
    const s = store.state?.system;
    if (!s) return;
    const gpus = s.gpus || [];
    const gpuText = gpus.length ? gpus.map((g) => g.name).join(", ") : "None found";
    const vram = gpus.reduce((a, g) => a + (g.unified ? 0 : g.memory_total || 0), 0);
    const apple = gpus.find((g) => g.unified);
    clear(hw,
      tile("Processor", s.cpu?.name || "–", `${s.cpu?.physical || "?"} cores · ${s.cpu?.arch || ""}${s.cpu?.avx2 === false ? " · no AVX2" : ""}`),
      tile("Memory", fmt.gib(s.memory?.total, 0), `${fmt.gib(s.memory?.available)} free right now`),
      tile(gpus.length > 1 ? "GPUs" : "GPU", apple ? apple.name : vram ? fmt.gib(vram, 0) : "–",
        apple ? `Unified memory · ${fmt.gib(apple.metal_limit, 0)} usable by the GPU` : gpuText),
      tile("Disk", fmt.bytes(s.disk_free), h("span", { title: s.models_dir }, "free in the models folder · ",
        h("a", { href: "/settings", "data-link": true }, "change"))));
  }

  function tile(k, v, sub) {
    return h("div", { class: "tile" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v), h("div", { class: "sub" }, sub));
  }

  // ------------------------------------------------------------------------------------------- catalog

  function renderCatalog() {
    const cat = store.state?.catalog;
    if (!cat) return;
    const jobs = new Map(store.downloads.map((d) => [d.dedupe, d]));
    clear(catGrid, cat.map((m) => modelCard(m, jobs.get(`catalog:${m.id}`))));
  }

  function modelCard(m, job) {
    const [cls, label] = FIT[m.fit?.verdict] || ["", ""];
    const local = m.local;
    const active = job && ["queued", "downloading", "verifying", "paused", "failed"].includes(job.state);
    const extras = store.state?.extras || {};
    const withMtp = mtpChoice.has(m.id) ? mtpChoice.get(m.id) : true;
    let action;
    if (active) {
      action = jobBlock(job);
    } else if (local?.complete) {
      const mtpMissing = m.mtp_status?.state === "missing";
      action = h("div", { class: "row wrap" },
        h("span", { class: "pill good" }, icon("check", "sm"), "In your library"),
        mtpMissing ? h("button", { class: "btn sm", title: "Faster answers through MTP drafting", onclick: () => download(m, true) },
          icon("download", "sm"), `Add MTP head (${fmt.bytes(extras.mtp_head?.size)})`) : null,
        h("span", { class: "grow" }),
        h("button", { class: "btn primary", onclick: () => loadModel(local.id) }, icon("play", "sm"), "Load"));
    } else {
      const cb = h("input", { type: "checkbox", checked: withMtp, onchange: (e) => { mtpChoice.set(m.id, e.target.checked); } });
      action = h("div", { class: "row wrap" },
        m.mtp ? h("label", { class: "check small muted" }, cb, `with MTP head (+${fmt.bytes(extras.mtp_head?.size)})`) : null,
        h("span", { class: "grow" }),
        h("button", { class: `btn ${m.fit?.verdict === "too_big" ? "" : "primary"}`, onclick: () => download(m, cb.checked) },
          icon("download", "sm"), `Download ${fmt.bytes(m.size)}`));
    }
    const kld = m.kld ? m.kld[0] : null;
    return h("div", { class: "card model-card" },
      h("div", { class: "title" }, h("h3", null, m.name), m.id.includes("reap") ? h("span", { class: "tag" }, "REAP") : null,
        h("span", { class: "grow" }), label ? h("span", { class: `pill ${cls}` }, label) : null),
      h("div", { class: "muted" }, m.summary),
      h("div", { class: "facts" },
        fact("Size", fmt.bytes(m.size)),
        fact("KL divergence", kld !== null ? fmt.two(kld) : "–", m.kld ? `Mean KL divergence against BF16 on three test texts: ${m.kld.map((x) => x.toFixed(3)).join(" / ")} (lower is better)` : ""),
        fact("Load time", m.measured?.load_s ? fmt.dur(m.measured.load_s) : "–", "Measured on the reference machine")),
      h("div", { class: "small faint" }, m.speed),
      m.mtp_status ? h("div", { class: "row small" },
        h("span", { class: `pill ${{ ready: "good", missing: "warn", none: "" }[m.mtp_status.state] || ""}`,
          title: m.mtp_status.state === "none" ? "This model has no MTP head; the engine drafts without it"
            : `MTP head ${m.mtp_status.head ? "there" : "missing"}, vocabulary file ${m.mtp_status.vocab ? "there" : "missing"}` },
        m.mtp_status.label)) : null,
      m.fit?.advice ? h("div", { class: "fit-line" }, icon("info", "sm"), h("span", null, m.fit.advice)) : null,
      action,
      h("div", { class: "row small faint" }, h("span", null, m.license), h("span", { class: "grow" }),
        h("a", { href: m.url, target: "_blank", rel: "noopener" }, "Model card ", icon("external", "sm"))));
  }

  function fact(k, v, title = "") {
    return h("div", { class: "fact", title }, h("span", { class: "k" }, k), h("span", { class: "v" }, v));
  }

  async function download(m, mtp) {
    try {
      const job = await post(`/api/catalog/${m.id}/download`, { mtp });
      upsertJob(job);
      toast(`Downloading ${m.name}`);
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  function upsertJob(job) {
    const i = store.downloads.findIndex((d) => d.id === job.id);
    if (i >= 0) store.downloads[i] = job; else store.downloads.push(job);
    emit("downloads");
  }

  // ------------------------------------------------------------------------------------------- downloads

  function jobBlock(job) {
    const pct = Math.round((job.progress || 0) * 100);
    const remaining = job.speed > 0 ? (job.bytes_total - job.bytes_done) / job.speed : null;
    const stateText = {
      queued: "Waiting", downloading: "Downloading", verifying: "Checking SHA-256", paused: "Paused", failed: "Failed",
      installing: "Installing", done: "Done", cancelled: "Cancelled",
    }[job.state] || job.state;
    return h("div", { class: "stack tight" },
      h("div", { class: "row small" }, h("b", null, stateText), h("span", { class: "muted num" }, `${pct} %`),
        h("span", { class: "grow" }),
        ["queued", "downloading", "verifying"].includes(job.state) ? h("button", { class: "btn ghost sm", onclick: () => act(job, "pause") }, icon("pause", "sm"), "Pause") : null,
        ["paused", "failed"].includes(job.state) ? h("button", { class: "btn sm", onclick: () => act(job, "resume") }, icon("play", "sm"), "Resume") : null,
        !["done", "cancelled"].includes(job.state) ? h("button", { class: "btn ghost sm danger", onclick: () => act(job, "cancel") }, "Cancel") : null,
        ["done", "cancelled", "failed"].includes(job.state) ? h("button", { class: "btn ghost sm", onclick: () => act(job, "remove") }, "Clear") : null),
      h("div", { class: `bar${job.state === "verifying" ? " cyan" : ""}` }, h("i", { style: { width: `${pct}%` } })),
      h("div", { class: "sub row small muted num wrap" },
        h("span", null, `${fmt.bytes(job.bytes_done)} of ${fmt.bytes(job.bytes_total)}`),
        job.state === "downloading" && job.speed ? h("span", null, `${fmt.bytes(job.speed)}/s`) : null,
        job.state === "downloading" && remaining ? h("span", null, `about ${fmt.dur(remaining)} left at this speed`) : null),
      job.error ? h("div", { class: `callout ${job.state === "failed" ? "bad" : "warn"}` }, icon("alert", "sm"), h("div", { class: "small" }, job.error)) : null);
  }

  async function act(job, what) {
    try {
      upsertJob(await post(`/api/downloads/${job.id}/${what}`));
      if (what === "remove") { store.downloads = store.downloads.filter((d) => d.id !== job.id); emit("downloads"); }
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  function renderDownloads() {
    const jobs = store.downloads.filter((d) => d.state !== "done" || Date.now() / 1000 - (d.finished || 0) < 600);
    if (!jobs.length) { clear(dlBox); return; }
    clear(dlBox, section("Downloads", "", h("div", { class: "card", style: { padding: 0 } },
      jobs.map((j) => h("div", { class: "dl" }, h("div", { class: "top" }, icon(j.kind === "engine" ? "engine" : "download", "sm"),
        h("b", null, j.title), h("span", { class: "tag" }, `${j.files.length} file${j.files.length > 1 ? "s" : ""}`)), jobBlock(j))))));
  }

  // ------------------------------------------------------------------------------------------- library

  function renderLibrary() {
    const models = store.state?.models || [];
    const path = h("input", { class: "input mono", placeholder: "/path/to/model.gguf or a folder" });
    const add = async () => {
      if (!path.value.trim()) return;
      try {
        await post("/api/models/add", { path: path.value.trim() });
        path.value = "";
        await refreshState();
        toast("Added to the library", "good");
      } catch (e) { toast(e.message, "bad", 7000); }
    };
    path.addEventListener("keydown", (e) => { if (e.key === "Enter") add(); });
    const engineModel = store.engine?.model?.id;
    const rows = models.map((m) => {
      const info = m.info || {};
      const loaded = engineModel === m.id && ["ready", "starting"].includes(store.engine?.state);
      const meta = [
        h("span", null, fmt.bytes(m.size)),
        info.arch ? h("span", null, info.arch) : null,
        info.experts ? h("span", null, `${info.experts_used} of ${info.experts} experts active`) : null,
        info.context_length ? h("span", null, `${fmt.ctx(info.context_length)} context`) : null,
        m.kind !== "model" ? h("span", { class: "tag" }, m.kind === "draft" ? "MTP / draft" : m.kind) : null,
        m.mtp?.supported ? h("span", null, m.mtp.head && m.mtp.vocab ? "MTP ready" : "MTP head missing") : null,
        !m.complete ? h("span", { class: "pill warn" }, `${m.parts.filter((p) => p.state !== "present").length} part(s) missing`) : null,
        m.error ? h("span", { class: "pill bad", title: m.error }, "Unreadable") : null,
      ];
      return h("div", { class: "lib-row" },
        h("div", { style: { minWidth: 0 } }, h("div", { class: "name" }, m.name, loaded ? h("span", { class: "pill good", style: { marginLeft: "8px" } }, "loaded") : null),
          h("div", { class: "meta" }, meta), h("div", { class: "small faint mono", style: { marginTop: "3px", overflowWrap: "anywhere" } }, m.path)),
        h("div", { class: "row" },
          m.kind === "model" && m.complete && !loaded ? h("button", { class: "btn sm primary", onclick: () => loadModel(m.id) }, icon("play", "sm"), "Load") : null,
          h("button", { class: "btn ghost icon sm", title: m.source === "import" ? "Remove from the list" : "Delete the files",
            "aria-label": "Remove", onclick: () => removeModel(m) }, icon("trash", "sm"))));
    });
    clear(libBox, section("Your library", `Models in ${store.state?.settings?.models_dir || "the models folder"} and files you added.`,
      h("div", { class: "card", style: { padding: 0 } },
        rows.length ? rows : h("div", { class: "empty" }, "No models yet. Download one above or add a GGUF file you already have."),
        h("div", { class: "lib-row", style: { borderTop: rows.length ? "1px solid var(--line)" : "0" } },
          path, h("button", { class: "btn", onclick: add }, icon("plus", "sm"), "Add from disk")))));
  }

  async function removeModel(m) {
    const imported = m.source === "import";
    const ok = await confirmDialog(imported
      ? { title: `Remove ${m.name} from the list?`, body: "The file stays where it is.", ok: "Remove" }
      : { title: `Delete ${m.name}?`, body: `This deletes ${fmt.bytes(m.size)} from the models folder.`, ok: "Delete files", danger: true });
    if (!ok) return;
    try {
      await post(`/api/models/${m.id}/remove`, { delete: !imported });
      await refreshState();
    } catch (e) { toast(e.message, "bad"); }
  }

  async function loadModel(id) {
    try {
      store.engine = await post("/api/engine/start", { model: id });
      emit("engine");
      navigate("/");
    } catch (e) {
      toast(e.message, "bad", 8000);
      navigate("/engine");
    }
  }

  // ------------------------------------------------------------------------------------------- Hugging Face

  function renderHF() {
    const input = h("input", { class: "input", placeholder: "Search GGUF models, or paste owner/name or a Hugging Face link", value: hfState.q });
    const results = h("div", { class: "hf-results" });
    const repoBox = h("div");
    const doSearch = debounce(async () => {
      const q = input.value.trim();
      hfState.q = q;
      if (q.length < 2) { clear(results); return; }
      try {
        const r = await get(`/api/hf/search?q=${encodeURIComponent(q)}`);
        hfState.results = r.results;
        const looksLikeRepo = /^[\w.-]+\/[\w.-]+$/.test(q) || q.includes("huggingface.co/");
        clear(results,
          looksLikeRepo ? h("div", { class: "hf-item", onclick: () => openRepo(q) }, icon("link", "sm"), h("span", { class: "repo" }, `Open ${q}`)) : null,
          r.results.map((x) => h("div", { class: "hf-item", onclick: () => openRepo(x.id) }, icon("models", "sm"),
            h("span", { class: "repo" }, x.id), h("span", { class: "small faint num" }, `${fmt.tokens(x.downloads || 0)} downloads`))),
          !r.results.length && !looksLikeRepo ? h("div", { class: "small muted", style: { padding: "8px 10px" } }, "No GGUF repos found.") : null);
      } catch (e) {
        clear(results, h("div", { class: "callout bad" }, icon("alert", "sm"), h("div", { class: "small" }, e.message)));
      }
    }, 350);
    input.addEventListener("input", doSearch);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && /\//.test(input.value)) openRepo(input.value.trim()); });

    async function openRepo(repo) {
      clear(repoBox, h("div", { class: "row small muted" }, h("div", { class: "bar indet", style: { width: "120px" } }, h("i")), "Reading the repo"));
      try {
        const r = await get(`/api/hf/repo?repo=${encodeURIComponent(repo)}`);
        const groups = r.groups;
        clear(repoBox, h("div", { class: "stack" },
          h("div", { class: "row wrap" }, h("b", { class: "mono" }, r.repo), r.info?.license ? h("span", { class: "tag" }, r.info.license) : null,
            r.info?.gated ? h("span", { class: "pill warn", title: "Accept the license on huggingface.co and add a token in Settings" }, "Gated") : null,
            h("span", { class: "grow" }), h("a", { href: `https://huggingface.co/${r.repo}`, target: "_blank", rel: "noopener", class: "small" }, "Open on Hugging Face ", icon("external", "sm"))),
          groups.length ? h("div", { class: "table-wrap" }, h("table", { class: "t" },
            h("thead", null, h("tr", null, h("th", null, "File"), h("th", null, "Kind"), h("th", { class: "r" }, "Size"), h("th", null, "On this machine"), h("th"))),
            h("tbody", null, groups.map((g) => {
              const [cls, label] = FIT[g.fit?.verdict] || ["", ""];
              return h("tr", null,
                h("td", null, h("b", { class: "mono" }, g.name), g.parts > 1 ? h("span", { class: "faint small" }, ` · ${g.parts} parts`) : null),
                h("td", null, h("span", { class: "tag" }, g.kind)),
                h("td", { class: "r" }, fmt.bytes(g.size)),
                h("td", null, label ? h("span", { class: `pill ${cls}` }, label) : h("span", { class: "faint" }, "–")),
                h("td", { class: "r" }, h("button", { class: "btn sm", onclick: () => hfDownload(r.repo, r.revision, g) }, icon("download", "sm"), "Download")));
            })))) : h("div", { class: "empty" }, "This repo has no GGUF files.")));
      } catch (e) {
        clear(repoBox, h("div", { class: "callout bad" }, icon("alert", "sm"), h("div", { class: "small" }, e.message)));
      }
    }

    clear(hfBox,
      h("div", { class: "card-head" }, icon("search"), h("div", null, h("h3", null, "Any GGUF from Hugging Face"),
        h("p", null, "Works with the upstream llama.cpp engine too. Big MoE models run best with the FLE engine."))),
      h("div", { class: "stack" }, input, results, repoBox));
  }

  async function hfDownload(repo, revision, group) {
    try {
      const job = await post("/api/hf/download", { repo, revision, files: group.files.map((f) => f.path) });
      upsertJob(job);
      toast(`Downloading ${group.name}`);
      window.scrollTo({ top: 0 });
    } catch (e) {
      toast(e.message, "bad", 7000);
    }
  }

  return { unmount() { unsub.forEach((u) => u()); } };
}

function section(title, sub, body) {
  return h("section", { class: "stack" },
    h("div", null, h("h3", { style: { margin: 0, fontSize: "16px" } }, title), sub ? h("p", { class: "muted small", style: { margin: "2px 0 0" } }, sub) : null),
    body);
}
