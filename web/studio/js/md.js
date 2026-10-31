// Markdown for model answers, safe by construction: the text is HTML-escaped first and only tags written here
// are added. Links are limited to http(s) and mailto. Unclosed code fences (while an answer streams) render as code.

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);

const LIST_RE = /^(\s*)([-*+]|\d{1,9}[.)])(?:(\s+)(.*))?$/;
const FENCE_RE = /^(\s{0,3})(`{3,}|~{3,})\s*([^`\s]*)[^`]*$/;
const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const HR_RE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
const COPY_SVG = '<svg class="i sm" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 9h10v11H9zM5 15V4h10"/></svg>';

export function renderMarkdown(src) {
  const text = String(src || "").replace(/\r\n?/g, "\n").replace(/\u0000|\u0001/g, "");
  return blocks(text.split("\n")).join("");
}

// ----------------------------------------------------------------------------------------------- streaming

// For a text that only grows (an answer while it streams). The text is cut at blank lines that surely end a block;
// the part above the last cut is rendered once and kept, only the open rest is rendered again on every update.
// update(text) returns {reset, fresh, tail}: `fresh` is the HTML of blocks that became final since the last call
// (append it after what is shown), `tail` the HTML of the open rest (replace the old tail with it). `reset` means the
// text changed in a way that is not growth: show `fresh` instead of what was shown. fresh + tail is always the same
// as renderMarkdown of the whole text.
export function createStreamRenderer() {
  let done = ""; // the text up to the last cut, as it was rendered
  let doneHtml = "";
  return {
    get html() { return doneHtml; },
    update(src) {
      const text = String(src || "").replace(/\r\n?/g, "\n").replace(/\u0000|\u0001/g, "");
      let reset = false;
      if (!text.startsWith(done)) { done = ""; doneHtml = ""; reset = true; }
      const cut = findCut(text, done.length);
      let fresh = "";
      if (cut > done.length) {
        fresh = blocks(text.slice(done.length, cut).split("\n")).join("");
        done = text.slice(0, cut);
        doneHtml += fresh;
      }
      const rest = text.slice(done.length);
      return { reset, fresh: reset ? doneHtml : fresh, tail: blocks(rest.split("\n")).join("") };
    },
  };
}

// The end (just after the "\n" of the last line of a block) of the last block that is final, searching from `from`
// (a line start); `from` when there is none. A block is final when a blank line follows it, and after the blank line
// comes a whole line that cannot continue it. Fences and $$ / \[ formulas may contain blank lines.
function findCut(text, from) {
  let cut = from;
  let pos = from;
  let fence = null; // the closing pattern of an open code fence
  let math = null;
  let lastText = ""; // the last line with text before the blank lines
  let blankAt = -1; // where the blank lines after it begin
  while (pos < text.length) {
    const nl = text.indexOf("\n", pos);
    if (nl < 0) break; // the last line is not complete: it is never a reason to cut
    const line = text.slice(pos, nl);
    const next = nl + 1;
    if (fence) {
      if (fence.test(line)) fence = null;
    } else if (math) {
      if (math.test(line)) math = null;
    } else if (blank(line)) {
      if (blankAt < 0 && lastText) blankAt = pos;
    } else {
      if (blankAt >= 0 && !/^\s/.test(line) && !(LIST_RE.test(line) && LIST_RE.exec(line)[4] !== undefined && continuesList(lastText))) {
        cut = blankAt;
      }
      blankAt = -1;
      lastText = line;
      const f = line.match(FENCE_RE);
      if (f) fence = new RegExp(`^\\s{0,3}${f[2][0] === "`" ? "`" : "~"}{${f[2].length},}\\s*$`);
      else if (/^\s*(\$\$|\\\[)/.test(line)) {
        const closeRe = line.trim().startsWith("$$") ? /\$\$\s*$/ : /\\\]\s*$/;
        const first = line.trim().slice(2);
        if (!(first && closeRe.test(first))) math = closeRe;
      }
    }
    pos = next;
  }
  return cut;
}

// A line of a list (or of an item's content): a list item after it belongs to the same list.
const continuesList = (line) => /^\s/.test(line) || (LIST_RE.test(line) && LIST_RE.exec(line)[4] !== undefined);

const blank = (l) => /^\s*$/.test(l);
const indentOf = (l) => l.match(/^\s*/)[0].replace(/\t/g, "    ").length;

function isTableSep(line) {
  if (!line || !line.includes("-")) return false;
  const cells = splitRow(line);
  return cells.length > 0 && cells.every((c) => /^:?-{1,}:?$/.test(c)) && (line.includes("|") || cells.length > 1);
}

function startsBlock(line, next) {
  return FENCE_RE.test(line) || HEADING_RE.test(line) || HR_RE.test(line) || /^\s{0,3}>/.test(line) ||
    /^\s*\$\$/.test(line) || /^\s*\\\[\s*$/.test(line) || (LIST_RE.test(line) && LIST_RE.exec(line)[4] !== undefined) ||
    (line.includes("|") && isTableSep(next));
}

function blocks(lines) {
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (blank(line)) { i += 1; continue; }

    let m = line.match(FENCE_RE);
    if (m) {
      const fence = m[2];
      const indent = m[1].length;
      const close = new RegExp(`^\\s{0,3}${fence[0] === "`" ? "`" : "~"}{${fence.length},}\\s*$`);
      const body = [];
      let closed = false;
      i += 1;
      while (i < lines.length) {
        if (close.test(lines[i])) { closed = true; i += 1; break; }
        body.push(indent ? lines[i].replace(new RegExp(`^ {0,${indent}}`), "") : lines[i]);
        i += 1;
      }
      out.push(codeBlock(body.join("\n"), m[3], !closed));
      continue;
    }

    if (/^\s*(\$\$|\\\[)/.test(line)) {
      const isDollar = line.trim().startsWith("$$");
      const closeRe = isDollar ? /\$\$\s*$/ : /\\\]\s*$/;
      const first = line.trim().slice(2);
      const body = [];
      if (first && closeRe.test(first)) {
        body.push(first.replace(closeRe, ""));
        i += 1;
      } else {
        if (first) body.push(first);
        i += 1;
        while (i < lines.length) {
          if (closeRe.test(lines[i])) { body.push(lines[i].replace(closeRe, "")); i += 1; break; }
          body.push(lines[i]);
          i += 1;
        }
      }
      out.push(`<div class="math">${esc(body.join("\n").trim())}</div>`);
      continue;
    }

    m = line.match(HEADING_RE);
    if (m) {
      const n = m[1].length;
      out.push(`<h${n}>${inline(m[2])}</h${n}>`);
      i += 1;
      continue;
    }

    if (HR_RE.test(line)) { out.push("<hr>"); i += 1; continue; }

    if (/^\s{0,3}>/.test(line)) {
      const inner = [];
      while (i < lines.length && /^\s{0,3}>/.test(lines[i])) {
        inner.push(lines[i].replace(/^\s{0,3}>\s?/, ""));
        i += 1;
      }
      out.push(`<blockquote>${blocks(inner).join("")}</blockquote>`);
      continue;
    }

    if (line.includes("|") && isTableSep(lines[i + 1])) {
      const head = splitRow(line);
      const align = splitRow(lines[i + 1]).map((c) => (c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : ""));
      i += 2;
      const rows = [];
      while (i < lines.length && !blank(lines[i]) && lines[i].includes("|")) {
        rows.push(splitRow(lines[i]));
        i += 1;
      }
      const cell = (tag, txt, k) => `<${tag}${align[k] ? ` style="text-align:${align[k]}"` : ""}>${inline(txt || "")}</${tag}>`;
      out.push(`<table><thead><tr>${head.map((c, k) => cell("th", c, k)).join("")}</tr></thead><tbody>${
        rows.map((r) => `<tr>${head.map((_, k) => cell("td", r[k], k)).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }

    m = line.match(LIST_RE);
    if (m && m[4] !== undefined) {
      const [html, next] = list(lines, i);
      out.push(html);
      i = next;
      continue;
    }

    const para = [line];
    i += 1;
    while (i < lines.length && !blank(lines[i]) && !startsBlock(lines[i], lines[i + 1])) {
      para.push(lines[i]);
      i += 1;
    }
    out.push(`<p>${inline(para.join("\n"))}</p>`);
  }
  return out;
}

function list(lines, start) {
  const first = lines[start].match(LIST_RE);
  const base = indentOf(lines[start]);
  const ordered = /\d/.test(first[2]);
  const same = (mm) => mm && /\d/.test(mm[2]) === ordered;
  const items = [];
  let cur = null;
  let contentIndent = 0;
  let loose = false;
  let sawBlank = false;
  let i = start;
  while (i < lines.length) {
    const line = lines[i];
    const m = line.match(LIST_RE);
    const ind = indentOf(line);
    if (m && m[4] !== undefined && ind === base && same(m)) {
      if (sawBlank && cur) loose = true;
      cur = [m[4]];
      items.push(cur);
      contentIndent = base + m[2].length + (m[3] ? Math.min(m[3].length, 4) : 1);
      sawBlank = false;
      i += 1;
      continue;
    }
    if (blank(line)) {
      let j = i + 1;
      while (j < lines.length && blank(lines[j])) j += 1;
      if (j >= lines.length) { i = j; break; }
      const nm = lines[j].match(LIST_RE);
      const nind = indentOf(lines[j]);
      if ((nm && nind === base && same(nm)) || nind >= contentIndent) {
        cur.push("");
        sawBlank = true;
        i += 1;
        continue;
      }
      break;
    }
    if (ind >= contentIndent || (m && ind > base)) {
      cur.push(line.slice(Math.min(ind, contentIndent)));
      i += 1;
      continue;
    }
    if (!m && !sawBlank && !startsBlock(line, lines[i + 1])) {
      cur.push(line.trim()); // lazy continuation of the item's paragraph
      i += 1;
      continue;
    }
    break;
  }
  const tag = ordered ? "ol" : "ul";
  const num = ordered ? parseInt(first[2], 10) : 1;
  const lis = items.map((itemLines) => {
    let task = "";
    const t = itemLines[0].match(/^\[([ xX])\]\s+/);
    if (t) {
      task = t[1] === " " ? "☐ " : "☑ ";
      itemLines[0] = itemLines[0].slice(t[0].length);
    }
    let inner = blocks(itemLines).join("");
    if (!loose) inner = inner.replace(/^<p>([\s\S]*?)<\/p>/, "$1");
    return `<li>${task}${inner}</li>`;
  });
  return [`<${tag}${ordered && num !== 1 ? ` start="${num}"` : ""}>${lis.join("")}</${tag}>`, i];
}

function splitRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
  const cells = [];
  let cur = "";
  let code = false;
  for (let k = 0; k < s.length; k += 1) {
    const ch = s[k];
    if (ch === "\\" && s[k + 1] === "|") { cur += "|"; k += 1; continue; }
    if (ch === "`") code = !code;
    if (ch === "|" && !code) { cells.push(cur.trim()); cur = ""; continue; }
    cur += ch;
  }
  cells.push(cur.trim());
  return cells;
}

function inline(text) {
  const stash = [];
  const keep = (html) => { stash.push(html); return `\u0000${stash.length - 1}\u0000`; };
  let s = text
    .replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, (_, _t, code) => keep(`<code>${esc(code.replace(/^ (.*) $/, "$1"))}</code>`))
    .replace(/\\([\\`*_{}[\]()#+\-.!|~>$])/g, (_, ch) => keep(esc(ch)));
  s = esc(s);
  s = s.replace(/\[([^\]\n]+)\]\(\s*((?:https?:\/\/|mailto:)[^\s)]+)(?:\s+&quot;[^\n]*?&quot;)?\s*\)/g,
    (_, label, url) => keep(`<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`));
  s = s.replace(/(^|[\s(])(https?:\/\/[^\s<]*[^\s<.,:;!?'")\]])/g,
    (_, pre, url) => `${pre}${keep(`<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`)}`);
  s = s.replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^\w])__(?=\S)([\s\S]*?\S)__(?!\w)/g, "$1<strong>$2</strong>")
    .replace(/(^|[^*\w])\*(?=[^\s*])([^*\n]*?[^\s*])\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/(^|[^_\w])_(?=[^\s_])([^_\n]*?[^\s_])_(?![_\w])/g, "$1<em>$2</em>")
    .replace(/~~(?=\S)([\s\S]*?\S)~~/g, "<del>$1</del>")
    .replace(/ {2,}\n|\\\n/g, "<br>")
    .replace(/\n/g, "<br>");
  for (let n = 0; n < 3 && s.includes("\u0000"); n += 1) {
    s = s.replace(/\u0000(\d+)\u0000/g, (_, k) => stash[+k]);
  }
  return s;
}

// ----------------------------------------------------------------------------------------------- code

function codeBlock(code, lang, open) {
  const label = (lang || "").toLowerCase();
  return `<div class="codeblock${open ? " open" : ""}"><div class="head"><span>${esc(label || "text")}</span>` +
    `<button class="btn ghost sm" data-copy type="button">${COPY_SVG}<span>Copy</span></button></div>` +
    `<pre><code>${highlight(code, label)}</code></pre></div>`;
}

const words = (s) => new Set(s.split(/\s+/).filter(Boolean));
const LANGS = {
  py: {
    kw: words("and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case"),
    lit: words("True False None self cls"), comments: ["#"], strings: ['"""', "'''", '"', "'"],
  },
  js: {
    kw: words("async await break case catch class const continue debugger default delete do else export extends finally for from function if import in instanceof let new of return super switch this throw try typeof var void while with yield interface type enum implements private public protected readonly as declare"),
    lit: words("true false null undefined NaN Infinity"), comments: ["//", "/*"], strings: ['"', "'", "`"],
  },
  c: {
    kw: words("auto break case char const continue default do double else enum extern float for goto if inline int long register return short signed sizeof static struct switch typedef union unsigned void volatile while class namespace template typename public private protected virtual override new delete using bool constexpr __global__ __device__ __host__ fn let mut pub impl trait use mod match loop move ref where dyn async await func package import go defer chan select range var type map interface size_t uint8_t uint32_t int64_t float32 float64 i32 i64 u8 u32 u64 f32 f64 usize String Vec Option Result"),
    lit: words("true false NULL nullptr None Some Ok Err self Self this"), comments: ["//", "/*"], strings: ['"', "'"],
  },
  sh: {
    kw: words("if then else elif fi for while until do done case esac function in return export local set unset source alias echo cd sudo exit"),
    lit: words("true false"), comments: ["#"], strings: ['"', "'"],
  },
  json: { kw: words(""), lit: words("true false null"), comments: [], strings: ['"'] },
  yaml: { kw: words(""), lit: words("true false null yes no on off"), comments: ["#"], strings: ['"', "'"] },
  sql: {
    kw: words("select from where insert into values update set delete create table drop alter add join left right inner outer on group by order having limit offset as and or not is in like distinct union all case when then else end primary key foreign references index with"),
    lit: words("true false null"), comments: ["--"], strings: ["'", '"'], ci: true,
  },
};
const ALIAS = {
  python: "py", py: "py", python3: "py", javascript: "js", js: "js", mjs: "js", ts: "js", typescript: "js", jsx: "js", tsx: "js",
  json: "json", jsonc: "json", c: "c", h: "c", cpp: "c", "c++": "c", cc: "c", hpp: "c", cu: "c", cuda: "c", rust: "c", rs: "c",
  go: "c", golang: "c", java: "c", kotlin: "c", kt: "c", swift: "c", cs: "c", csharp: "c", zig: "c", bash: "sh", sh: "sh",
  shell: "sh", zsh: "sh", console: "sh", powershell: "sh", ps1: "sh", ps: "sh", bat: "sh", cmd: "sh", yaml: "yaml", yml: "yaml",
  toml: "yaml", ini: "yaml", conf: "yaml", sql: "sql",
};
const RE_CACHE = new Map();

function tokenRe(spec) {
  if (RE_CACHE.has(spec)) return RE_CACHE.get(spec);
  const parts = [];
  const com = [];
  for (const c of spec.comments) {
    if (c === "/*") com.push("\\/\\*[\\s\\S]*?(?:\\*\\/|$)");
    else com.push(`${c.replace(/[/*]/g, "\\$&")}[^\\n]*`);
  }
  parts.push(com.length ? `(${com.join("|")})` : "(\\u0000never)");
  const str = spec.strings.map((q) => {
    if (q.length === 3) return `${q}[\\s\\S]*?(?:${q}|$)`;
    if (q === "`") return "`(?:\\\\[\\s\\S]|[^`\\\\])*`?";
    return `${q}(?:\\\\.|[^${q}\\\\\\n])*${q}?`;
  });
  parts.push(`(${str.join("|")})`);
  parts.push("(\\b(?:0[xX][\\da-fA-F_]+|\\d[\\d_]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\b)");
  parts.push("([A-Za-z_$][\\w$]*)(?=(\\s*\\()?)");
  const re = new RegExp(parts.join("|"), "g");
  RE_CACHE.set(spec, re);
  return re;
}

function highlight(code, lang) {
  if (lang === "diff" || lang === "patch") {
    return code.split("\n").map((l) => {
      const cls = l.startsWith("+") && !l.startsWith("+++") ? "tok-add" : l.startsWith("-") && !l.startsWith("---") ? "tok-del" :
        l.startsWith("@@") ? "tok-hunk" : "";
      return cls ? `<span class="${cls}">${esc(l)}</span>` : esc(l);
    }).join("\n");
  }
  const spec = LANGS[ALIAS[lang]];
  if (!spec || code.length > 200000) return esc(code);
  const re = tokenRe(spec);
  re.lastIndex = 0;
  let out = "";
  let last = 0;
  let m;
  while ((m = re.exec(code))) {
    if (m[0] === "") { re.lastIndex += 1; continue; }
    out += esc(code.slice(last, m.index));
    const [tok, com, str, num, ident, call] = m;
    if (com) out += `<span class="tok-c">${esc(tok)}</span>`;
    else if (str) out += `<span class="tok-s">${esc(tok)}</span>`;
    else if (num) out += `<span class="tok-n">${esc(tok)}</span>`;
    else if (ident) {
      const key = spec.ci ? ident.toLowerCase() : ident;
      if (spec.kw.has(key)) out += `<span class="tok-k">${esc(ident)}</span>`;
      else if (spec.lit.has(key)) out += `<span class="tok-l">${esc(ident)}</span>`;
      else if (call !== undefined) out += `<span class="tok-f">${esc(ident)}</span>`;
      else out += esc(ident);
    } else out += esc(tok);
    last = m.index + tok.length;
  }
  return out + esc(code.slice(last));
}
