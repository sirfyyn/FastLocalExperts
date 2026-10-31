// The body of a /v1/chat/completions request, built from the messages on screen. No DOM, so a test can run it
// under Node (tests/js/check_request.mjs).
//
// One rule is kept here for every request: tools and a forced answer format (response_format, json_schema, grammar)
// never travel together. llama-server then builds a grammar for the format and the model does not call the tools;
// it answers with substitute JSON and says nothing about it. A request that asks for a format therefore loses its
// tools. (The server refuses such a request too, see tools.py.)

export const FORMAT_FIELDS = ["response_format", "json_schema", "grammar"];

// Same rule as attachments.py: about 3.6 characters per token for ASCII, one token per other character.
export function estimateTokens(text) {
  if (!text) return 0;
  let ascii = 0;
  for (let i = 0; i < text.length; i += 1) if (text.charCodeAt(i) < 128) ascii += 1;
  return Math.max(1, Math.ceil(ascii / 3.6 + (text.length - ascii) * 0.9));
}

const filled = (v) => (v === null || v === undefined || v === "" || v === false ? false :
  typeof v === "object" ? Object.keys(v).length > 0 : true);

export function forcesFormat(body) {
  const f = body.response_format;
  if (f && typeof f === "object" && f.type && f.type !== "text") return true;
  return filled(body.json_schema) || filled(body.grammar);
}

export function finalizeRequest(body) {
  if (!body.tools || !body.tools.length) {
    delete body.tools;
    delete body.tool_choice;
  } else if (forcesFormat(body)) {
    delete body.tools;
    delete body.tool_choice;
  }
  return body;
}

const attr = (s) => String(s).replace(/"/g, "'");

// What the model reads for a text attachment.
export function attachmentBlock(a) {
  return `<attachment name="${attr(a.name)}">\n${a.text}\n</attachment>`;
}

// A user message as the API wants it: a string, or text plus pictures when there are pictures.
export function userContent(node, images = new Map()) {
  const blocks = (node.attachments || []).filter((a) => a.kind === "text").map(attachmentBlock);
  const text = [...blocks, node.content || ""].filter(Boolean).join("\n\n");
  const pics = (node.attachments || []).filter((a) => a.kind === "image" && images.get(a.blob));
  if (!pics.length) return text;
  return [{ type: "text", text: text || " " }, ...pics.map((a) => ({ type: "image_url", image_url: { url: images.get(a.blob) } }))];
}

function toolMessages(node, withReasoning) {
  const calls = (node.tool_calls || []).filter((c) => c.name);
  if (!calls.length) return [];
  const asked = { role: "assistant", content: node.content || "", tool_calls: calls.map((c) => ({
    id: c.id, type: "function", function: { name: c.name, arguments: c.arguments || "{}" } })) };
  if (withReasoning && node.reasoning) asked.reasoning_content = node.reasoning;
  const answers = calls.map((c) => ({ role: "tool", tool_call_id: c.id,
    content: typeof c.result === "string" && c.result ? c.result : "The call was interrupted before it finished." }));
  return [asked, ...answers];
}

// The thoughts of an answer travel back only for the messages after the last user message, which is where the chat
// templates (Qwen and others) keep them. Then the prompt of the next step starts with exactly the text the engine
// has already read, and its prompt cache holds. Older thoughts are left out, as the templates do themselves.
export function apiMessages(nodes, images = new Map()) {
  const out = [];
  let lastUser = -1;
  nodes.forEach((m, i) => { if (m.role === "user") lastUser = i; });
  nodes.forEach((m, i) => {
    if (m.role === "user") out.push({ role: "user", content: userContent(m, images) });
    else if (m.role === "assistant") {
      const withReasoning = i > lastUser;
      const withTools = toolMessages(m, withReasoning);
      if (withTools.length) out.push(...withTools);
      else if (m.content) {
        const msg = { role: "assistant", content: m.content };
        if (withReasoning && m.reasoning) msg.reasoning_content = m.reasoning;
        out.push(msg);
      }
    }
  });
  return out;
}

// By name, so that the same set of tools is always the same text in the prompt.
export function sortedTools(tools) {
  const name = (t) => String(t?.function?.name ?? "");
  return [...tools].sort((a, b) => (name(a) < name(b) ? -1 : name(a) > name(b) ? 1 : 0));
}

export function usedImages(nodes) {
  return nodes.flatMap((m) => (m.attachments || []).filter((a) => a.kind === "image").map((a) => a.blob));
}

// nodes: the conversation up to (not including) the message that will be generated.
// noMoreTools: the last round of a turn. The tools stay in the request, only `tool_choice: "none"` stops the call:
// taking the tools out would change the front of the prompt and make the engine read all of it again.
export function buildRequest({ nodes, system = "", settings = {}, tools = [], images = new Map(), extra = {}, noMoreTools = false }) {
  const msgs = apiMessages(nodes, images);
  if (system) msgs.unshift({ role: "system", content: system });
  const body = { messages: msgs, stream_options: { include_usage: true },
    chat_template_kwargs: { enable_thinking: settings.thinking !== false } };
  if (settings.sampling === "custom") {
    Object.assign(body, { temperature: settings.temperature, top_p: settings.top_p, top_k: settings.top_k,
      min_p: settings.min_p, presence_penalty: settings.presence_penalty });
  }
  if (settings.max_tokens > 0) body.max_tokens = settings.max_tokens;
  if (tools.length) {
    body.tools = sortedTools(tools);
    if (noMoreTools) body.tool_choice = "none";
  }
  Object.assign(body, extra);
  return finalizeRequest(body);
}

// A rough size of the request in tokens, for the context warning. Pictures count a fixed amount.
export function estimateRequest(body) {
  let total = 0;
  for (const m of body.messages) {
    if (typeof m.content === "string") total += estimateTokens(m.content) + 4;
    else for (const p of m.content || []) total += p.type === "text" ? estimateTokens(p.text) : 600;
    if (m.reasoning_content) total += estimateTokens(m.reasoning_content);
    for (const c of m.tool_calls || []) total += estimateTokens(c.function.name + c.function.arguments) + 8;
  }
  if (body.tools) total += estimateTokens(JSON.stringify(body.tools));
  return total;
}
