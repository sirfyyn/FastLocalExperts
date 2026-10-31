// A chat is a tree of messages (see chat_tree.py): every message has an id and a parent, `leaf` is the last message
// of the branch on screen, and `pick` on a message remembers which of its children was shown last.
// Plain functions on the chat object, no DOM, so a test can run them under Node.

export function newId() {
  const bytes = new Uint8Array(6);
  (globalThis.crypto || {}).getRandomValues?.(bytes);
  if (!bytes.some(Boolean)) for (let i = 0; i < bytes.length; i += 1) bytes[i] = Math.floor(Math.random() * 256);
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// Older chats and brand-new ones: give every message an id and a parent, and the chat a valid leaf.
export function ensureTree(chat) {
  const seen = new Set();
  const legacy = !chat.messages.some((m) => "parent" in m);
  let prev = null;
  for (const m of chat.messages) {
    if (!m.id || seen.has(m.id)) m.id = newId();
    seen.add(m.id);
    if (legacy) m.parent = prev;
    prev = m.id;
  }
  for (const m of chat.messages) if (m.parent && !seen.has(m.parent)) m.parent = null;
  if (!chat.leaf || !seen.has(chat.leaf)) chat.leaf = chat.messages.length ? chat.messages[chat.messages.length - 1].id : null;
  return chat;
}

export function indexOf(chat) {
  const byId = new Map();
  const kids = new Map();
  for (const m of chat.messages) {
    byId.set(m.id, m);
    const key = m.parent || "";
    if (!kids.has(key)) kids.set(key, []);
    kids.get(key).push(m);
  }
  return { byId, kids };
}

// The messages on screen: from the first one down to the leaf.
export function pathOf(chat, idx = indexOf(chat)) {
  const out = [];
  let node = idx.byId.get(chat.leaf);
  while (node && out.length <= chat.messages.length) {
    out.push(node);
    node = node.parent ? idx.byId.get(node.parent) : null;
  }
  return out.reverse();
}

export function siblingsOf(node, idx) {
  return idx.kids.get(node.parent || "") || [node];
}

// The last message of the branch below `node`: follow what was shown last, else the newest child.
export function deepest(node, idx) {
  let cur = node;
  for (let guard = 0; guard < 100000; guard += 1) {
    const kids = idx.kids.get(cur.id) || [];
    if (!kids.length) return cur;
    cur = kids.find((k) => k.id === cur.pick) || kids[kids.length - 1];
  }
  return cur;
}

// Show another sibling (and what was last shown below it).
export function switchTo(chat, node) {
  const idx = indexOf(chat);
  if (node.parent) idx.byId.get(node.parent).pick = node.id;
  chat.leaf = deepest(node, idx).id;
}

// Add a message below `parent` (null: a new first message) and show it.
export function addChild(chat, parent, message) {
  const node = { ...message, id: newId(), parent: parent ? parent.id : null };
  chat.messages.push(node);
  if (parent) parent.pick = node.id;
  chat.leaf = node.id;
  return node;
}

// Remove a message and everything below it. Another branch from the same parent is shown if there is one,
// otherwise the parent is the last message.
export function removeBranch(chat, node) {
  const idx = indexOf(chat);
  const drop = new Set();
  const stack = [node];
  while (stack.length) {
    const cur = stack.pop();
    drop.add(cur.id);
    stack.push(...(idx.kids.get(cur.id) || []));
  }
  chat.messages = chat.messages.filter((m) => !drop.has(m.id));
  const parent = node.parent ? idx.byId.get(node.parent) : null;
  if (parent && parent.pick === node.id) delete parent.pick;
  const idx2 = indexOf(chat);
  if (parent) chat.leaf = deepest(parent, idx2).id;
  else chat.leaf = chat.messages.length ? deepest(idx2.kids.get("")?.[0] || chat.messages[0], idx2).id : null;
}
