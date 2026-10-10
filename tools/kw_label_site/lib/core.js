// Server-side rules for the labelling site. Kept free of I/O so they can be tested directly.
import { createHmac, randomBytes, scryptSync, timingSafeEqual } from "node:crypto";

const NAME_RE = /^[a-z0-9-]{2,20}$/;
const VALUES = {
  claim: ["yes", "no"],
  sentiment: ["positive", "negative", "neutral", "mixed"],
  support: ["supported", "unsupported", "contradicted"],
};
const MAX_NOTE = 500;

/** Token -> annotator name, from the ANNOTATOR_TOKENS environment variable (a JSON object). */
export function parseTokens(json) {
  let raw;
  try { raw = JSON.parse(json); } catch { throw new Error("ANNOTATOR_TOKENS is not valid JSON"); }
  const tokens = new Map();
  for (const [token, name] of Object.entries(raw || {})) {
    if (typeof token !== "string" || token.length < 16) throw new Error("every token needs at least 16 characters");
    if (!NAME_RE.test(String(name))) throw new Error(`annotator name ${JSON.stringify(name)} must be 2-20 lowercase letters, digits or hyphens`);
    tokens.set(token, String(name));
  }
  return tokens;
}

export function annotatorFor(tokens, token) {
  return typeof token === "string" && tokens.has(token) ? tokens.get(token) : null;
}

export function isAdmin(adminToken, given) {
  if (!adminToken || typeof given !== "string") return false;
  const a = Buffer.from(adminToken), b = Buffer.from(given);
  return a.length === b.length && timingSafeEqual(a, b);
}

// --- Sign-in on the main page: a name and a PIN the annotator chooses the first time. The first
// sign-in with a new name claims it; later ones need the same PIN, so nobody can type someone
// else's name and change their labels. A signed session then stands in for the personal link.

const SESSION_DAYS = 60;

/** The annotator name as typed, made case-insensitive and safe for storage paths, or null. */
export function normaliseName(raw) {
  if (typeof raw !== "string") return null;
  const name = raw.trim().toLowerCase().replace(/\s+/g, "-");
  return NAME_RE.test(name) ? name : null;
}

export const validPin = (pin) => typeof pin === "string" && /^\d{4,8}$/.test(pin);

/** What is stored for a claimed name: a salted scrypt hash, never the PIN itself. */
export function hashPin(pin, salt = randomBytes(16).toString("hex")) {
  return { salt, hash: scryptSync(pin, salt, 32).toString("hex") };
}

export function checkPin(pin, account) {
  if (!validPin(pin) || !account || typeof account.salt !== "string" || typeof account.hash !== "string") return false;
  const want = Buffer.from(account.hash, "hex");
  const got = scryptSync(pin, account.salt, 32);
  return want.length === got.length && timingSafeEqual(want, got);
}

const mac = (payload, secret) => createHmac("sha256", secret).update(payload).digest("base64url");

/** A session token: "s1.<name>.<expiry ms>.<HMAC>", checked without any stored state. */
export function signSession(name, secret, now = Date.now()) {
  const payload = `s1.${name}.${now + SESSION_DAYS * 864e5}`;
  return `${payload}.${mac(payload, secret)}`;
}

export function verifySession(token, secret, now = Date.now()) {
  if (typeof token !== "string" || !secret) return null;
  const parts = token.split(".");
  if (parts.length !== 4 || parts[0] !== "s1" || !NAME_RE.test(parts[1])) return null;
  const payload = parts.slice(0, 3).join(".");
  const want = Buffer.from(mac(payload, secret)), got = Buffer.from(parts[3]);
  if (want.length !== got.length || !timingSafeEqual(want, got)) return null;
  return Number(parts[2]) > now ? parts[1] : null;
}

const norm = (v) => (typeof v === "string" ? v.trim().toLowerCase() : "");

/** Labels for one summary, as the page sends them. Unknown items and fields are dropped;
 * an unknown value refuses the whole save, so nothing malformed is ever stored. */
export function validateLabels(body, item) {
  if (!body || typeof body !== "object" || Array.isArray(body)) return { ok: false, error: "labels must be an object" };
  const allowed = new Set(item.sentences.map((s) => s.item_id));
  const items = {};
  for (const [id, row] of Object.entries(body)) {
    if (!allowed.has(id) || !row || typeof row !== "object") continue;
    const out = { claim: null, sentiment: null, support: null, notes: "" };
    for (const key of Object.keys(VALUES)) {
      const v = norm(row[key]);
      if (!v) continue;
      if (!VALUES[key].includes(v)) return { ok: false, error: `${id}: ${key} "${v}" is not one of ${VALUES[key].join("/")}` };
      out[key] = v;
    }
    const notes = typeof row.notes === "string" ? row.notes.trim() : "";
    if (notes.length > MAX_NOTE) return { ok: false, error: `${id}: notes longer than ${MAX_NOTE} characters` };
    out.notes = notes;
    if (out.claim === "no") { out.sentiment = null; out.support = null; }
    items[id] = out;
  }
  return { ok: true, items };
}

/** The study data the owner seeds: {status, guide, items}. Checked before it is stored, so the
 * page and the label check can rely on its shape. */
export function validateData(d) {
  const fail = (error) => ({ ok: false, error });
  if (!d || typeof d !== "object") return fail("data must be an object");
  if (!["practice", "open", "closed"].includes(d.status)) return fail("status must be practice, open or closed");
  if (!Array.isArray(d.guide) || !d.guide.every((l) => typeof l === "string")) return fail("guide must be a list of lines");
  if (!Array.isArray(d.items) || d.items.length === 0) return fail("items must be a non-empty list");
  const seen = new Set();
  for (const it of d.items) {
    if (!it || typeof it.id !== "string" || !/^[A-Za-z0-9]{1,8}$/.test(it.id)) return fail("every item needs a short id");
    if (seen.has(it.id)) return fail(`duplicate item ${it.id}`);
    seen.add(it.id);
    if (typeof it.topic !== "string" || !Array.isArray(it.reviews) || it.reviews.length === 0
        || !it.reviews.every((r) => typeof r === "string")) return fail(`${it.id}: needs a topic and its reviews`);
    if (!Array.isArray(it.sentences) || it.sentences.length === 0) return fail(`${it.id}: needs sentences`);
    for (const s of it.sentences) {
      if (!s || typeof s.text !== "string" || typeof s.item_id !== "string" || !s.item_id.startsWith(`${it.id}-`)) {
        return fail(`${it.id}: every sentence needs text and an item id starting with ${it.id}-`);
      }
    }
  }
  return { ok: true };
}

/** From a blob listing of labels/<annotator>/<summary>/<file>, the URL of each summary's latest save. */
export function latestPerSummary(blobs) {
  const best = {};
  for (const b of blobs) {
    const parts = String(b.pathname).split("/");
    if (parts.length !== 4 || parts[0] !== "labels") continue;
    const [, who, sid] = parts;
    const at = Date.parse(b.uploadedAt);
    best[who] = best[who] || {};
    if (!best[who][sid] || at > best[who][sid].at) best[who][sid] = { at, url: b.url };
  }
  return Object.fromEntries(Object.entries(best).map(([who, sids]) =>
    [who, Object.fromEntries(Object.entries(sids).map(([sid, v]) => [sid, v.url]))]));
}
