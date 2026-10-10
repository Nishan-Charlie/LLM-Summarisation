// Shared plumbing for the API routes: who is asking, the study data, and where labels are kept
// (Vercel Blob). Every write is a new, never-overwritten file and the latest one counts, so a
// reader can never be served a stale cached copy.
import { list, put } from "@vercel/blob";
import { annotatorFor, latestPerSummary, parseTokens, verifySession } from "./core.js";

let tokens = null;
let dataCache = { at: 0, data: null };

/** The annotator a request belongs to, or null: a signed session from the sign-in on the main
 * page ("s1.…"), or an older personal link's token. Throws a SetupError when the setup itself is
 * broken, so the site says so instead of calling every sign-in invalid. */
export class SetupError extends Error {}
export function sessionSecret() {
  const secret = process.env.SESSION_SECRET || "";
  if (secret.length < 32) throw new SetupError("SESSION_SECRET is missing or shorter than 32 characters");
  return secret;
}
export function annotator(req) {
  const given = req.headers["x-annotator-token"];
  if (typeof given === "string" && given.startsWith("s1.")) return verifySession(given, sessionSecret());
  if (!tokens) {
    try { tokens = parseTokens(process.env.ANNOTATOR_TOKENS || "{}"); }
    catch (err) { console.error("setup", err.message); throw new SetupError(err.message); }
  }
  return annotatorFor(tokens, given);
}

/** A claimed name's record ({salt, hash}), or null when the name is free or was reset. Like
 * labels, every write is a new file and the latest counts. */
export async function readAccount(name) {
  const blobs = await listAll(`accounts/${name}/`);
  if (!blobs.length) return null;
  const newest = blobs.reduce((a, b) => (Date.parse(b.uploadedAt) > Date.parse(a.uploadedAt) ? b : a));
  const r = await fetch(newest.url, { cache: "no-store" });
  if (!r.ok) throw new Error(`account read failed (${r.status})`);
  const rec = await r.json();
  return rec && rec.reset ? null : rec;
}

export async function writeAccount(name, record) {
  await put(`accounts/${name}/${Date.now()}.json`, JSON.stringify(record),
    { access: "public", addRandomSuffix: true, contentType: "application/json" });
}

/** Run a handler, turning a broken setup into a clear 500 rather than a crash. */
export function guarded(handler) {
  return async (req, res) => {
    try { return await handler(req, res); }
    catch (err) {
      if (err instanceof SetupError) return refuse(res, 500, "The site is not set up correctly. Tell the study owner.");
      console.error("unhandled", err && err.message);
      return refuse(res, 500, "Something went wrong. Try again in a moment.");
    }
  };
}

export function refuse(res, status, message) {
  res.setHeader("Cache-Control", "no-store");
  return res.status(status).json({ error: message });
}

async function listAll(prefix) {
  const blobs = [];
  let cursor;
  do {
    const page = await list({ prefix, cursor, limit: 1000 });
    blobs.push(...page.blobs);
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return blobs;
}

async function latest(prefix) {
  const blobs = await listAll(prefix);
  if (!blobs.length) return null;
  const newest = blobs.reduce((a, b) => (Date.parse(b.uploadedAt) > Date.parse(a.uploadedAt) ? b : a));
  const r = await fetch(newest.url, { cache: "no-store" });
  return r.ok ? r.json() : null;
}

/** The study data ({status, guide, items}) the owner seeded, kept in memory for a minute. */
export async function loadData() {
  if (dataCache.data && Date.now() - dataCache.at < 60_000) return dataCache.data;
  const data = await latest("config/items/");
  dataCache = { at: Date.now(), data };
  return data;
}

export async function seedData(data) {
  await put(`config/items/${Date.now()}.json`, JSON.stringify(data),
    { access: "public", addRandomSuffix: true, contentType: "application/json" });
  dataCache = { at: Date.now(), data };
}

/** annotator -> summary id -> labels, from each summary's latest save under `prefix`. */
export async function readLabels(prefix) {
  const urls = latestPerSummary(await listAll(prefix));
  const out = {};
  for (const [who, sids] of Object.entries(urls)) {
    out[who] = {};
    await Promise.all(Object.entries(sids).map(async ([sid, url]) => {
      const r = await fetch(url, { cache: "no-store" });
      if (r.ok) out[who][sid] = (await r.json()).items || {};
    }));
  }
  return out;
}

export async function saveLabels(who, sid, items) {
  await put(`labels/${who}/${sid}/${Date.now()}.json`,
    JSON.stringify({ items, saved_at: new Date().toISOString() }),
    { access: "public", addRandomSuffix: true, contentType: "application/json" });
}
