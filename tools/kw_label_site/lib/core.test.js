// Tests for the labelling site's server-side rules: who is an annotator, which labels are
// accepted, and which save of a summary counts. Run: node --test (from tools/kw_label_site).
import { test } from "node:test";
import assert from "node:assert/strict";
import { annotatorFor, isAdmin, latestPerSummary, parseTokens, validateData, validateLabels } from "./core.js";

const TOKENS = parseTokens(JSON.stringify({ "tok-nish-0123456789abcdef": "nishankar", "tok-prat-0123456789abcdef": "prathyuman" }));

test("a token names its annotator; anything else names no one", () => {
  assert.equal(annotatorFor(TOKENS, "tok-nish-0123456789abcdef"), "nishankar");
  assert.equal(annotatorFor(TOKENS, "tok-nish-0123456789abcdeX"), null);
  assert.equal(annotatorFor(TOKENS, ""), null);
  assert.equal(annotatorFor(TOKENS, undefined), null);
});

test("token lists with unsafe annotator names or short tokens are refused", () => {
  assert.throws(() => parseTokens(JSON.stringify({ "tok-0123456789abcdef": "Bad Name" })), /annotator name/);
  assert.throws(() => parseTokens(JSON.stringify({ short: "nishankar" })), /token/);
  assert.throws(() => parseTokens("not json"), /JSON/);
});

test("the admin check needs the exact admin token", () => {
  assert.equal(isAdmin("admin-0123456789abcdef", "admin-0123456789abcdef"), true);
  assert.equal(isAdmin("admin-0123456789abcdef", "admin-0123456789abcdeg"), false);
  assert.equal(isAdmin("admin-0123456789abcdef", undefined), false);
  assert.equal(isAdmin("", ""), false); // an unset admin token never matches
});

const ITEM = { id: "S07", sentences: [{ item_id: "S07-1" }, { item_id: "S07-2" }] };

test("valid labels are kept and normalised; unknown items and fields are dropped", () => {
  const out = validateLabels({
    "S07-1": { claim: "Yes", sentiment: " positive", support: "SUPPORTED", notes: "R3", extra: "x" },
    "S07-2": { claim: "no", sentiment: "negative", support: "supported" },
    "S99-1": { claim: "yes", sentiment: "positive", support: "supported" },
  }, ITEM);
  assert.equal(out.ok, true);
  assert.deepEqual(out.items, {
    "S07-1": { claim: "yes", sentiment: "positive", support: "supported", notes: "R3" },
    "S07-2": { claim: "no", sentiment: null, support: null, notes: "" },   // claim = no clears the rest
  });
});

test("unknown label values and oversized notes are refused, partial labels are allowed", () => {
  assert.equal(validateLabels({ "S07-1": { claim: "maybe" } }, ITEM).ok, false);
  assert.equal(validateLabels({ "S07-1": { claim: "yes", sentiment: "happy" } }, ITEM).ok, false);
  assert.equal(validateLabels({ "S07-1": { claim: "yes", notes: "x".repeat(501) } }, ITEM).ok, false);
  assert.equal(validateLabels("not an object", ITEM).ok, false);
  const partial = validateLabels({ "S07-1": { claim: "yes", sentiment: "mixed" } }, ITEM);
  assert.equal(partial.ok, true);
  assert.deepEqual(partial.items["S07-1"], { claim: "yes", sentiment: "mixed", support: null, notes: "" });
});

test("the latest save of each summary counts", () => {
  const blobs = [
    { pathname: "labels/nishankar/S07/a.json", uploadedAt: "2026-10-02T05:00:00.000Z", url: "u1" },
    { pathname: "labels/nishankar/S07/b.json", uploadedAt: "2026-10-02T05:03:00.000Z", url: "u2" },
    { pathname: "labels/nishankar/S01/c.json", uploadedAt: "2026-10-02T04:00:00.000Z", url: "u3" },
    { pathname: "labels/prathyuman/S07/d.json", uploadedAt: "2026-10-02T06:00:00.000Z", url: "u4" },
    { pathname: "something/else.json", uploadedAt: "2026-10-02T07:00:00.000Z", url: "u5" },
  ];
  assert.deepEqual(latestPerSummary(blobs), {
    nishankar: { S07: "u2", S01: "u3" },
    prathyuman: { S07: "u4" },
  });
});

const ITEMS = { status: "open", guide: ["claim", "  yes: ..."], items: [
  { id: "P1", practice: true, topic: "Singulair", reviews: ["r1", "r2"], sentences: [{ item_id: "P1-1", text: "s" }] },
  { id: "S01", topic: "Lumia", reviews: ["a", "b"], sentences: [{ item_id: "S01-1", text: "t" }] },
] };

test("seeded study data must have the shape the page and the label check rely on", () => {
  assert.equal(validateData(ITEMS).ok, true);
  assert.equal(validateData({ ...ITEMS, status: "maybe" }).ok, false);
  assert.equal(validateData({ ...ITEMS, items: [] }).ok, false);
  assert.equal(validateData({ ...ITEMS, items: [{ id: "S01", topic: "x", reviews: [], sentences: [] }] }).ok, false);
  const wrongId = structuredClone(ITEMS); wrongId.items[1].sentences[0].item_id = "S02-1";
  assert.equal(validateData(wrongId).ok, false);  // item ids must belong to their summary
  const dup = structuredClone(ITEMS); dup.items.push(structuredClone(dup.items[1]));
  assert.equal(validateData(dup).ok, false);
});

// --- Sign-in by name and PIN on the main page ----------------------------------------------
import { checkPin, hashPin, normaliseName, signSession, validPin, verifySession } from "./core.js";

const SECRET = "s".repeat(40);

test("names are case-insensitive, trimmed, and limited to safe characters", () => {
  assert.equal(normaliseName("  Nishankar "), "nishankar");
  assert.equal(normaliseName("Anna Lee"), "anna-lee");
  assert.equal(normaliseName("a"), null);
  assert.equal(normaliseName("x".repeat(21)), null);
  assert.equal(normaliseName("bad/name"), null);
  assert.equal(normaliseName(undefined), null);
});

test("a PIN is 4 to 8 digits", () => {
  assert.ok(validPin("2468") && validPin("12345678"));
  assert.ok(!validPin("123") && !validPin("123456789") && !validPin("12a4") && !validPin(1234));
});

test("a stored PIN hash accepts the same PIN and nothing else", () => {
  const account = hashPin("2468", "salt-0123456789");
  assert.notEqual(account.hash, "2468");
  assert.ok(checkPin("2468", account));
  assert.ok(!checkPin("2469", account));
  assert.ok(!checkPin("2468", { salt: account.salt, hash: "00" }));
});

test("a session names its annotator until it expires, and a changed one names no one", () => {
  const now = Date.parse("2026-10-03T12:00:00Z");
  const token = signSession("nishankar", SECRET, now);
  assert.equal(verifySession(token, SECRET, now + 1000), "nishankar");
  assert.equal(verifySession(token, SECRET, now + 61 * 864e5), null);           // expired
  assert.equal(verifySession(token.replace("nishankar", "prathyuman"), SECRET, now), null);
  assert.equal(verifySession(token, "t".repeat(40), now), null);                // other secret
  assert.equal(verifySession("tok-nish-0123456789abcdef", SECRET, now), null);   // not a session
});
