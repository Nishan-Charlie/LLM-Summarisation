// POST /api/login {name, pin} -- sign in on the main page. The first sign-in with a new name
// claims it with that PIN; later sign-ins need the same PIN. Returns a session for the page.
// DELETE /api/login?name=<name> -- the study owner frees a name (admin token), e.g. after a
// forgotten PIN; that annotator's labels are kept and come back when the name is claimed again.
import { checkPin, hashPin, isAdmin, normaliseName, signSession, validPin } from "../lib/core.js";
import { guarded, readAccount, refuse, sessionSecret, writeAccount } from "../lib/server.js";

const pause = (ms) => new Promise((r) => setTimeout(r, ms));

export default guarded(async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  if (req.method === "DELETE") {
    if (!isAdmin(process.env.ADMIN_TOKEN, req.headers["x-admin-token"])) return refuse(res, 401, "Not allowed.");
    const name = normaliseName(String(req.query.name || ""));
    if (!name) return refuse(res, 400, "Give the name to free.");
    await writeAccount(name, { reset: true, at: new Date().toISOString() });
    return res.status(200).json({ ok: true, freed: name });
  }
  if (req.method !== "POST") return refuse(res, 405, "Use POST.");
  const secret = sessionSecret();
  const body = req.body || {};
  const name = normaliseName(body.name);
  const pin = typeof body.pin === "string" ? body.pin.trim() : "";
  if (!name) return refuse(res, 400, "Use 2 to 20 letters, digits or hyphens for your name.");
  if (!validPin(pin)) return refuse(res, 400, "Your PIN must be 4 to 8 digits.");
  try {
    const account = await readAccount(name);
    if (!account) {
      await writeAccount(name, { name, ...hashPin(pin), created_at: new Date().toISOString() });
      return res.status(200).json({ token: signSession(name, secret), name, created: true });
    }
    if (!checkPin(pin, account)) {
      await pause(700);  // slows guessing; every save is kept, so nothing can be lost silently
      return refuse(res, 401, "That PIN does not match this name. If you are new here, choose a different name.");
    }
    return res.status(200).json({ token: signSession(name, secret), name, created: false });
  } catch (err) {
    console.error("login", name, err && err.message);
    return refuse(res, 503, "Could not reach the sign-in store. Try again in a moment.");
  }
});
