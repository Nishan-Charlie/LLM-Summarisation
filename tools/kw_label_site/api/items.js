// GET /api/items -- the study materials, for signed-in annotators (or an older personal link).
import { annotator, guarded, loadData, refuse } from "../lib/server.js";

export default guarded(async function handler(req, res) {
  if (req.method !== "GET") return refuse(res, 405, "Use GET.");
  const who = annotator(req);
  if (!who) return refuse(res, 401, "Please sign in again.");
  try {
    const data = await loadData();
    if (!data) return refuse(res, 503, "The study materials have not been loaded yet. Try again later.");
    res.setHeader("Cache-Control", "no-store");
    return res.status(200).json({ annotator: who, status: data.status, guide: data.guide, items: data.items });
  } catch (err) {
    console.error("items", err && err.message);
    return refuse(res, 503, "Could not load the study materials. Try again in a moment.");
  }
});
