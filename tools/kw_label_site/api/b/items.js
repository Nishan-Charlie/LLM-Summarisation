// GET /api/b/items -- Part B: the sets this annotator labels (the practice set plus their groups).
import { itemsFor } from "../../lib/core.js";
import { PARTB, annotator, guarded, loadData, refuse } from "../../lib/server.js";

export default guarded(async function handler(req, res) {
  if (req.method !== "GET") return refuse(res, 405, "Use GET.");
  const who = annotator(req);
  if (!who) return refuse(res, 401, "Please sign in again.");
  try {
    const data = await loadData(PARTB.config);
    if (!data) return refuse(res, 503, "Part B has not been loaded yet. Try again later.");
    const items = itemsFor(data, who);
    res.setHeader("Cache-Control", "no-store");
    return res.status(200).json({ annotator: who, status: data.status, guide: data.guide,
      assigned: items !== null, items: items || [] });
  } catch (err) {
    console.error("b/items", err && err.message);
    return refuse(res, 503, "Could not load the study materials. Try again in a moment.");
  }
});
