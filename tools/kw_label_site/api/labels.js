// GET /api/labels -- this annotator's saved labels.
// PUT /api/labels {sid, items} -- save this annotator's labels for one summary.
import { validateLabels } from "../lib/core.js";
import { annotator, guarded, loadData, readLabels, refuse, saveLabels } from "../lib/server.js";

export default guarded(async function handler(req, res) {
  const who = annotator(req);
  if (!who) return refuse(res, 401, "Please sign in again.");
  res.setHeader("Cache-Control", "no-store");
  try {
    if (req.method === "GET") {
      const all = await readLabels(`labels/${who}/`);
      return res.status(200).json({ annotator: who, labels: all[who] || {} });
    }
    if (req.method === "PUT") {
      const data = await loadData();
      if (!data) return refuse(res, 503, "The study materials have not been loaded yet.");
      if (data.status === "closed") return refuse(res, 403, "The check is closed; labels can no longer be changed.");
      const body = req.body || {};
      if (JSON.stringify(body).length > 20000) return refuse(res, 413, "Too much data in one save.");
      const item = data.items.find((it) => it.id === body.sid);
      if (!item) return refuse(res, 400, "Unknown summary.");
      const checked = validateLabels(body.items, item);
      if (!checked.ok) return refuse(res, 400, checked.error);
      await saveLabels(who, item.id, checked.items);
      return res.status(200).json({ ok: true });
    }
    return refuse(res, 405, "Use GET or PUT.");
  } catch (err) {
    console.error("labels", req.method, who, err && err.message);
    return refuse(res, 503, "Could not reach the label store. Try again in a moment.");
  }
});
