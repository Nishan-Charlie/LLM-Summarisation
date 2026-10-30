// GET /api/b/labels -- this annotator's saved Part B answers.
// PUT /api/b/labels {sid, labels} -- save this annotator's answers for one set. The last save is
// read first, so the overall feeling (step 1) cannot change once the reviews are open.
import { itemsFor, validateLabelsB } from "../../lib/core.js";
import { PARTB, annotator, guarded, loadData, readLabels, refuse, saveLabels } from "../../lib/server.js";

export default guarded(async function handler(req, res) {
  const who = annotator(req);
  if (!who) return refuse(res, 401, "Please sign in again.");
  res.setHeader("Cache-Control", "no-store");
  try {
    if (req.method === "GET") {
      const all = await readLabels(`${PARTB.root}/${who}/`, PARTB.root, PARTB.field);
      return res.status(200).json({ annotator: who, labels: all[who] || {} });
    }
    if (req.method === "PUT") {
      const data = await loadData(PARTB.config);
      if (!data) return refuse(res, 503, "Part B has not been loaded yet.");
      if (data.status === "closed") return refuse(res, 403, "Part B is closed; answers can no longer be changed.");
      const body = req.body || {};
      if (JSON.stringify(body).length > 8000) return refuse(res, 413, "Too much data in one save.");
      const item = (itemsFor(data, who) || []).find((it) => it.id === body.sid);
      if (!item) return refuse(res, 403, "This set is not one of yours.");
      const before = await readLabels(`${PARTB.root}/${who}/${item.id}/`, PARTB.root, PARTB.field);
      const checked = validateLabelsB(body.labels, (before[who] || {})[item.id] || null);
      if (!checked.ok) return refuse(res, 400, checked.error);
      await saveLabels(who, item.id, checked.labels, PARTB.root, PARTB.field);
      return res.status(200).json({ ok: true });
    }
    return refuse(res, 405, "Use GET or PUT.");
  } catch (err) {
    console.error("b/labels", req.method, who, err && err.message);
    return refuse(res, 503, "Could not reach the label store. Try again in a moment.");
  }
});
