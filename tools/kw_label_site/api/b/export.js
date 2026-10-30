// GET /api/b/export -- every annotator's latest Part B answers, for the study owner only (admin
// token). The response is the file `scripts/kw_human_partb.py check --labels` reads.
import { completeB, isAdmin, itemsFor } from "../../lib/core.js";
import { PARTB, guarded, loadData, readLabels, refuse } from "../../lib/server.js";

export default guarded(async function handler(req, res) {
  if (req.method !== "GET") return refuse(res, 405, "Use GET.");
  if (!isAdmin(process.env.ADMIN_TOKEN, req.headers["x-admin-token"])) return refuse(res, 401, "Not allowed.");
  res.setHeader("Cache-Control", "no-store");
  try {
    const [all, data] = await Promise.all([readLabels(`${PARTB.root}/`, PARTB.root, PARTB.field), loadData(PARTB.config)]);
    const assign = (data && data.assign) || {};
    const progress = {};
    for (const who of new Set([...Object.keys(assign), ...Object.keys(all)])) {
      const mine = (itemsFor(data, who) || []).filter((it) => !it.practice);
      progress[who] = `${mine.filter((it) => completeB((all[who] || {})[it.id])).length} of ${mine.length} sets`;
    }
    return res.status(200).json({
      exported_at: new Date().toISOString(),
      status: data ? data.status : null,
      assign,
      progress,
      annotators: Object.fromEntries(Object.entries(all).map(([who, sets]) =>
        [who, Object.entries(sets).map(([sid, labels]) => ({ id: sid, labels }))])),
    });
  } catch (err) {
    console.error("b/export", err && err.message);
    return refuse(res, 503, "Could not reach the label store.");
  }
});
