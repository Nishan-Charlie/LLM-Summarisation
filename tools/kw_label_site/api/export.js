// GET /api/export -- every annotator's latest labels, for the study owner only (admin token).
// The response is the {"annotators": {name: [per-summary documents]}} file that
// `scripts/kw_human_check.py from-web --labels` reads.
import { isAdmin } from "../lib/core.js";
import { guarded, loadData, readLabels, refuse } from "../lib/server.js";

export default guarded(async function handler(req, res) {
  if (req.method !== "GET") return refuse(res, 405, "Use GET.");
  if (!isAdmin(process.env.ADMIN_TOKEN, req.headers["x-admin-token"])) return refuse(res, 401, "Not allowed.");
  res.setHeader("Cache-Control", "no-store");
  try {
    const [all, data] = await Promise.all([readLabels("labels/"), loadData()]);
    const study = (data ? data.items : []).filter((it) => !it.practice);
    const total = study.reduce((n, it) => n + it.sentences.length, 0);
    const done = (labels) => study.reduce((n, it) => n + it.sentences.filter((s) => {
      const l = (labels[it.id] || {})[s.item_id];
      return l && (l.claim === "no" || (l.claim === "yes" && l.sentiment && l.support));
    }).length, 0);
    return res.status(200).json({
      exported_at: new Date().toISOString(),
      progress: Object.fromEntries(Object.entries(all).map(([who, labels]) => [who, `${done(labels)} of ${total}`])),
      annotators: Object.fromEntries(Object.entries(all).map(([who, labels]) =>
        [who, Object.entries(labels).map(([sid, items]) => ({ id: sid, items }))])),
    });
  } catch (err) {
    console.error("export", err && err.message);
    return refuse(res, 503, "Could not reach the label store.");
  }
});
