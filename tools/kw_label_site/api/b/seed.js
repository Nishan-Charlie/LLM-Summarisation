// PUT /api/b/seed -- the study owner loads or replaces Part B's data (admin token):
//   curl -X PUT -H "x-admin-token: $ADMIN_TOKEN" -H "content-type: application/json" \
//        --data-binary @data/partb.json https://<site>/api/b/seed
import { isAdmin, validateDataB } from "../../lib/core.js";
import { PARTB, guarded, refuse, seedData } from "../../lib/server.js";

export default guarded(async function handler(req, res) {
  if (req.method !== "PUT") return refuse(res, 405, "Use PUT.");
  if (!isAdmin(process.env.ADMIN_TOKEN, req.headers["x-admin-token"])) return refuse(res, 401, "Not allowed.");
  const checked = validateDataB(req.body);
  if (!checked.ok) return refuse(res, 400, checked.error);
  try {
    await seedData(req.body, PARTB.config);
    const sets = req.body.items.filter((it) => !it.practice).length;
    return res.status(200).json({ ok: true, status: req.body.status, sets, annotators: Object.keys(req.body.assign) });
  } catch (err) {
    console.error("b/seed", err && err.message);
    return refuse(res, 503, "Could not store the study data.");
  }
});
