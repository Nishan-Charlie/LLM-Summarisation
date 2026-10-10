"""Write data/items.json for the labelling site: the blinded study items, the practice summary
and the labelling rules, verbatim. The site reads it after it is loaded through /api/seed.

    python scripts/kw_human_check.py web-items --config configs/kw_week1.toml   # first
    python tools/kw_label_site/build_data.py

Nothing here links an item to its model, prompt or imbalance level; that key stays in
results/kw/week1/human_check_key.json and is never deployed.
"""

import json
from pathlib import Path

from fairsum.kwsum.collections import read_collections
from fairsum.kwsum.config import COLLECTIONS_PATH, REPO_ROOT
from fairsum.kwsum.human_check import GUIDE
from fairsum.kwsum.propositions import split_summary

HERE = Path(__file__).resolve().parent
ITEMS = REPO_ROOT / "results" / "kw" / "week1" / "human_check_web_items.json"
GENERATIONS = REPO_ROOT / "results" / "kw" / "week1" / "generations.jsonl"
# The practice summary, as on the claude.ai labelling page: not part of the sample.
PRACTICE = {"collection_id": "drugs-test-pos88-000", "model": "llama3.2-3b", "condition": "kw_strat_neg",
            "propositions": (0, 2, 6, 7)}


def practice_item() -> dict:
    coll = next(c for c in read_collections(COLLECTIONS_PATH) if c.collection_id == PRACTICE["collection_id"])
    with GENERATIONS.open(encoding="utf-8") as fh:
        for line in fh:
            g = json.loads(line)
            if (g["collection_id"], g["model"], g["condition"]) == (coll.collection_id, PRACTICE["model"],
                                                                   PRACTICE["condition"]):
                props = split_summary(g["raw"]).propositions
                break
    sentences = [props[i] for i in PRACTICE["propositions"]]
    return {"id": "P1", "order": 0, "practice": True, "topic": coll.topic, "reviews": list(coll.texts),
            "sentences": [{"item_id": f"P1-{k}", "text": t} for k, t in enumerate(sentences, start=1)]}


def main() -> None:
    study = json.loads(ITEMS.read_text(encoding="utf-8"))["items"]
    practice = practice_item()
    if any(s["topic"] == practice["topic"] and s["reviews"] == practice["reviews"] for s in study):
        raise SystemExit("the practice collection is in the study sample; pick another practice item")
    data = {"status": "open", "guide": GUIDE[GUIDE.index("claim"):], "items": [practice, *study]}
    out = HERE / "data" / "items.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")  # loaded through /api/seed
    print(f"{len(study)} study summaries + practice -> {out}")


if __name__ == "__main__":
    main()
