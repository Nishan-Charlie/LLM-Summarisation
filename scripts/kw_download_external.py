"""Download the additional review and summarisation datasets into data/external/.

    python scripts/kw_download_external.py                 # everything available
    python scripts/kw_download_external.py cadec fewsum    # selected datasets

Each dataset goes to data/external/<name>/. data/external/MANIFEST.json records the
source URL, licence, download date, and the size and SHA-256 of every file, so the
paper can cite exactly what was used. Downloads resume after a dropped connection.
The folder is git-ignored: several licences forbid redistribution.

Not downloadable without your own account (see MANUAL below): WebMD (Kaggle) and
PsyTAR (askapatient.com store).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1] / "data" / "external"
UA = {"User-Agent": "Mozilla/5.0 (research dataset download; fairsum kwsum)"}

DATASETS = {
    "cadec": {
        "what": "CSIRO Adverse Drug Event Corpus v2: 1,250 AskaPatient posts with expert ADE spans "
                "(Karimi et al., J. Biomed. Inform. 2015)",
        "licence": "CSIRO Data Licence (research use; no redistribution)",
        "source": "https://data.csiro.au/collection/csiro:10948",
    },
    "fewsum": {
        "what": "FewSum gold summaries: Amazon and Yelp, 8 reviews + 3 human summaries per entity "
                "(Brazinskas et al., EMNLP 2020)",
        "licence": "see https://github.com/abrazinskas/FewSum (MIT code; research data)",
        "source": "https://github.com/abrazinskas/FewSum/tree/master/artifacts",
    },
    "meansum_yelp": {
        "what": "MeanSum Yelp reference summaries: 200 businesses, 8 reviews + 1 human summary "
                "(Chu & Liu, ICML 2019)",
        "licence": "research use; underlying reviews under the Yelp Dataset terms",
        "source": "https://s3.us-east-2.amazonaws.com/unsup-sum/summaries_0-200_cleaned.csv",
    },
    "space": {
        "what": "SPACE: 50 TripAdvisor hotels x 100 reviews, 3 general + 18 aspect summaries each "
                "(Angelidis et al., TACL 2021)",
        "licence": "research use (see https://github.com/stangelid/qt)",
        "source": "https://github.com/stangelid/qt",
    },
    "amasum": {
        "what": "AmaSum raw JSON, 10-100 reviews per product: 33,324 verdict/pros/cons summaries "
                "(Brazinskas et al., EMNLP 2021)",
        "licence": "non-commercial and educational use only (SelSum data/LICENSE.txt)",
        "source": "https://abrazinskas.s3.eu-west-1.amazonaws.com/downloads/projects/selsum/data/"
                  "raw_min_10_max_100_revs.zip",
    },
}

MANUAL = {
    "webmd": "Kaggle 'WebMD Drug Reviews Dataset' (rohanharode07/webmd-drug-reviews-dataset, ~362k reviews "
             "with age, sex, effectiveness, satisfaction). Needs a Kaggle account: download the zip and "
             "unpack it into data/external/webmd/.",
    "psytar": "PsyTAR (Zolnoori et al., JBI 2019), CC BY 4.0, distributed through the askapatient.com store "
              "('Download PsyTAR Data Set' on https://www.askapatient.com/research/pharmacovigilance/"
              "corpus-ades-psychiatric-medications.asp). Save PsyTAR_dataset.xlsx into data/external/psytar/.",
}


def fetch(url: str, dest: Path, params: dict | None = None, attempts: int = 30) -> Path:
    """Stream `url` to `dest`, resuming a partial .part file after dropped connections."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {**UA, **({"Range": f"bytes={have}-"} if have else {})}
        try:
            with requests.get(url, params=params, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:  # already complete
                    break
                if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                    raise RuntimeError(f"{url}: HTTP {r.status_code} (not retried)")
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(part, mode) as fh:
                    for block in r.iter_content(1 << 20):
                        fh.write(block)
            break
        except (requests.RequestException, OSError) as exc:
            print(f"  {dest.name}: attempt {attempt} failed ({type(exc).__name__}); retrying", flush=True)
            time.sleep(min(60, 5 * attempt))
    else:
        raise RuntimeError(f"gave up on {url}")
    part.replace(dest)
    return dest


def unzip(archive: Path, into: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():  # refuse paths that escape the target folder
            if not (into / member).resolve().is_relative_to(into.resolve()):
                raise ValueError(f"unsafe path in {archive.name}: {member}")
        z.extractall(into)


def get_cadec(out: Path) -> None:
    meta = requests.get("https://data.csiro.au/dap/ws/v2/collections/17190/data",
                        headers={**UA, "Accept": "application/json"}, timeout=60).json()
    entry = next(f for f in meta["file"] if f["filename"] == "CADEC.v2.zip")
    unzip(fetch(entry["presignedLink"]["href"], out / "CADEC.v2.zip"), out)


def get_fewsum(out: Path) -> None:
    for domain in ("amazon", "yelp"):
        listing = requests.get(f"https://api.github.com/repos/abrazinskas/FewSum/contents/artifacts/"
                               f"{domain}/gold_summs", headers=UA, timeout=60).json()
        for entry in listing:
            if entry["type"] == "file":
                fetch(entry["download_url"], out / domain / entry["name"])
    fetch("https://raw.githubusercontent.com/abrazinskas/FewSum/master/artifacts/README.md",
          out / "README.md")


def get_meansum_yelp(out: Path) -> None:
    fetch(DATASETS["meansum_yelp"]["source"], out / "summaries_0-200_cleaned.csv")


def get_space(out: Path) -> None:
    file_id = "1C6SaRQkas2B-9MolbwZbl0fuLgqdSKDT"
    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}
    probe = requests.get(url, params=params, headers=UA, timeout=60)
    if "text/html" in probe.headers.get("Content-Type", ""):  # large-file warning page
        uuid = re.search(r'name="uuid" value="([^"]+)"', probe.text)
        if uuid:
            params["uuid"] = uuid.group(1)
    archive = fetch(url, out / "space.tar.gz", params=params)  # 424 MB gzipped tar
    if not tarfile.is_tarfile(archive):
        archive.unlink()
        raise RuntimeError("Google Drive returned a page, not the archive; download SPACE by hand from "
                           "https://github.com/stangelid/qt")
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")  # refuses absolute paths and links outside `out`


def get_amasum(out: Path) -> None:
    unzip(fetch(DATASETS["amasum"]["source"], out / "raw_min_10_max_100_revs.zip"), out)


GETTERS = {"cadec": get_cadec, "fewsum": get_fewsum, "meansum_yelp": get_meansum_yelp,
           "space": get_space, "amasum": get_amasum}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    names = sys.argv[1:] or list(GETTERS)
    unknown = [n for n in names if n not in GETTERS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; available: {list(GETTERS)}; manual: {list(MANUAL)}")
    manifest_path = ROOT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    failed = []
    for name in names:
        out = ROOT / name
        print(f"== {name}: {DATASETS[name]['what']}", flush=True)
        try:
            GETTERS[name](out)
        except Exception as exc:  # one source failing must not stop the others
            print(f"   FAILED: {type(exc).__name__}: {exc}", flush=True)
            failed.append(name)
            continue
        files = sorted(p for p in out.rglob("*") if p.is_file() and not p.name.endswith(".part"))
        manifest[name] = {**DATASETS[name],
                          "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "files": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"),
                                     "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]}
        print(f"   ok: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    manifest["_manual"] = MANUAL
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("manual downloads still needed:", ", ".join(MANUAL))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
