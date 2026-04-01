#!/usr/bin/env python3
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
RESULT_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULT_DIR / "results.csv"

# Default metric file paths (expected locations from evaluation runs)
METRIC_FILES = [
    ROOT / "hydra_jobs" / "singlerun" / "CSP-carbon24-baseline-gpu0" / "eval_metrics_carbon24_baseline_gpu5_texttest.json",
    ROOT / "hydra_jobs" / "singlerun" / "CSP-carbon24-text-gpu1-offline-emb" / "eval_metrics_carbon24_text_gpu6_texttest.json",
    ROOT / "hydra_jobs" / "singlerun" / "CSP-perov5-baseline-gpu4" / "eval_metrics_perov5_baseline_gpu5_texttest.json",
    ROOT / "hydra_jobs" / "singlerun" / "CSP-perov5-text-gpu7-offline-emb" / "eval_metrics_perov5_text_gpu6_texttest.json",
    # mp_20 run dirs (baseline + text)
    ROOT / "hydra_jobs" / "singlerun" / "CSP-mp20-baseline-gpu2" / "eval_metrics_mp20_baseline_gpu5_texttest_csp1.json",
    ROOT / "hydra_jobs" / "singlerun" / "CSP-mp20-text-gpu3-offline-emb" / "eval_metrics_mp20_text_gpu6_texttest_csp1.json",
]

rows = []
for p in METRIC_FILES:
    rec = {
        "dataset": "",
        "model": "",
        "match_rate": "",
        "rms_dist": "",
        "metrics_file": str(p.relative_to(ROOT))
    }
    if not p.exists():
        print(f"Warning: metrics file not found: {p}")
        rows.append(rec)
        continue
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        print(f"Error reading {p}: {e}")
        rows.append(rec)
        continue

    # determine dataset & model from filename
    name = p.name
    if "carbon24" in name:
        rec["dataset"] = "carbon_24"
    elif "perov5" in name or "perov" in name:
        rec["dataset"] = "perov_5"
    else:
        rec["dataset"] = name

    if "baseline" in name:
        rec["model"] = "baseline"
    elif "text" in name or "offline" in name:
        rec["model"] = "text"
    else:
        rec["model"] = "unknown"

    # extract metrics
    match = data.get("match_rate") if isinstance(data, dict) else None
    rms = data.get("rms_dist") if isinstance(data, dict) else None

    rec["match_rate"] = match if match is not None else ""
    rec["rms_dist"] = rms if rms is not None else ""

    rows.append(rec)

# write CSV
with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset", "model", "match_rate", "rms_dist", "metrics_file"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote results to {OUT_CSV}")
