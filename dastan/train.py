#!/usr/bin/env python3
"""Train Dastan from the frame and write models/.

    python -m dastan.train --n-jobs 8

Everything is fitted here -- heads, band calibration, blend weights. Nothing is
loaded from a previous model. The split is chronological, always:

    train      all seasons through 2025-26 GW30      156,490 rows
    early stop 2025-26 GW31-38                         6,582 rows
    calibrate  the same slice, fitted not loaded
    blend      per-position weight fitted on the same slice by the ranking objective

Using the same held-out slice for early stopping, calibration and blending means
those numbers are in-sample for the fitting slice. That is why the accuracy report
scores separate chronological blocks instead of quoting anything from here -- see
docs/ACCURACY.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import data
from .model import POSITIONS, fit_position

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"

TRAIN_MAX_GW = 30
HOLDOUT = (31, 38)
HOLDOUT_SEASON = "2025-26"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    df = data.load()
    feats = data.shipped_features(df)
    ts = df["season"].eq(HOLDOUT_SEASON)
    hold = ts & df["gameweek"].between(*HOLDOUT)
    pool = df[~(ts & df["gameweek"].gt(TRAIN_MAX_GW))]
    train = pool[~hold.reindex(pool.index, fill_value=False)]
    val = df[hold]
    print(f"{len(feats)} features | train {len(train):,} | holdout {len(val):,}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    calibration, blend_w, per_pos = {}, {}, {}
    for pos in POSITIONS:
        models, cal, w = fit_position(train, val, feats, pos, args.n_jobs, args.seed)
        calibration[pos], blend_w[pos] = cal, w

        models["p60"].save_model(str(args.out / f"p60_{pos}.json"))
        models["non60"].save_model(str(args.out / f"non60_{pos}.json"))
        models["bucket"].save_model(str(args.out / f"bucket_{pos}.json"))
        for k, r in enumerate(models["bucketreg"]):
            r.save_model(str(args.out / f"bucketreg_{pos}_{k}.json"))
        models["direct"].save_model(str(args.out / f"direct_{pos}.json"))

        per_pos[pos] = {
            "train_rows": int((train["position"] == pos).sum()),
            "holdout_rows": int((val["position"] == pos).sum()),
            "blend_weight": w,
            "p60_best_iteration": int(getattr(models["p60"], "best_iteration", -1)),
        }
        print(f"  {pos}: 7 heads + direct saved | blend w={w} | "
              f"calibration gamma={cal['gamma'] if cal else None}", flush=True)

    (args.out / "feature_cols.json").write_text(json.dumps(feats, indent=2) + "\n")
    (args.out / "bucket_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (args.out / "blend.json").write_text(json.dumps({
        "per_position_direct_weight": blend_w,
        "formula": "xpts = (1 - w[pos]) * multibucket + w[pos] * direct",
        "output_clip_min": 0.0,
        "note": ("Fitted on the held-out slice by the ranking objective. GKP lands near "
                 "zero because the decomposition already handles goalkeepers well: their "
                 "points come from saves and clean sheets, both of which the band "
                 "structure captures directly."),
    }, indent=2) + "\n")
    (args.out / "train_metadata.json").write_text(json.dumps({
        "name": "Dastan",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": ("p60 gate -> calibrated 4-band mixture -> per-band regressors, "
                         "plus a per-position blended direct regressor"),
        "train": f"all seasons through {HOLDOUT_SEASON} GW{TRAIN_MAX_GW}",
        "holdout": f"{HOLDOUT_SEASON} GW{HOLDOUT[0]}-{HOLDOUT[1]}",
        "frame": "deadline-anchored: every fixture in a gameweek shares one history snapshot",
        "features": len(feats),
        "seasons": data.SEASONS,
        "rows": int(len(df)),
        "fitted_here": ["heads", "band_calibration", "blend_weights"],
        "per_position": per_pos,
    }, indent=2) + "\n")

    print(f"\nwrote {len(list(args.out.glob('*.json')))} artefacts to {args.out}")
    print(f"blend weights: {blend_w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
