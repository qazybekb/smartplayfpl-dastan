#!/usr/bin/env python3
"""Train Dastan from an accepted annual or mid-season frame.

    python -m dastan.train --n-jobs 8

Everything is fitted here -- heads, band/minutes calibration, and blend weights.
Nothing is loaded from a previous model. The versioned season registry selects a
chronological training window and a later holdout. That same holdout is used for
early stopping, calibration, and fitting the per-position blend weights.

Using the same held-out slice for early stopping, calibration and blending means
those numbers are in-sample for the fitting slice. That is why the accuracy report
scores separate chronological blocks instead of quoting anything from here -- see
docs/ACCURACY.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, seasons
from .minutes import build_payload as build_minutes_payload
from .minutes import rows_from_artifacts as minutes_rows_from_artifacts
from .model import POSITIONS, fit_position

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"

_DEFAULT_PLAN = seasons.published_plan()
TRAIN_MAX_GW = _DEFAULT_PLAN.train_max_gw
HOLDOUT = _DEFAULT_PLAN.holdout
HOLDOUT_SEASON = _DEFAULT_PLAN.data_season


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--data-dir", type=Path, default=data.DATA)
    ap.add_argument("--mode", choices=("annual", "midseason"))
    ap.add_argument("--season")
    ap.add_argument("--through-gw", type=int)
    ap.add_argument(
        "--accepted-manifest",
        type=Path,
        help="required for a candidate frame; produced by python -m dastan.release accept",
    )
    args = ap.parse_args()

    try:
        plan = (
            seasons.published_plan()
            if args.mode is None and args.season is None and args.through_gw is None
            else seasons.make_plan(
                args.mode or "annual", season=args.season, through_gw=args.through_gw
            )
        )
    except ValueError as exc:
        ap.error(str(exc))
    release_data = args.data_dir.resolve() == data.DATA.resolve()
    if not release_data or plan != _DEFAULT_PLAN:
        if args.accepted_manifest is None:
            ap.error("candidate/mid-season training requires --accepted-manifest")
        from .release import verify_accepted_manifest

        verify_accepted_manifest(args.accepted_manifest, args.data_dir, plan)

    df = data.load(data_dir=args.data_dir, check_rows=release_data)
    feats = data.shipped_features(df)
    missing_seasons = sorted(set(plan.seasons) - set(df["season"].astype(str)))
    if missing_seasons:
        raise RuntimeError(f"training frame is missing plan seasons: {missing_seasons}")
    active = df[df["season"].eq(plan.data_season)]
    max_gw = int(active["gameweek"].max()) if not active.empty else 0
    if max_gw != plan.available_through_gw:
        raise RuntimeError(
            f"{plan.data_season} must end at GW{plan.available_through_gw}, got GW{max_gw}"
        )
    ts = df["season"].eq(plan.data_season)
    hold = ts & df["gameweek"].between(*plan.holdout)
    pool = df[~(ts & df["gameweek"].gt(plan.train_max_gw))]
    train = pool[~hold.reindex(pool.index, fill_value=False)]
    val = df[hold]
    if train.empty or val.empty:
        raise RuntimeError(f"empty training split: train={len(train)}, holdout={len(val)}")
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
    (args.out / "minutes_calibration.json").write_text(
        json.dumps(
            build_minutes_payload(
                minutes_rows_from_artifacts(args.out, val, feats),
                f"{plan.data_season} GW{plan.holdout[0]}-{plan.holdout[1]} packaging holdout",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    (args.out / "blend.json").write_text(json.dumps({
        "per_position_direct_weight": blend_w,
        "formula": "xpts = (1 - w[pos]) * multibucket + w[pos] * direct",
        "output_clip_min": 0.0,
        "note": ("Fitted on the held-out slice by the ranking objective. "
                 f"Learned GKP direct weight: {blend_w['GKP']}."),
    }, indent=2) + "\n")
    (args.out / "train_metadata.json").write_text(json.dumps({
        "name": "Dastan",
        "version": "1.1.0",
        "model_version": "dastan",
        "target_season": plan.target_season,
        "public_name": "Dastan",
        "public_repo": "https://github.com/qazybekb/smartplayfpl-dastan",
        "scenario": {"base_model": "dastan_production"},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "architecture": ("p60 gate -> calibrated 4-band mixture -> per-band regressors, "
                         "plus a per-position blended direct regressor"),
        "retraining_plan": plan.to_dict(),
        "release_kind": plan.mode,
        "train": f"all configured seasons through {plan.data_season} GW{plan.train_max_gw}",
        "holdout": f"{plan.data_season} GW{plan.holdout[0]}-{plan.holdout[1]}",
        "frame": "deadline-anchored: every fixture in a gameweek shares one history snapshot",
        "features": len(feats),
        "seasons": list(plan.seasons),
        "rows": int(len(df)),
        "fitted_here": [
            "heads",
            "bucket_calibration",
            "minutes_calibration",
            "blend_weights",
        ],
        "per_position": per_pos,
    }, indent=2) + "\n")

    for name in ("core_feature_cols.json", "hyperparameters.json", "tuned_params.json"):
        source = ROOT / "models" / name
        destination = args.out / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    print(f"\nwrote {len(list(args.out.glob('*.json')))} artifacts to {args.out}")
    print(f"blend weights: {blend_w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
