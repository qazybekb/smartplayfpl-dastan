#!/usr/bin/env python3
"""Reload the released weights from disk and score them.

    python -m dastan.verify

Training scripts report numbers from objects still in memory. This reloads all 36
inference artifacts from the 40-file published model contract the way an application
would, so a model that trains correctly but serialises wrongly fails loudly rather
than silently.

Scored on the fitting holdout, so these numbers check that the artefacts *reproduce*
-- not that the model generalises. Generalisation numbers are in docs/ACCURACY.md and
come from separate chronological blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import artifacts, data, seasons
from .metrics import auc, brier, collapse_to_player_gameweek, score
from .minutes import build_payload as build_minutes_payload
from .minutes import payload_equivalent
from .predictor import Dastan


def main() -> int:
    published = seasons.published_plan()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=published.data_season)
    ap.add_argument("--gw", nargs=2, type=int, default=list(published.holdout))
    ap.add_argument("--model-dir", type=Path, default=artifacts.MODELS)
    ap.add_argument("--data-dir", type=Path, default=data.DATA)
    args = ap.parse_args()

    released_models = args.model_dir.resolve() == artifacts.MODELS.resolve()
    released_data = args.data_dir.resolve() == data.DATA.resolve()
    contract = (
        artifacts.verify_manifest()
        if released_models
        else artifacts.verify_directory(args.model_dir)
    )
    df = data.load(data_dir=args.data_dir, check_rows=released_data)
    m = Dastan(args.model_dir)
    val = df[df["season"].eq(args.season) & df["gameweek"].between(*args.gw)]
    print(f"{len(m.features)} features | {len(val):,} rows "
          f"({args.season} GW{args.gw[0]}-{args.gw[1]})\n")

    out = m.predict_frame(val, with_parts=True)
    released_minutes = json.loads(
        (args.model_dir / "minutes_calibration.json").read_text(encoding="utf-8")
    )
    expected_minutes = build_minutes_payload(
        out[["position", "p60", "minutes"]],
        f"{args.season} GW{args.gw[0]}-{args.gw[1]} packaging holdout",
    )
    calibration_fresh = payload_equivalent(expected_minutes, released_minutes)
    neg = int((out["xpts"] < 0).sum())
    print(f"negative predictions: {neg}  (must be 0)")
    finite = np.isfinite(
        out[["xpts", "p60", "p_any", "expected_minutes"]].to_numpy(dtype=float)
    ).all()
    coherent = bool(
        out["p_any"].ge(out["p60"] - 1e-9).all()
        and out["expected_minutes"].ge(60.0 * out["p60"] - 1e-6).all()
        and out["p_any"].ge(out["expected_minutes"] / 90.0 - 1e-9).all()
        and out["expected_minutes"].between(0.0, 90.0).all()
        and out["p_any"].between(0.0, 1.0).all()
    )
    print(
        f"minutes outputs: finite={finite} coherent={coherent} "
        f"calibration_fresh={calibration_fresh}"
    )

    pf = collapse_to_player_gameweek(
        out.assign(pred=out["xpts"], actual=out["target_points"]), "pred")
    p60 = out.groupby(["season", "gameweek", "fpl_code"], as_index=False).agg(
        p60=("p60", "mean"), minutes=("minutes", "sum"))
    y = (p60["minutes"].fillna(0) >= 60).astype(int)

    for cohort in ("all", "starters"):
        s = score(pf, "pred", cohort)
        print(f"  {cohort:9} obj {s['obj']:.4f}  spearman {s['spearman']:.4f}  "
              f"ndcg@10 {s['ndcg@10']:.4f}  mae {s['mae']:.3f}")
    print(f"  p60 gate  AUC {auc(y, p60['p60']):.4f}  Brier {brier(y, p60['p60']):.4f}")

    if neg or not finite or not coherent or not calibration_fresh:
        raise SystemExit("invalid output from reloaded Dastan artifacts")
    print(
        f"\n{contract['files']}-file contract reloads and predicts correctly "
        f"for target {contract['target_season']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
