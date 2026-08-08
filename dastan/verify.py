#!/usr/bin/env python3
"""Reload the released weights from disk and score them.

    python -m dastan.verify

Training scripts report numbers from objects still in memory. This reloads all 37
files the way an application would, so a model that trains correctly but serialises
wrongly fails loudly rather than silently.

Scored on the fitting holdout, so these numbers check that the artefacts *reproduce*
-- not that the model generalises. Generalisation numbers are in docs/ACCURACY.md and
come from separate chronological blocks.
"""

from __future__ import annotations

import argparse

from . import data
from .metrics import auc, brier, collapse_to_player_gameweek, score
from .predictor import Dastan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--gw", nargs=2, type=int, default=[31, 38])
    args = ap.parse_args()

    df = data.load()
    m = Dastan()
    val = df[df["season"].eq(args.season) & df["gameweek"].between(*args.gw)]
    print(f"{len(m.features)} features | {len(val):,} rows "
          f"({args.season} GW{args.gw[0]}-{args.gw[1]})\n")

    out = m.predict_frame(val, with_parts=True)
    neg = int((out["xpts"] < 0).sum())
    print(f"negative predictions: {neg}  (must be 0)")

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

    if neg:
        raise SystemExit("negative xpts from reloaded artefacts")
    print("\nartefacts reload and predict correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
