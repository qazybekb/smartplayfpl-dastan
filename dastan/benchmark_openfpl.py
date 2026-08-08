#!/usr/bin/env python3
"""Head-to-head against OpenFPL, on identical rows.

    python -m dastan.benchmark_openfpl --n-jobs 8

OpenFPL (Groos & Zhang, arXiv:2508.09992, https://github.com/daniegr/OpenFPL) is the
work Dastan's feature engineering is built on, and the fairest reference point for it.
Its stored predictions cover 2025-26 GW1-24 under a published regime:

    train        2020-21 .. 2023-24
    early stop   2024-25
    test         2025-26 GW1-24        18,173 player-fixtures

Dastan is retrained here under exactly that regime rather than scored from the
released weights, because the released weights trained through 2025-26 GW30 and would
be predicting a window they were fitted on. Both models are then scored on the joined
frame, so neither is evaluated on rows the other did not see.

Note on `ep_next`: the FPL projection stored alongside OpenFPL's predictions is *not*
provenance-checked -- some of it was captured after the deadline it describes, which
makes its top-10 metric look far better than it is. Use the provenance-checked
`ep_next` in `data/pre_deadline_ep_next.parquet` for any honest FPL comparison; that
is what docs/ACCURACY.md reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .metrics import ndcg, score, spearman
from .model import POSITIONS, fit_position, predict

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "openfpl_predictions.csv"
BRIDGE = ROOT / "data" / "openfpl_row_keys.csv"

TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
VALID_SEASON = "2024-25"
TEST_SEASON, TEST_GWS = "2025-26", (1, 24)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "openfpl_benchmark.json")
    args = ap.parse_args()

    df = data.load()
    feats = data.shipped_features(df)
    train = df[df["season"].isin(TRAIN_SEASONS)]
    valid = df[df["season"].eq(VALID_SEASON)]
    test = df[df["season"].eq(TEST_SEASON) & df["gameweek"].between(*TEST_GWS)]
    print(f"train {len(train):,} | early stop {len(valid):,} | test {len(test):,}\n")

    frames = []
    for pos in POSITIONS:
        t = test[test["position"].eq(pos)]
        if t.empty:
            continue
        print(f"  fitting {pos}...", flush=True)
        models, cal, w = fit_position(train, valid, feats, pos, args.n_jobs, args.seed)
        frames.append(pd.DataFrame({
            "season": t["season"].to_numpy(), "gameweek": t["gameweek"].to_numpy(),
            "fixture": t["fixture"].to_numpy(), "fpl_code": t["fpl_code"].to_numpy(),
            "dastan": predict(models, cal, w, t[feats].fillna(0.0).to_numpy()),
            "actual": t["target_points"].to_numpy(),
            "minutes": t["minutes"].fillna(0).to_numpy()}))
    ours = pd.concat(frames, ignore_index=True)

    keys = pd.read_csv(BRIDGE)
    stored = pd.read_csv(BENCH).rename(columns={"gw": "gameweek", "openfpl_xpts": "openfpl"})
    stored = keys.merge(stored[["element", "gameweek", "openfpl"]],
                        on=["element", "gameweek"], validate="one_to_one")
    joined = ours.merge(stored[["season", "gameweek", "fixture", "fpl_code", "openfpl"]],
                        on=["season", "gameweek", "fixture", "fpl_code"], validate="one_to_one")
    print(f"\njoined on identical rows: {len(joined):,}\n")

    pg = joined.groupby(["season", "gameweek", "fpl_code"], as_index=False).agg(
        dastan=("dastan", "sum"), openfpl=("openfpl", "sum"),
        actual=("actual", "sum"), minutes=("minutes", "sum"))

    report = {"regime": {"train": TRAIN_SEASONS, "early_stopping": VALID_SEASON,
                         "test": f"{TEST_SEASON} GW{TEST_GWS[0]}-{TEST_GWS[1]}",
                         "rows": int(len(joined)),
                         "player_gameweeks": int(len(pg))},
              "results": {}}
    print(f"{'model':10} {'cohort':10} {'obj':>8} {'spearman':>9} {'ndcg@10':>8} "
          f"{'mae':>7} {'rmse':>7}")
    for m in ("dastan", "openfpl"):
        report["results"][m] = {}
        for cohort in ("all", "starters"):
            s = score(pg.rename(columns={m: "pred"}), "pred", cohort)
            report["results"][m][cohort] = s
            print(f"{m:10} {cohort:10} {s['obj']:>8.4f} {s['spearman']:>9.4f} "
                  f"{s['ndcg@10']:>8.4f} {s['mae']:>7.3f} {s['rmse']:>7.3f}")

    for cohort in ("all", "starters"):
        d = report["results"]["dastan"][cohort]
        o = report["results"]["openfpl"][cohort]
        report.setdefault("delta", {})[cohort] = {
            k: round(d[k] - o[k], 4) for k in ("obj", "spearman", "ndcg@10", "mae", "rmse")}
    print("\ndelta (Dastan - OpenFPL):")
    for cohort, dd in report["delta"].items():
        print(f"  {cohort:10} obj {dd['obj']:+.4f}  spearman {dd['spearman']:+.4f}  "
              f"mae {dd['mae']:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
