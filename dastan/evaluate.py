#!/usr/bin/env python3
"""Walk-forward evaluation against baselines.

    python -m dastan.evaluate --seeds 3

Retrains from scratch for each chronological block, so nothing in a block's test
window was available when its model was fitted. This is slower than scoring the
released weights everywhere, and it is the only version worth quoting: the released
weights were fitted with a holdout that overlaps some of these windows.

Baselines are computed on the same rows:

  FPL ep_next       FPL's own published projection, captured before the deadline.
                    The most meaningful external benchmark, since it is what a
                    manager sees for free in the app.
  last GW points    what the player scored last week.
  rolling-5 points  mean points over the last five appearances -- "form".
  price             current cost. Included because it is a genuinely strong
                    ranking signal: the market prices players roughly by expectation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .metrics import auc, brier, collapse_to_player_gameweek, score
from .model import POSITIONS, fit_position, predict

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 7, 2026)

# (label, season, last training GW, test range, early-stopping range)
BLOCKS = [
    ("2024-25 GW15-22", "2024-25", 14, (15, 22), (11, 14), "clean"),
    ("2024-25 GW23-30", "2024-25", 22, (23, 30), (19, 22), "clean"),
    ("2024-25 GW31-38", "2024-25", 30, (31, 38), (27, 30), "clean"),
    ("2025-26 GW23-30", "2025-26", 22, (23, 30), (19, 22), "discovery"),
    ("2025-26 GW31-38", "2025-26", 30, (31, 38), (27, 30), "discovery"),
]
BASELINES = {"FPL ep_next": "ep_next", "last GW points": "player_fpl_points_1",
             "rolling-5 points": "player_fpl_points_5", "price": "value"}
BASELINE_KIND = {name: "rank_only" if name == "price" else "point_forecast"
                 for name in BASELINES}
CORE_FEATURES = ROOT / "models" / "core_feature_cols.json"
DATA_MANIFEST = ROOT / "data" / "release_manifest.json"
MODEL_MANIFEST = ROOT / "models" / "artifact_manifest.json"
EVIDENCE_CODE = {
    "evaluate": Path(__file__).resolve(),
    "metrics": ROOT / "dastan" / "metrics.py",
    "model": ROOT / "dastan" / "model.py",
    "data": ROOT / "dastan" / "data.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_score(frame: pd.DataFrame, column: str, cohort: str, kind: str) -> dict:
    result = score(frame, column, cohort)
    if kind == "rank_only":
        for key in ("mae", "rmse", "mean_pred", "mean_actual"):
            result.pop(key, None)
    return result


def block_split(full: pd.DataFrame, season: str, train_max_gw: int, es: tuple):
    """Chronological split. Every later season is dropped, not just later gameweeks.

    Forgetting the season drop is the easy way to leak: holding back 2024-25 GW31-38
    while leaving all of 2025-26 in the training set trains the model on a year of
    football from after the test window.
    """
    df = full[full["season"] <= season]
    ts = df["season"].eq(season)
    pool = df[~(ts & df["gameweek"].gt(train_max_gw))]
    vm = pool["season"].eq(season) & pool["gameweek"].between(*es)
    return df, pool[~vm].copy(), pool[vm].copy()


def run_arm(train, val, test, feats, seeds, n_jobs):
    """Train at each seed and average the predictions.

    Averaging rather than reporting a single seed: the seed-to-seed range of this
    objective is 0.0106, which is larger than most real feature effects, so a
    one-seed number is not reproducible.
    """
    preds = []
    for sd in seeds:
        frames = []
        for pos in POSITIONS:
            t = test[test["position"].eq(pos)]
            if t.empty or train[train["position"].eq(pos)].empty:
                continue
            models, cal, w = fit_position(train, val, feats, pos, n_jobs, sd)
            X = t[feats].fillna(0.0).to_numpy()
            frames.append(pd.DataFrame({
                "season": t["season"].to_numpy(), "gameweek": t["gameweek"].to_numpy(),
                "fpl_code": t["fpl_code"].to_numpy(),
                "actual": t["target_points"].to_numpy(),
                "minutes": t["minutes"].fillna(0).to_numpy(),
                "pred": predict(models, cal, w, X),
                "p60": models["p60"].predict_proba(X)[:, 1]}))
        fx = pd.concat(frames, ignore_index=True)
        pg = collapse_to_player_gameweek(fx, "pred")
        # p60 is a probability, so it averages across a double gameweek rather than
        # summing like points do.
        pg = pg.merge(fx.groupby(["season", "gameweek", "fpl_code"], as_index=False)
                      .agg(p60=("p60", "mean")),
                      on=["season", "gameweek", "fpl_code"], validate="one_to_one")
        preds.append(pg)
    avg = preds[0].copy()
    for c in ("pred", "p60"):
        avg[c] = np.mean([p[c].to_numpy() for p in preds], axis=0)
    return avg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--scope", choices=("clean", "all"), default="all",
                    help="run only untouched 2024-25 blocks, or include discovery blocks")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "evaluation.json")
    args = ap.parse_args()
    seeds = SEEDS[: args.seeds]

    full = data.load()
    base, shipped = data.feature_sets(full, data.SHIPPED_CANDIDATES)
    core = [column for column in json.loads(CORE_FEATURES.read_text())
            if column in full.columns]
    arms = {"v13": shipped, "v13 - availability": base, "v12": core}
    blocks = [block for block in BLOCKS if args.scope == "all" or block[-1] == "clean"]
    print(f"v13 {len(shipped)} features | no availability {len(base)} | "
          f"v12 recipe {len(core)} | seeds {seeds}\n")

    out: dict = {}
    for label, season, tmax, test_gws, es, tag in blocks:
        df, tr, va = block_split(full, season, tmax, es)
        te = df[df["season"].eq(season) & df["gameweek"].between(*test_gws)]
        print(f"{label}  train {len(tr):,}  test {len(te):,}", flush=True)
        player_gameweeks = te.groupby(
            ["season", "gameweek", "fpl_code"], as_index=False
        ).agg(minutes=("minutes", "sum"))
        block = {
            "tag": tag,
            "test_rows": int(len(te)),
            "player_gameweeks": int(len(player_gameweeks)),
            "starter_player_gameweeks": int(player_gameweeks["minutes"].ge(60).sum()),
            "arms": {},
            "paired_baselines": {},
        }
        predictions = {}

        for arm, feats in arms.items():
            pf = run_arm(tr, va, te, feats, seeds, args.n_jobs)
            predictions[arm] = pf
            block["arms"][arm] = {c: score(pf, "pred", c) for c in ("all", "starters")}
            o = block["arms"][arm]["all"]
            print(f"  {arm:28} obj {o['obj']:.4f}  spearman {o['spearman']:.4f}  "
                  f"mae {o['mae']:.3f}", flush=True)
            if arm == "v13":
                y = (pf["minutes"] >= 60).astype(int)
                block["p60"] = {"auc": round(auc(y, pf["p60"]), 4),
                                "brier": round(brier(y, pf["p60"]), 4),
                                "base_rate": round(float(y.mean()), 4)}
                print(f"  {'p60 minutes gate':28} AUC {block['p60']['auc']:.4f}  "
                      f"Brier {block['p60']['brier']:.4f}  "
                      f"base rate {block['p60']['base_rate']:.3f}", flush=True)

        raw = te.groupby(["season", "gameweek", "fpl_code"], as_index=False).agg(
            **{c: (c, "first") for c in BASELINES.values()},
            actual=("target_points", "sum"), minutes=("minutes", "sum"))
        for name, col in BASELINES.items():
            values = pd.to_numeric(raw[col], errors="coerce")
            available = np.isfinite(values.to_numpy())
            # Only exactly -1 means no ep_next snapshot. Negative forecasts such as
            # -1.5 are unusual but valid and must remain in the benchmark.
            if col == "ep_next":
                available &= values.ne(-1.0).to_numpy()
            d = raw.loc[available].copy()
            d[col] = values.loc[available]
            if len(d) < 100:
                continue
            keys = ["season", "gameweek", "fpl_code"]
            ours = predictions["v13"].merge(d[keys], on=keys, validate="one_to_one")
            kind = BASELINE_KIND[name]
            comparison = {
                "kind": kind,
                "rows": int(len(d)),
                "coverage": round(len(d) / len(raw), 6),
                "dastan": {
                    cohort: baseline_score(ours, "pred", cohort, kind)
                    for cohort in ("all", "starters")
                },
                "baseline": {
                    cohort: baseline_score(d, col, cohort, kind)
                    for cohort in ("all", "starters")
                },
            }
            block["paired_baselines"][name] = comparison
            # Retain the historical key so existing consumers can migrate without
            # mistaking an old unpaired result for this exact-row comparison.
            if col == "ep_next":
                block["v13_on_ep_next_rows"] = comparison["dastan"]
            o = comparison["baseline"]["all"]
            print(f"  {name:28} obj {o['obj']:.4f}  spearman {o['spearman']:.4f}  "
                  f"rows {len(d):,}" +
                  (f"  mae {o['mae']:.3f}" if kind == "point_forecast" else ""))

        out[label] = block
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_id": "dastan-clean-walk-forward-v13",
            "model": {"features": len(shipped), "seeds": list(seeds)},
            "protocol": {
                "grain": "player-gameweek",
                "metric_aggregation": "compute within gameweek, then average",
                "comparison_rule": "Dastan and every baseline use identical eligible rows",
                "scope": args.scope,
            },
            "provenance": {
                "data_release_manifest_sha256": sha256(DATA_MANIFEST),
                "model_artifact_manifest_sha256": sha256(MODEL_MANIFEST),
                "code": {
                    name: {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256(path),
                    }
                    for name, path in EVIDENCE_CODE.items()
                },
            },
            "blocks": out,
        }
        args.out.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(flush=True)

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
