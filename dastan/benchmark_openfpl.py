#!/usr/bin/env python3
"""Head-to-head benchmark against OpenFPL and public-data baselines.

    python -m dastan.benchmark_openfpl --seeds 3 --n-jobs 8

OpenFPL (Groos, arXiv:2508.09992, https://github.com/daniegr/OpenFPL) is the
work Dastan's feature engineering is built on, and the fairest external reference
point for it. Its stored predictions cover 2025-26 GW1-24 under this regime:

    train        2020-21 .. 2023-24
    early stop   2024-25
    test         2025-26 GW1-24        18,173 player-fixtures

Dastan is retrained here under exactly that regime. The released weights trained
through 2025-26 GW30 and therefore cannot be scored on this window without leakage.
OpenFPL and every Dastan seed are joined on the same fixture keys before scoring.

The report also includes three checks a headline score should not hide:

* FPL's official ``ep_next`` projection, from provenance-checked pre-deadline data;
* mean points over the previous five matches, the naive baseline used by OpenFPL;
* player price as a rank-only market baseline.

Those baselines have different coverage. For each one, every model is rescored on
the baseline's exact eligible rows. Return-band errors use OpenFPL's definitions so
we can see whether a result comes from non-players, blanks, tickers, or haulers.
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
from .metrics import score, score_by_gameweek
from .model import POSITIONS, fit_position, predict

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "openfpl_predictions.csv"
BRIDGE = ROOT / "data" / "openfpl_row_keys.csv"
DATA_MANIFEST = ROOT / "data" / "release_manifest.json"
MODEL_MANIFEST = ROOT / "models" / "artifact_manifest.json"
EVIDENCE_CODE = {
    "benchmark_openfpl": Path(__file__).resolve(),
    "metrics": ROOT / "dastan" / "metrics.py",
    "model": ROOT / "dastan" / "model.py",
    "data": ROOT / "dastan" / "data.py",
}

TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
VALID_SEASON = "2024-25"
TEST_SEASON, TEST_GWS = "2025-26", (1, 24)
SEEDS = (42, 7, 2026)
KEYS = ["season", "gameweek", "fixture", "fpl_code"]
PLAYER_GW_KEYS = ["season", "gameweek", "fpl_code"]
METRIC_KEYS = ("obj", "spearman", "ndcg@10", "mae", "rmse")
RANK_KEYS = (
    "obj",
    "spearman",
    "ndcg@10",
    "top10_overlap",
    "captain_in_top10",
    "gameweeks",
    "rows",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_set(frame: pd.DataFrame, column: str, cohort: str, *, rank_only=False) -> dict:
    result = score(frame, column, cohort)
    if rank_only:
        return {key: result[key] for key in RANK_KEYS}
    return result


def _delta(left: dict, right: dict) -> dict:
    return {key: round(left[key] - right[key], 4) for key in METRIC_KEYS}


def _eligible(frame: pd.DataFrame, column: str, *, missing_value=None) -> pd.DataFrame:
    values = pd.to_numeric(frame[column], errors="coerce")
    mask = np.isfinite(values.to_numpy())
    if missing_value is not None:
        mask &= values.ne(missing_value).to_numpy()
    out = frame.loc[mask].copy()
    out[column] = values.loc[mask]
    return out


def _point_error(frame: pd.DataFrame, column: str) -> dict:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values) & np.isfinite(actual)
    error = values[finite] - actual[finite]
    if not len(error):
        return {"rows": 0, "mae": None, "rmse": None, "bias": None}
    return {
        "rows": int(len(error)),
        "mae": round(float(np.mean(np.abs(error))), 4),
        "rmse": round(float(np.sqrt(np.mean(error ** 2))), 4),
        "bias": round(float(np.mean(error)), 4),
    }


def _return_band_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    minutes = frame["minutes"].fillna(0)
    actual = frame["actual"]
    masks = {
        # OpenFPL calls this category "Zeros" because a non-player normally scores
        # zero. Keep the minutes definition authoritative: the released test data has
        # one zero-minute row with a -1 retrospective scoring correction.
        "zeros": minutes.eq(0),
        "blanks": minutes.gt(0) & actual.le(2),
        "tickers": actual.between(3, 4),
        "haulers": actual.ge(5),
    }
    assigned = sum((mask.astype(int) for mask in masks.values()))
    if not assigned.eq(1).all():
        bad = frame.loc[assigned.ne(1), [*PLAYER_GW_KEYS, "minutes", "actual"]]
        raise RuntimeError(
            f"return-band definitions did not partition {len(bad)} rows; "
            f"examples: {bad.head(3).to_dict('records')}"
        )
    return masks


def _return_band_errors(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict:
    report = {}
    for band, mask in _return_band_masks(frame).items():
        subset = frame.loc[mask]
        report[band] = {
            "definition": {
                "zeros": "0 minutes (normally 0 points; includes scoring corrections)",
                "blanks": "played and scored at most 2 points",
                "tickers": "scored 3 or 4 points",
                "haulers": "scored at least 5 points",
            }[band],
            "rows": int(len(subset)),
            "results": {column: _point_error(subset, column) for column in columns},
        }
    return report


def _position_return_band_errors(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict:
    return {
        str(position): _return_band_errors(group, columns)
        for position, group in frame.groupby("position", sort=True)
    }


def _paired_bootstrap(
    frame: pd.DataFrame,
    left: str,
    right: str,
    cohort: str,
    *,
    samples: int,
    seed: int = 20260808,
) -> dict:
    """Paired gameweek bootstrap; measures week-to-week, not training, uncertainty."""
    left_gw = score_by_gameweek(frame, left, cohort)
    right_gw = score_by_gameweek(frame, right, cohort)
    joined = left_gw.merge(
        right_gw,
        on=["season", "gameweek"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    report = {
        "confidence": 0.95,
        "resampling_unit": "gameweek",
        "samples": int(samples),
        "gameweeks": int(len(joined)),
        "scope": "week-to-week variation with fitted predictions held fixed",
    }
    for metric in METRIC_KEYS:
        deltas = (
            joined[f"{metric}_left"].to_numpy()
            - joined[f"{metric}_right"].to_numpy()
        )
        draws = rng.choice(deltas, size=(samples, len(deltas)), replace=True).mean(axis=1)
        report[metric] = {
            "delta": round(float(deltas.mean()), 4),
            "lower": round(float(np.quantile(draws, 0.025)), 4),
            "upper": round(float(np.quantile(draws, 0.975)), 4),
        }
    return report


def _train_dastan(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    seeds: tuple[int, ...],
    n_jobs: int,
) -> pd.DataFrame:
    seed_predictions = []
    base = None
    for seed in seeds:
        frames = []
        print(f"seed {seed}", flush=True)
        for position in POSITIONS:
            subset = test[test["position"].eq(position)]
            if subset.empty:
                continue
            print(f"  fitting {position}...", flush=True)
            models, calibration, weights = fit_position(
                train, valid, features, position, n_jobs, seed
            )
            frames.append(
                pd.DataFrame(
                    {
                        "season": subset["season"].to_numpy(),
                        "gameweek": subset["gameweek"].to_numpy(),
                        "fixture": subset["fixture"].to_numpy(),
                        "fpl_code": subset["fpl_code"].to_numpy(),
                        "position": subset["position"].to_numpy(),
                        "dastan": predict(
                            models,
                            calibration,
                            weights,
                            subset[features].fillna(0.0).to_numpy(),
                        ),
                        "actual": subset["target_points"].to_numpy(),
                        "minutes": subset["minutes"].fillna(0).to_numpy(),
                        "ep_next": subset["ep_next"].to_numpy(),
                        "last_five": subset["player_fpl_points_5"].to_numpy(),
                        "price": subset["value"].to_numpy(),
                    }
                )
            )
        current = pd.concat(frames, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        if base is None:
            base = current.drop(columns="dastan")
        elif not current[KEYS].equals(base[KEYS]):
            raise RuntimeError(f"Dastan row keys changed at seed {seed}")
        seed_predictions.append(current["dastan"].to_numpy())
    if base is None:
        raise RuntimeError("Dastan produced no benchmark predictions")
    base["dastan"] = np.mean(seed_predictions, axis=0)
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3, choices=range(1, len(SEEDS) + 1))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs" / "openfpl_benchmark.json"
    )
    args = parser.parse_args()
    seeds = SEEDS[: args.seeds]

    frame = data.load()
    features = data.shipped_features(frame)
    train = frame[frame["season"].isin(TRAIN_SEASONS)]
    valid = frame[frame["season"].eq(VALID_SEASON)]
    test = frame[
        frame["season"].eq(TEST_SEASON)
        & frame["gameweek"].between(*TEST_GWS)
    ]
    print(
        f"train {len(train):,} | early stop {len(valid):,} | "
        f"test {len(test):,} | seeds {seeds}\n",
        flush=True,
    )
    ours = _train_dastan(train, valid, test, features, seeds, args.n_jobs)

    bridge = pd.read_csv(BRIDGE)
    stored = pd.read_csv(BENCH).rename(
        columns={"gw": "gameweek", "openfpl_xpts": "openfpl"}
    )
    stored = bridge.merge(
        stored[["element", "gameweek", "openfpl"]],
        on=["element", "gameweek"],
        validate="one_to_one",
    )
    joined = ours.merge(
        stored[[*KEYS, "openfpl"]], on=KEYS, validate="one_to_one"
    )
    if len(joined) != len(ours) or len(joined) != len(stored):
        raise RuntimeError(
            f"benchmark join lost rows: Dastan={len(ours):,}, "
            f"OpenFPL={len(stored):,}, joined={len(joined):,}"
        )
    print(f"\njoined on identical fixture rows: {len(joined):,}\n")

    player_gameweeks = joined.groupby(PLAYER_GW_KEYS, as_index=False).agg(
        dastan=("dastan", "sum"),
        openfpl=("openfpl", "sum"),
        actual=("actual", "sum"),
        minutes=("minutes", "sum"),
        position=("position", "first"),
        ep_next=("ep_next", "first"),
        last_five=("last_five", "first"),
        price=("price", "first"),
    )

    primary_results = {
        model: {
            cohort: _metric_set(player_gameweeks, model, cohort)
            for cohort in ("all", "starters")
        }
        for model in ("dastan", "openfpl")
    }
    primary_delta = {
        cohort: _delta(
            primary_results["dastan"][cohort],
            primary_results["openfpl"][cohort],
        )
        for cohort in ("all", "starters")
    }

    baseline_specs = {
        "fpl_ep_next": {
            "column": "ep_next",
            "label": "FPL ep_next",
            "missing_value": -1.0,
            "kind": "point_forecast",
            "note": "Official FPL projection captured before the deadline.",
        },
        "last_five": {
            "column": "last_five",
            "label": "Last-five mean points",
            "kind": "point_forecast",
            "note": "Mean FPL points over the previous five matches; OpenFPL's naive baseline.",
        },
        "price": {
            "column": "price",
            "label": "Player price",
            "kind": "rank_only",
            "note": "Market-cost sanity check; error metrics are invalid because price is not points.",
        },
    }
    paired_baselines = {}
    for name, spec in baseline_specs.items():
        subset = _eligible(
            player_gameweeks,
            spec["column"],
            missing_value=spec.get("missing_value"),
        )
        models = ("dastan", "openfpl", spec["column"])
        rank_only = spec["kind"] == "rank_only"
        results = {
            model: {
                cohort: _metric_set(subset, model, cohort, rank_only=rank_only)
                for cohort in ("all", "starters")
            }
            for model in models
        }
        paired_baselines[name] = {
            "label": spec["label"],
            "kind": spec["kind"],
            "note": spec["note"],
            "rows": int(len(subset)),
            "coverage": round(len(subset) / len(player_gameweeks), 6),
            "gameweeks": int(subset["gameweek"].nunique()),
            "results": results,
        }

    report = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": "dastan-openfpl-2025-26-gw1-24",
        "model": {
            "name": "Dastan",
            "features": int(len(features)),
            "seeds": list(seeds),
            "prediction_aggregation": "arithmetic mean across seeds",
        },
        "regime": {
            "train": TRAIN_SEASONS,
            "early_stopping": VALID_SEASON,
            "test": f"{TEST_SEASON} GW{TEST_GWS[0]}-{TEST_GWS[1]}",
            "fixture_rows": int(len(joined)),
            "player_gameweeks": int(len(player_gameweeks)),
            "grain": "player-gameweek",
            "join_keys": KEYS,
            "comparison_rule": "all models are rescored on identical eligible rows",
        },
        "provenance": {
            "openfpl_predictions": {
                "path": str(BENCH.relative_to(ROOT)),
                "sha256": _sha256(BENCH),
            },
            "openfpl_row_keys": {
                "path": str(BRIDGE.relative_to(ROOT)),
                "sha256": _sha256(BRIDGE),
            },
            "data_release_manifest_sha256": _sha256(DATA_MANIFEST),
            "model_artifact_manifest_sha256": _sha256(MODEL_MANIFEST),
            "ep_next_source": "data/pre_deadline_ep_next.parquet",
            "code": {
                name: {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256(path),
                }
                for name, path in EVIDENCE_CODE.items()
            },
        },
        "primary": {
            "models": ["dastan", "openfpl"],
            "results": primary_results,
            "delta_dastan_minus_openfpl": primary_delta,
            "paired_gameweek_bootstrap": {
                cohort: _paired_bootstrap(
                    player_gameweeks,
                    "dastan",
                    "openfpl",
                    cohort,
                    samples=args.bootstrap_samples,
                )
                for cohort in ("all", "starters")
            },
        },
        "paired_baselines": paired_baselines,
        "return_bands": {
            "protocol": "OpenFPL one-gameweek return categories; RMSE/MAE are pooled within each category",
            "overall": _return_band_errors(
                player_gameweeks, ("dastan", "openfpl")
            ),
            "by_position": _position_return_band_errors(
                player_gameweeks, ("dastan", "openfpl")
            ),
        },
        "limitations": [
            "The OpenFPL comparison uses stored predictions and cannot measure later OpenFPL changes.",
            "The bootstrap interval captures gameweek variation with fitted predictions held fixed; it is not a training-seed confidence interval.",
            "This is a one-gameweek forecast benchmark. Two- and three-gameweek horizons are not claimed.",
            "FPL Review is not included because synchronized proprietary predictions for these rows are unavailable.",
        ],
    }

    print(
        f"{'model':12} {'cohort':10} {'obj':>8} {'spearman':>9} "
        f"{'ndcg@10':>8} {'mae':>7} {'rmse':>7}"
    )
    for model in ("dastan", "openfpl"):
        for cohort in ("all", "starters"):
            result = primary_results[model][cohort]
            print(
                f"{model:12} {cohort:10} {result['obj']:>8.4f} "
                f"{result['spearman']:>9.4f} {result['ndcg@10']:>8.4f} "
                f"{result['mae']:>7.3f} {result['rmse']:>7.3f}"
            )
    print("\npaired baselines:")
    for name, benchmark in paired_baselines.items():
        baseline_column = baseline_specs[name]["column"]
        result = benchmark["results"][baseline_column]["all"]
        print(
            f"  {benchmark['label']:24} rows {benchmark['rows']:>6,}  "
            f"objective {result['obj']:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
