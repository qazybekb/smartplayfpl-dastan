#!/usr/bin/env python3
"""Clean walk-forward benchmark against public, reproducible baselines.

    python -m dastan.benchmark_baselines --seeds 3 --n-jobs 8

The command retrains Dastan, plus a separate arm without ``ep_next``, for the three
untouched 2024-25 test blocks. It retains the out-of-sample predictions and scores
every comparison on identical available rows.
It also publishes the position and return-bucket error tables used by OpenFPL, while
keeping those tables separate from ranking metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .evaluate import BLOCKS, SEEDS, block_split, run_arm
from .metrics import auc, brier, score

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = ROOT / "docs" / "baseline_benchmark.json"
DEFAULT_PREDICTIONS = ROOT / "docs" / "walkforward_predictions.parquet"
DEFAULT_MARKDOWN = ROOT / "docs" / "BASELINES.md"

KEYS = ["season", "gameweek", "fpl_code"]
CLEAN_BLOCKS = BLOCKS[:3]
COHORTS = ("all", "starters")


@dataclass(frozen=True)
class Baseline:
    label: str
    column: str
    point_forecast: bool
    missing_value: float | None = None


BASELINES = (
    Baseline("FPL ep_next", "ep_next", True, -1.0),
    Baseline("Last appearance points", "player_fpl_points_1", True),
    Baseline("Last-5 appearance mean", "player_fpl_points_5", True),
    Baseline("Price", "value", False),
)

RANK_METRICS = (
    "obj",
    "spearman",
    "ndcg@10",
    "top10_overlap",
    "captain_in_top10",
    "gameweeks",
    "rows",
)
POINT_METRICS = RANK_METRICS + ("mae", "rmse", "mean_pred", "mean_actual")
BOOTSTRAP_SEED = 2026
BOOTSTRAP_SAMPLES = 20_000


def available_rows(frame: pd.DataFrame, baseline: Baseline) -> pd.DataFrame:
    """Return rows on which a baseline published a real value.

    Only ``ep_next == -1`` is a missing-value sentinel. Negative historical points
    are legitimate FPL scores, so form baselines discard null/non-finite values only.
    """
    values = pd.to_numeric(frame[baseline.column], errors="coerce")
    keep = np.isfinite(values)
    if baseline.missing_value is not None:
        keep &= values.ne(baseline.missing_value)
    return frame.loc[keep].copy()


def _select_metrics(result: dict, point_forecast: bool) -> dict:
    keys = POINT_METRICS if point_forecast else RANK_METRICS
    return {key: result[key] for key in keys}


def _bootstrap_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(sample), size=(BOOTSTRAP_SAMPLES, len(sample)))
    means = sample[indices].mean(axis=1)
    return [round(float(x), 4) for x in np.quantile(means, (0.025, 0.975))]


def paired_uncertainty(
    frame: pd.DataFrame,
    baseline: Baseline,
    cohort: str,
    model_column: str = "dastan",
) -> dict:
    rows = frame if cohort == "all" else frame[frame["minutes"].ge(60)]
    differences = {
        "objective_advantage": [],
        "spearman_advantage": [],
        "ndcg@10_advantage": [],
    }
    if baseline.point_forecast:
        differences |= {"mae_reduction": [], "rmse_reduction": []}

    for _, gameweek in rows.groupby(["season", "gameweek"]):
        if len(gameweek) < 20:
            continue
        model = score(
            gameweek.rename(columns={model_column: "pred"}),
            "pred",
            min_players=20,
        )
        reference = score(
            gameweek.rename(columns={baseline.column: "pred"}),
            "pred",
            min_players=20,
        )
        differences["objective_advantage"].append(model["obj"] - reference["obj"])
        differences["spearman_advantage"].append(
            model["spearman"] - reference["spearman"]
        )
        differences["ndcg@10_advantage"].append(model["ndcg@10"] - reference["ndcg@10"])
        if baseline.point_forecast:
            differences["mae_reduction"].append(reference["mae"] - model["mae"])
            differences["rmse_reduction"].append(reference["rmse"] - model["rmse"])

    return {
        "method": "paired gameweek bootstrap, 95% percentile interval",
        "gameweeks": len(next(iter(differences.values()))),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "intervals": {
            metric: _bootstrap_interval(values)
            for metric, values in differences.items()
        },
    }


def _paired_scores(
    frame: pd.DataFrame, baseline: Baseline, model_column: str = "dastan"
) -> dict:
    paired = available_rows(frame, baseline)
    results = {}
    for cohort in COHORTS:
        model = score(paired.rename(columns={model_column: "pred"}), "pred", cohort)
        reference = score(
            paired.rename(columns={baseline.column: "pred"}), "pred", cohort
        )
        model = _select_metrics(model, baseline.point_forecast)
        reference = _select_metrics(reference, baseline.point_forecast)
        advantage = {
            "objective": round(model["obj"] - reference["obj"], 4),
            "spearman": round(model["spearman"] - reference["spearman"], 4),
            "ndcg@10": round(model["ndcg@10"] - reference["ndcg@10"], 4),
        }
        if baseline.point_forecast:
            advantage |= {
                "mae_reduction": round(reference["mae"] - model["mae"], 4),
                "rmse_reduction": round(reference["rmse"] - model["rmse"], 4),
            }
        results[cohort] = {
            model_column: model,
            "baseline": reference,
            "advantage": advantage,
            "uncertainty": paired_uncertainty(
                paired, baseline, cohort, model_column=model_column
            ),
        }
    return {
        "column": baseline.column,
        "kind": "point forecast" if baseline.point_forecast else "ranking proxy",
        "coverage": {
            "rows": int(len(paired)),
            "total_rows": int(len(frame)),
            "percent": round(100.0 * len(paired) / len(frame), 2),
            "gameweeks": int(paired[["season", "gameweek"]].drop_duplicates().shape[0]),
        },
        "results": results,
    }


def _error_metrics(frame: pd.DataFrame, prediction: str) -> dict:
    error = frame[prediction].astype(float) - frame["actual"].astype(float)
    return {
        "rows": int(len(frame)),
        "mae": round(float(error.abs().mean()), 4),
        "rmse": round(float(np.sqrt(np.square(error).mean())), 4),
        "mean_prediction": round(float(frame[prediction].mean()), 4),
        "mean_actual": round(float(frame["actual"].mean()), 4),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(KEYS, kind="mergesort")
    payload = ordered.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="",
        float_format="%.12g",
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def return_bucket(frame: pd.DataFrame) -> pd.Series:
    """OpenFPL's outcome buckets, evaluated after the result is known."""
    conditions = [
        frame["minutes"].le(0),
        frame["minutes"].gt(0) & frame["actual"].le(2),
        frame["actual"].between(3, 4),
    ]
    return pd.Series(
        np.select(conditions, ["zeros", "blanks", "tickers"], default="haulers"),
        index=frame.index,
        dtype="string",
    )


def error_breakdown(
    frame: pd.DataFrame, group_column: str, predictions: tuple[str, ...]
) -> dict:
    report = {}
    for group, rows in frame.groupby(group_column, sort=True):
        report[str(group)] = {
            prediction: _error_metrics(rows, prediction) for prediction in predictions
        }
    return report


def build_report(frame: pd.DataFrame, seeds: tuple[int, ...]) -> dict:
    """Build the complete benchmark report from frozen walk-forward predictions."""
    if frame.duplicated(KEYS).any():
        raise ValueError("walk-forward predictions contain duplicate player-gameweeks")
    comparisons = {
        baseline.label: _paired_scores(frame, baseline) for baseline in BASELINES
    }
    no_ep_next = "dastan_without_ep_next"
    independence = {
        "without_ep_next_vs_fpl": _paired_scores(
            frame, BASELINES[0], model_column=no_ep_next
        ),
        "full_vs_without_ep_next": _paired_scores(
            frame,
            Baseline("Dastan without ep_next", no_ep_next, True),
        ),
    }

    ep = available_rows(frame, BASELINES[0]).copy()
    ep["return_bucket"] = return_bucket(ep)
    outcome = error_breakdown(ep, "return_bucket", ("dastan", no_ep_next, "ep_next"))
    position = error_breakdown(ep, "position", ("dastan", no_ep_next, "ep_next"))

    headline = {
        cohort: score(frame.rename(columns={"dastan": "pred"}), "pred", cohort)
        for cohort in COHORTS
    }
    minutes_blocks = {}
    for block, rows in frame.groupby("block", sort=False):
        actual = rows["minutes"].ge(60).astype(int)
        minutes_blocks[str(block)] = {
            "auc": round(auc(actual, rows["p60"]), 4),
            "brier": round(brier(actual, rows["p60"]), 4),
            "base_rate": round(float(actual.mean()), 4),
            "rows": int(len(rows)),
        }
    minutes_gate = {
        "aggregation": "mean of the three block-level metrics",
        "auc": round(float(np.mean([x["auc"] for x in minutes_blocks.values()])), 4),
        "brier": round(
            float(np.mean([x["brier"] for x in minutes_blocks.values()])), 4
        ),
        "base_rate": round(
            float(np.mean([x["base_rate"] for x in minutes_blocks.values()])), 4
        ),
        "blocks": minutes_blocks,
    }
    zero = frame.assign(always_zero=0.0)
    return {
        "schema_version": 2,
        "command": (
            "python -m dastan.benchmark_baselines " f"--seeds {len(seeds)} --n-jobs 8"
        ),
        "protocol": {
            "method": "clean chronological walk-forward",
            "test": "2024-25 GW15-38 in three non-overlapping eight-GW blocks",
            "seeds": list(seeds),
            "metric_aggregation": "score inside each gameweek, then average gameweeks",
            "paired_comparisons": True,
            "player_gameweeks": int(len(frame)),
            "gameweeks": int(frame[["season", "gameweek"]].drop_duplicates().shape[0]),
        },
        "provenance": {
            "prediction_rows_sha256": _prediction_sha256(frame),
            "training_frame_sha256": _file_sha256(data.FRAME),
            "feature_manifest_sha256": _file_sha256(data.FEATURE_COLS),
            "ep_next_artifact_sha256": _file_sha256(data.EP_NEXT),
        },
        "headline": headline,
        "minutes_gate": minutes_gate,
        "comparisons": comparisons,
        "ep_next_feature_check": independence,
        "error_references": {
            "aggregation": "pooled player-gameweeks; error only, not a ranking benchmark",
            "Dastan": _error_metrics(zero, "dastan"),
            "Always zero": _error_metrics(zero, "always_zero"),
        },
        "openfpl_style": {
            "scope": "Dastan and ep_next on the same pre-deadline-covered rows",
            "aggregation": (
                "pooled player-gameweeks using the return and position presentation "
                "from OpenFPL Tables 4 and 5"
            ),
            "return_buckets": {
                "definitions": {
                    "zeros": "played 0 minutes",
                    "blanks": "played and scored at most 2",
                    "tickers": "scored 3 or 4",
                    "haulers": "scored at least 5",
                },
                "results": outcome,
            },
            "positions": position,
        },
    }


def render_markdown(report: dict) -> str:
    protocol = report["protocol"]
    lines = [
        "# Reproducible baselines",
        "",
        "This report is generated from retained, out-of-sample walk-forward predictions.",
        "Every head-to-head comparison restricts Dastan and the baseline to identical",
        "player-gameweek rows. Ranking metrics are calculated within each gameweek and",
        "then averaged.",
        "",
        "```bash",
        "python -m dastan.benchmark_baselines --seeds 3 --n-jobs 8",
        "```",
        "",
        f"Scope: **{protocol['gameweeks']} gameweeks, "
        f"{protocol['player_gameweeks']:,} player-gameweeks**.",
        "",
        "## FPL's projection",
        "",
        "FPL `ep_next` is the strongest free external baseline available in the official",
        "game. It is also one Dastan input, so the full-model row measures value added",
        "over FPL; the separately retrained no-`ep_next` row tests independence.",
        "",
        "| cohort | forecast | rows | objective | Spearman | NDCG@10 | MAE | RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ep = report["comparisons"]["FPL ep_next"]
    no_ep = report["ep_next_feature_check"]["without_ep_next_vs_fpl"]
    for cohort in COHORTS:
        forecasts = (
            ("Dastan", ep["results"][cohort]["dastan"]),
            (
                "Dastan without `ep_next`",
                no_ep["results"][cohort]["dastan_without_ep_next"],
            ),
            ("FPL `ep_next`", ep["results"][cohort]["baseline"]),
        )
        for label, result in forecasts:
            lines.append(
                f"| {cohort} | **{label}** | {result['rows']:,} | "
                f"{result['obj']:.4f} | {result['spearman']:.4f} | "
                f"{result['ndcg@10']:.4f} | {result['mae']:.3f} | "
                f"{result['rmse']:.3f} |"
            )

    all_ep = ep["results"]["all"]
    all_no_ep = no_ep["results"]["all"]
    feature_check = report["ep_next_feature_check"]["full_vs_without_ep_next"]
    feature_all = feature_check["results"]["all"]
    objective_ci = all_ep["uncertainty"]["intervals"]["objective_advantage"]
    mae_ci = all_ep["uncertainty"]["intervals"]["mae_reduction"]
    lines += [
        "",
        "Paired-gameweek bootstrap (95%): objective advantage",
        f"**{objective_ci[0]:+.4f} to {objective_ci[1]:+.4f}**; MAE reduction",
        f"**{mae_ci[0]:.3f} to {mae_ci[1]:.3f} points**. This interval resamples the",
        "24 test gameweeks; it does not add another layer of model-seed uncertainty.",
        "",
        "Removing `ep_next` still leaves an objective advantage of",
        f"**{all_no_ep['advantage']['objective']:+.4f}** over FPL. Adding the feature",
        "back changes Dastan's objective by only",
        f"**{feature_all['advantage']['objective']:+.4f}**; see the JSON report for its",
        "paired interval and error metrics.",
    ]

    lines += [
        "",
        "## Public baselines",
        "",
        "The table below reports the all-player cohort. `Price` is a ranking proxy, not a",
        "points forecast, so point-error metrics do not apply.",
        "",
        "| baseline | coverage | Dastan objective | baseline objective | advantage | baseline MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, comparison in report["comparisons"].items():
        result = comparison["results"]["all"]
        baseline_mae = result["baseline"].get("mae")
        lines.append(
            f"| {label} | {comparison['coverage']['rows']:,}/"
            f"{comparison['coverage']['total_rows']:,} "
            f"({comparison['coverage']['percent']:.1f}%) | "
            f"{result['dastan']['obj']:.4f} | {result['baseline']['obj']:.4f} | "
            f"{result['advantage']['objective']:+.4f} | "
            f"{baseline_mae:.3f} |"
            if baseline_mae is not None
            else f"| {label} | {comparison['coverage']['rows']:,}/"
            f"{comparison['coverage']['total_rows']:,} "
            f"({comparison['coverage']['percent']:.1f}%) | "
            f"{result['dastan']['obj']:.4f} | {result['baseline']['obj']:.4f} | "
            f"{result['advantage']['objective']:+.4f} | n/a |"
        )

    lines += [
        "",
        "## Participation",
        "",
        "The 60-minute head is evaluated as a probability model, separately from xPts.",
        "AUC measures ranking of participation likelihood; Brier measures probability",
        "calibration and sharpness. AUC is not classification accuracy.",
        "",
        "| AUC | Brier | 60-minute base rate |",
        "|---:|---:|---:|",
        f"| {report['minutes_gate']['auc']:.4f} | "
        f"{report['minutes_gate']['brier']:.4f} | "
        f"{report['minutes_gate']['base_rate']:.1%} |",
        "",
        "## Error sanity check",
        "",
        "An all-player MAE can look artificially good because most registered players do",
        "not appear. The constant-zero forecast is included as an error-only sanity check;",
        "it cannot rank players and therefore has no Spearman or NDCG score.",
        "",
        "| forecast | rows | MAE | RMSE |",
        "|---|---:|---:|---:|",
    ]
    for label in ("Dastan", "Always zero"):
        result = report["error_references"][label]
        lines.append(
            f"| {label} | {result['rows']:,} | {result['mae']:.3f} | "
            f"{result['rmse']:.3f} |"
        )

    lines += [
        "",
        "## OpenFPL-style return buckets",
        "",
        "These outcome-conditioned errors use the buckets from the OpenFPL paper.",
        "They explain *where* error occurs; they do not test whether a model can identify a",
        "future bucket before the match. The test window differs, so these numbers are not",
        "directly comparable with OpenFPL's paper tables.",
        "",
        "| actual outcome | rows | Dastan RMSE (MAE) | FPL `ep_next` RMSE (MAE) |",
        "|---|---:|---:|---:|",
    ]
    buckets = report["openfpl_style"]["return_buckets"]["results"]
    for bucket in ("zeros", "blanks", "tickers", "haulers"):
        ours, ep_result = buckets[bucket]["dastan"], buckets[bucket]["ep_next"]
        lines.append(
            f"| {bucket.title()} | {ours['rows']:,} | {ours['rmse']:.3f} "
            f"({ours['mae']:.3f}) | {ep_result['rmse']:.3f} "
            f"({ep_result['mae']:.3f}) |"
        )

    lines += [
        "",
        "## Error by position",
        "",
        "This is the same paired `ep_next` sample, split by the player's FPL position.",
        "",
        "| position | rows | Dastan RMSE (MAE) | FPL `ep_next` RMSE (MAE) |",
        "|---|---:|---:|---:|",
    ]
    positions = report["openfpl_style"]["positions"]
    for position in ("GKP", "DEF", "MID", "FWD"):
        ours, ep_result = positions[position]["dastan"], positions[position]["ep_next"]
        lines.append(
            f"| {position} | {ours['rows']:,} | {ours['rmse']:.3f} "
            f"({ours['mae']:.3f}) | {ep_result['rmse']:.3f} "
            f"({ep_result['mae']:.3f}) |"
        )

    lines += [
        "",
        "## Limits of comparison",
        "",
        "- The public benchmark is one gameweek ahead. Historical two- and three-gameweek",
        "  pre-deadline forecasts were not captured, so no multi-horizon claim is made.",
        "- FPL Review is not included. A fair comparison requires licensed forecasts",
        "  captured before the same deadlines; OpenFPL's paper values use a different test",
        "  window and cannot be pasted beside Dastan's numbers.",
        "- The separate OpenFPL head-to-head retrains both methods under OpenFPL's published",
        "  regime and scores 18,173 identical rows. See `openfpl_benchmark.json`.",
        "",
        "The machine-readable source is [`baseline_benchmark.json`](baseline_benchmark.json),",
        "and the retained predictions are",
        "[`walkforward_predictions.parquet`](walkforward_predictions.parquet).",
        "",
    ]
    return "\n".join(lines)


def _prediction_frame(
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    label: str,
    model_column: str = "dastan",
) -> pd.DataFrame:
    raw = test.groupby(KEYS, as_index=False).agg(
        position=("position", "first"),
        **{baseline.column: (baseline.column, "first") for baseline in BASELINES},
        actual=("target_points", "sum"),
        minutes=("minutes", "sum"),
    )
    model = predictions[KEYS + ["pred", "p60"]].rename(columns={"pred": model_column})
    if model_column != "dastan":
        model = model.drop(columns="p60")
    out = raw.merge(model, on=KEYS, validate="one_to_one")
    out.insert(0, "block", label)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions-out", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--from-predictions",
        type=Path,
        help="regenerate reports from retained predictions without fitting models",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in JSON and Markdown against retained predictions",
    )
    args = parser.parse_args()
    if not 1 <= args.seeds <= len(SEEDS):
        raise SystemExit(f"--seeds must be between 1 and {len(SEEDS)}")
    seeds = SEEDS[: args.seeds]

    prediction_source = args.from_predictions
    if args.check and prediction_source is None:
        prediction_source = DEFAULT_PREDICTIONS
    if prediction_source:
        if args.seeds != len(SEEDS):
            raise SystemExit(
                "--seeds applies only when fitting; retained predictions use all 3 seeds"
            )
        prediction_frame = pd.read_parquet(prediction_source)
        print(f"loaded {len(prediction_frame):,} rows from {prediction_source}")
    else:
        full = data.load()
        features = data.shipped_features(full)
        without_ep_next = [c for c in features if c != "ar_ep_next"]
        if len(without_ep_next) != len(features) - 1:
            raise RuntimeError("released feature manifest does not contain ar_ep_next")
        model_specs = (
            ("dastan", features),
            ("dastan_without_ep_next", without_ep_next),
        )
        frames = []
        print(f"Dastan {len(features)} features | seeds {seeds}\n")
        for label, season, train_max_gw, test_gws, early_stop, _ in CLEAN_BLOCKS:
            source, train, valid = block_split(full, season, train_max_gw, early_stop)
            test = source[
                source["season"].eq(season) & source["gameweek"].between(*test_gws)
            ]
            print(
                f"{label}  train {len(train):,}  validation {len(valid):,}  "
                f"test {len(test):,}",
                flush=True,
            )
            block_frame = None
            for model_column, model_features in model_specs:
                predictions = run_arm(
                    train, valid, test, model_features, seeds, args.n_jobs
                )
                candidate = _prediction_frame(
                    test, predictions, label, model_column=model_column
                )
                if block_frame is None:
                    block_frame = candidate
                else:
                    block_frame = block_frame.merge(
                        candidate[KEYS + [model_column]],
                        on=KEYS,
                        validate="one_to_one",
                    )
                result = score(predictions, "pred", "all")
                print(
                    f"  {model_column:24} objective {result['obj']:.4f}  "
                    f"spearman {result['spearman']:.4f}  MAE {result['mae']:.3f}",
                    flush=True,
                )
            if block_frame is None:
                raise RuntimeError(f"no predictions generated for {label}")
            frames.append(block_frame)
            print(flush=True)
        prediction_frame = pd.concat(frames, ignore_index=True)
    report = build_report(prediction_frame, seeds)
    report_text = json.dumps(report, indent=2) + "\n"
    markdown_text = render_markdown(report)
    if args.check:
        failures = []
        if not args.out.exists() or args.out.read_text() != report_text:
            failures.append(str(args.out))
        if (
            not args.markdown_out.exists()
            or args.markdown_out.read_text() != markdown_text
        ):
            failures.append(str(args.markdown_out))
        if failures:
            raise SystemExit(
                "benchmark artifacts are stale; regenerate: " + ", ".join(failures)
            )
        print(
            f"verified benchmark artifacts | {len(prediction_frame):,} rows | "
            f"{report['protocol']['gameweeks']} gameweeks"
        )
        return 0

    for path in (args.out, args.predictions_out, args.markdown_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report_text)
    if not args.from_predictions:
        prediction_frame.to_parquet(args.predictions_out, index=False)
    args.markdown_out.write_text(markdown_text)
    print(f"wrote {args.out}")
    if not args.from_predictions:
        print(f"wrote {args.predictions_out}")
    print(f"wrote {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
