#!/usr/bin/env python3
"""Retrain Dastan in a temporary directory and compare it with the release.

    python -m dastan.reproduce --n-jobs 8

This is stronger than ``dastan.verify``: verify checks that the published weights
reload and score, while this command starts from the published training frame and fits
every head again. The default check requires equivalent holdout quality across CPU
architectures; ``--strict`` additionally requires identical fitted artefacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import xgboost

from . import data
from .metrics import collapse_to_player_gameweek, score
from .model import POSITIONS
from .predictor import Dastan
from .train import HOLDOUT, HOLDOUT_SEASON, ROOT

EXPECTED_XGBOOST = "3.2.0"
DEFAULT_ATOL = 1e-6
# Reproduction fits one seed, so use the measured single-seed objective noise.
DEFAULT_OBJECTIVE_ATOL = 0.0106


def _json(path: Path):
    return json.loads(path.read_text())


def _model_files() -> list[str]:
    names: list[str] = []
    for pos in POSITIONS:
        names.extend([
            f"p60_{pos}.json",
            f"non60_{pos}.json",
            f"bucket_{pos}.json",
            *(f"bucketreg_{pos}_{k}.json" for k in range(4)),
            f"direct_{pos}.json",
        ])
    return names


def _scores(out) -> dict[str, dict]:
    players = collapse_to_player_gameweek(
        out.assign(pred=out["xpts"], actual=out["target_points"]), "pred"
    )
    return {cohort: score(players, "pred", cohort) for cohort in ("all", "starters")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    ap.add_argument("--objective-atol", type=float, default=DEFAULT_OBJECTIVE_ATOL)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="require byte-identical model files, fitted config, and predictions",
    )
    args = ap.parse_args()

    if xgboost.__version__ != EXPECTED_XGBOOST:
        raise SystemExit(
            f"reproduction requires xgboost=={EXPECTED_XGBOOST}; "
            f"found {xgboost.__version__}. Install requirements.txt first."
        )

    frame = data.load()
    holdout = frame[
        frame["season"].eq(HOLDOUT_SEASON)
        & frame["gameweek"].between(*HOLDOUT)
    ]
    released = Dastan()
    released_out = released.predict_frame(holdout, with_parts=True)

    with tempfile.TemporaryDirectory(prefix="dastan-reproduce-") as tmp:
        candidate_dir = Path(tmp)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "dastan.train",
                "--n-jobs",
                str(args.n_jobs),
                "--seed",
                str(args.seed),
                "--out",
                str(candidate_dir),
            ],
            cwd=ROOT,
            check=True,
        )

        candidate = Dastan(candidate_dir)
        candidate_out = candidate.predict_frame(holdout, with_parts=True)
        if not released_out.index.equals(candidate_out.index):
            raise SystemExit("retrained predictions do not align with the released rows")

        released_models = ROOT / "models"
        if _json(released_models / "feature_cols.json") != _json(
            candidate_dir / "feature_cols.json"
        ):
            raise SystemExit("retrained feature columns differ from the release")

        calibration_matches = _json(
            released_models / "bucket_calibration.json"
        ) == _json(candidate_dir / "bucket_calibration.json")
        minutes_matches = _json(
            released_models / "minutes_calibration.json"
        ) == _json(candidate_dir / "minutes_calibration.json")

        released_blend = _json(released_models / "blend.json")["per_position_direct_weight"]
        candidate_blend = _json(candidate_dir / "blend.json")["per_position_direct_weight"]
        blend_matches = released_blend == candidate_blend

        released_xpts = released_out["xpts"].to_numpy(dtype=float)
        candidate_xpts = candidate_out["xpts"].to_numpy(dtype=float)
        if not np.isfinite(candidate_xpts).all() or (candidate_xpts < 0).any():
            raise SystemExit("retrained model produced invalid xPts")
        minute_values = candidate_out[
            ["p60", "p_any", "expected_minutes"]
        ].to_numpy(dtype=float)
        minutes_coherent = bool(
            np.isfinite(minute_values).all()
            and candidate_out["p_any"].ge(candidate_out["p60"] - 1e-9).all()
            and candidate_out["expected_minutes"]
            .ge(60.0 * candidate_out["p60"] - 1e-6)
            .all()
            and candidate_out["p_any"]
            .ge(candidate_out["expected_minutes"] / 90.0 - 1e-9)
            .all()
        )
        if not minutes_coherent:
            raise SystemExit("retrained model produced incoherent minutes outputs")

        delta = np.abs(released_xpts - candidate_xpts)
        exact_models = sum(
            (released_models / filename).read_bytes()
            == (candidate_dir / filename).read_bytes()
            for filename in _model_files()
        )
        released_scores = _scores(released_out)
        candidate_scores = _scores(candidate_out)

        print(f"xgboost: {xgboost.__version__}")
        print(f"holdout rows: {len(holdout):,}")
        print(f"model files byte-identical: {exact_models}/{len(_model_files())}")
        print(f"bucket calibration identical: {calibration_matches}")
        print(f"minutes calibration identical: {minutes_matches}")
        print(f"release blend weights: {released_blend}")
        print(f"retrained blend weights: {candidate_blend}")
        print(f"prediction mean absolute delta: {float(delta.mean()):.12g}")
        print(f"prediction max absolute delta: {float(delta.max()):.12g}")

        quality_failures = []
        for cohort in ("all", "starters"):
            released_obj = released_scores[cohort]["obj"]
            candidate_obj = candidate_scores[cohort]["obj"]
            objective_delta = candidate_obj - released_obj
            print(
                f"{cohort} objective: release={released_obj:.4f} "
                f"retrained={candidate_obj:.4f} delta={objective_delta:+.4f}"
            )
            if not np.isfinite(candidate_obj) or abs(objective_delta) > args.objective_atol:
                quality_failures.append(cohort)

        if quality_failures:
            sys.stdout.flush()
            raise SystemExit(
                "retrained objective exceeds the "
                f"{args.objective_atol:g} equivalence tolerance for: "
                f"{', '.join(quality_failures)}"
            )

        if args.strict:
            strict_failures = []
            if exact_models != len(_model_files()):
                strict_failures.append("model files")
            if not calibration_matches:
                strict_failures.append("bucket calibration")
            if not blend_matches:
                strict_failures.append("blend weights")
            if not minutes_matches:
                strict_failures.append("minutes calibration")
            if not np.allclose(released_xpts, candidate_xpts, rtol=0.0, atol=args.atol):
                strict_failures.append("predictions")
            if strict_failures:
                raise SystemExit(
                    "strict reproduction differs in: " + ", ".join(strict_failures)
                )

    mode = "strictly" if args.strict else "within the cross-platform quality tolerance"
    print(f"\nreleased model reproduced {mode} from the published training frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
