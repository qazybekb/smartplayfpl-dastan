#!/usr/bin/env python3
"""Retrain Dastan in a temporary directory and compare it with the release.

    python -m dastan.reproduce --n-jobs 8

This is stronger than ``dastan.verify``: verify checks that the published weights
reload and score, while this command starts from the published training frame, fits
every head again, and compares the resulting holdout predictions and fitted numeric
configuration with the checked-in release.
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
from .model import POSITIONS
from .predictor import Dastan
from .train import HOLDOUT, HOLDOUT_SEASON, ROOT

EXPECTED_XGBOOST = "3.2.0"
DEFAULT_ATOL = 1e-6


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
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
    released_out = released.predict_frame(holdout)

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
        candidate_out = candidate.predict_frame(holdout)
        if not released_out.index.equals(candidate_out.index):
            raise SystemExit("retrained predictions do not align with the released rows")

        released_models = ROOT / "models"
        config_pairs = [
            ("feature columns", "feature_cols.json"),
            ("bucket calibration", "bucket_calibration.json"),
        ]
        for label, filename in config_pairs:
            if _json(released_models / filename) != _json(candidate_dir / filename):
                raise SystemExit(f"retrained {label} differs from the release")

        released_blend = _json(released_models / "blend.json")["per_position_direct_weight"]
        candidate_blend = _json(candidate_dir / "blend.json")["per_position_direct_weight"]
        if released_blend != candidate_blend:
            raise SystemExit(
                f"retrained blend weights differ: release={released_blend}, "
                f"candidate={candidate_blend}"
            )

        released_xpts = released_out["xpts"].to_numpy(dtype=float)
        candidate_xpts = candidate_out["xpts"].to_numpy(dtype=float)
        delta = np.abs(released_xpts - candidate_xpts)
        exact_models = sum(
            (released_models / filename).read_bytes()
            == (candidate_dir / filename).read_bytes()
            for filename in _model_files()
        )

        print(f"xgboost: {xgboost.__version__}")
        print(f"holdout rows: {len(holdout):,}")
        print(f"model files byte-identical: {exact_models}/{len(_model_files())}")
        print(f"prediction mean absolute delta: {float(delta.mean()):.12g}")
        print(f"prediction max absolute delta: {float(delta.max()):.12g}")

        if not np.allclose(released_xpts, candidate_xpts, rtol=0.0, atol=args.atol):
            raise SystemExit(
                f"retrained predictions exceed the {args.atol:g} absolute tolerance"
            )

    print("\nreleased model reproduced from the published training frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
