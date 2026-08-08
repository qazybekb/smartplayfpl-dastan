"""Coherent expected-minutes and appearance probabilities from Dastan's p60 head."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .model import POSITIONS

N_BINS = 12


def fit_curve(rows: pd.DataFrame) -> dict[str, dict[str, list[float] | int]]:
    curve: dict[str, dict[str, list[float] | int]] = {}
    for pos in POSITIONS:
        group = rows[rows["position"].eq(pos)].copy()
        if len(group) < N_BINS * 25:
            raise RuntimeError(f"{pos}: only {len(group)} minutes-calibration rows")
        edges = np.unique(np.quantile(group["p60"], np.linspace(0, 1, N_BINS + 1)))
        if len(edges) < 3:
            raise RuntimeError(f"{pos}: p60 has no spread")
        bins = np.clip(np.digitize(group["p60"], edges[1:-1]), 0, len(edges) - 2)
        xs, expected_minutes, p_any = [], [], []
        for bucket in range(len(edges) - 1):
            selected = bins == bucket
            if int(selected.sum()) < 25:
                continue
            xs.append(float(group.loc[selected, "p60"].mean()))
            expected_minutes.append(float(group.loc[selected, "minutes"].mean()))
            p_any.append(float(group.loc[selected, "minutes"].gt(0).mean()))

        order = np.argsort(xs)
        x = np.asarray(xs, dtype=float)[order]
        minutes = np.maximum.accumulate(np.asarray(expected_minutes, dtype=float)[order])
        any_probability = np.maximum.accumulate(np.asarray(p_any, dtype=float)[order])
        minutes = np.clip(np.maximum(minutes, 60.0 * x), 0.0, 90.0)
        any_probability = np.clip(
            np.maximum.reduce([any_probability, x, minutes / 90.0]), 0.0, 1.0
        )
        curve[pos] = {
            "p60": [round(float(value), 6) for value in x],
            "e_minutes": [round(float(value), 4) for value in minutes],
            "p_any": [round(float(value), 6) for value in any_probability],
            "n": int(len(group)),
        }
    return curve


def apply_curve(
    p60: np.ndarray, curve: dict[str, list[float] | int]
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(p60, dtype=float)
    x = np.asarray(curve["p60"], dtype=float)
    minutes = np.clip(
        np.maximum(
            np.interp(probabilities, x, np.asarray(curve["e_minutes"], dtype=float)),
            60.0 * probabilities,
        ),
        0.0,
        90.0,
    )
    p_any = np.clip(
        np.maximum.reduce(
            [
                np.interp(probabilities, x, np.asarray(curve["p_any"], dtype=float)),
                probabilities,
                minutes / 90.0,
            ]
        ),
        0.0,
        1.0,
    )
    return minutes, p_any


def build_payload(rows: pd.DataFrame, source: str) -> dict:
    return {
        "note": (
            "Maps Dastan's p60 head onto expected minutes and p_any. Fitted on the "
            "packaging holdout used for calibration, using predictions from the exact "
            "packaged heads. All three published minutes values are coherent by construction."
        ),
        "method": "12 quantile-bin conditional means; cumulative-max and probability invariants",
        "source": source,
        "bins": N_BINS,
        "source_rows": int(len(rows)),
        "curve": fit_curve(rows),
    }


def payload_equivalent(expected: dict, actual: dict, *, atol: float = 5e-6) -> bool:
    """Compare calibration contracts while allowing harmless numeric roundoff."""
    for key in ("note", "method", "source", "bins", "source_rows"):
        if expected.get(key) != actual.get(key):
            return False
    if set(expected.get("curve", {})) != set(actual.get("curve", {})):
        return False
    for position, curve in expected["curve"].items():
        other = actual["curve"][position]
        if curve.get("n") != other.get("n"):
            return False
        for key in ("p60", "e_minutes", "p_any"):
            left = np.asarray(curve.get(key, []), dtype=float)
            right = np.asarray(other.get(key, []), dtype=float)
            if left.shape != right.shape or not np.allclose(
                left, right, rtol=0.0, atol=atol
            ):
                return False
    return True


def rows_from_artifacts(
    model_dir, holdout: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Predict p60 from saved heads so calibration matches the serving contract."""
    rows = []
    for position in POSITIONS:
        group = holdout[holdout["position"].eq(position)]
        model = xgb.XGBClassifier()
        model.load_model(str(model_dir / f"p60_{position}.json"))
        p60 = model.predict_proba(group[features].fillna(0.0).to_numpy())[:, 1]
        rows.append(
            pd.DataFrame(
                {
                    "position": position,
                    "p60": p60,
                    "minutes": group["minutes"].fillna(0.0).to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
