"""Load the released weights and predict. The shortest path to using Dastan.

    from dastan import data, predictor
    df = data.load()
    xpts = predictor.Dastan().predict_frame(df)

The class deliberately reloads every artefact from disk rather than accepting
in-memory objects, because that is what production does, and a model that trains
correctly but serialises wrongly should fail here rather than silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .model import POSITIONS, _pad, apply_bucket_calibration, compose
from .minutes import apply_curve

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


class Dastan:
    """The released model: 36 inference artifacts, including coherent minutes."""

    def __init__(self, model_dir: Path | str = MODELS):
        self.dir = Path(model_dir)
        self.features: list[str] = json.loads((self.dir / "feature_cols.json").read_text())
        self.calibration: dict = json.loads((self.dir / "bucket_calibration.json").read_text())
        self.blend: dict = json.loads(
            (self.dir / "blend.json").read_text())["per_position_direct_weight"]
        self.minutes: dict = json.loads(
            (self.dir / "minutes_calibration.json").read_text()
        )["curve"]
        self._cache: dict = {}

    def _load(self, name: str):
        if name not in self._cache:
            m = xgb.XGBClassifier() if name.startswith(("p60", "bucket_")) else xgb.XGBRegressor()
            m.load_model(str(self.dir / f"{name}.json"))
            self._cache[name] = m
        return self._cache[name]

    def predict_position(self, X: np.ndarray, pos: str) -> dict:
        """Returns the composed xPts and every intermediate quantity.

        The intermediates are part of the output on purpose. `p60` is a usable
        expected-minutes signal in its own right, and the band probabilities say
        something xPts alone cannot: whether a 5.0 projection is a steady five or a
        coin-flip between two and thirteen.
        """
        p60 = self._load(f"p60_{pos}").predict_proba(X)[:, 1]
        expected_minutes, p_any = apply_curve(p60, self.minutes[pos])
        non60 = np.clip(self._load(f"non60_{pos}").predict(X), 0.0, None)
        head = self._load(f"bucket_{pos}")
        pb = _pad(head.predict_proba(X), head.classes_)
        cal = self.calibration.get(pos)
        if cal:
            pb = apply_bucket_calibration(pb, cal)
        reg = np.column_stack([
            np.clip(self._load(f"bucketreg_{pos}_{k}").predict(X), 0.0, None) for k in range(4)])
        mb = compose(p60, non60, pb, reg)
        direct = np.clip(self._load(f"direct_{pos}").predict(X), 0.0, None)
        w = self.blend[pos]
        return {"xpts": np.clip((1.0 - w) * mb + w * direct, 0.0, None),
                "multibucket": mb, "direct": direct,
                "p60": p60, "p_any": p_any, "expected_minutes": expected_minutes,
                "bucket_probs": pb, "bucket_preds": reg}

    def predict_frame(self, df: pd.DataFrame, with_parts: bool = False) -> pd.DataFrame:
        """Predict for a frame containing the shipped feature columns."""
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"frame is missing {len(missing)} features, e.g. {missing[:5]}")
        out = []
        for pos in POSITIONS:
            d = df[df["position"].eq(pos)]
            if d.empty:
                continue
            r = self.predict_position(d[self.features].fillna(0.0).to_numpy(), pos)
            block = d.copy()
            block["xpts"] = r["xpts"]
            if with_parts:
                block["p60"] = r["p60"]
                block["p_any"] = r["p_any"]
                block["expected_minutes"] = r["expected_minutes"]
                block["multibucket"] = r["multibucket"]
                block["direct"] = r["direct"]
                for k in range(4):
                    block[f"p_band{k}"] = r["bucket_probs"][:, k]
            out.append(block)
        return pd.concat(out).sort_index()
