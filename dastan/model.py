"""The Dastan architecture: a conditional decomposition of FPL points.

A single regressor trained on FPL points spends most of its capacity learning that
most players score nothing. About 62% of player-gameweeks in this dataset are zero
minutes, and among players who *do* start, the points distribution is violently
right-skewed: a defender's typical return is 2, and their occasional return is 13.
Fitting one mean-squared-error model across that mixture produces something that is
wrong in a specific, useless way -- it never predicts a haul.

So the target is decomposed instead:

    xPts = p60 * SUM_k P(bucket k | started) * E[points | bucket k]
         + (1 - p60) * E[points | did not start]

Four heads, each trained on the population it actually models:

  p60        P(player plays 60+ minutes). A classifier over everyone.
  non60      expected points GIVEN under 60 minutes. Trained only on those rows.
  bucket     P(scoring band | started), 4 classes split at 1, 3 and 10 points.
             Trained only on 60+ rows.
  bucketreg  expected points within each band. Four regressors, one per band.

Each head early-stops on the population it models -- not on the full validation set.
That sounds pedantic and is not: early-stopping the "given under 60 minutes" head on
a validation set dominated by starters selects the wrong number of trees for the job
it has. Fixing this class of defect was worth more to this model than any feature.

A fifth head, `direct`, is an ordinary regressor on all rows. It is blended in with a
per-position weight fitted on validation. The decomposition is better at the tails;
the direct model is steadier in the middle, and mixing them beats either alone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .metrics import ndcg, spearman

ROOT = Path(__file__).resolve().parent.parent
HYPERPARAMS = ROOT / "models" / "hyperparameters.json"

POSITIONS = ["GKP", "DEF", "MID", "FWD"]
BUCKET_EDGES = [1, 3, 10]     # -> bands [0,1) [1,3) [3,10) [10,inf)
MIN_ES = 50                   # below this, a conditional head early-stops on all rows


def params(n_jobs: int = 8, seed: int = 42):
    """Hyperparameters. Shipped as data because they are part of the model.

    These were tuned once, years of gameweeks ago in FPL terms, and a 500-trial
    Optuna search failed to beat them out of sample -- see docs/METHODOLOGY.md.
    """
    m = json.loads(HYPERPARAMS.read_text())
    clf, reg = dict(m["clf_params"]), dict(m["reg_params"])
    ces = clf.pop("early_stopping_rounds", 30)
    res = reg.pop("early_stopping_rounds", 90)
    clf.update(n_jobs=n_jobs, random_state=seed)
    reg.update(n_jobs=n_jobs, random_state=seed)
    return clf, reg, ces, res


def fit_bucket_calibration(p_raw_val, bucket_val, pos: str) -> dict:
    """Fit gamma and band weights on THIS fold's validation split.

    Reusing a calibration fitted elsewhere is model-selection leakage even when no
    training row crosses a boundary: the calibration itself has seen the answers. It
    is refitted per fold, per position, on the same log-likelihood every time.
    """
    best = {"gamma": 1.0, "w0": 1.0, "w1": 1.0, "w2": 1.0, "w3": 1.0}
    best_ll = -np.inf
    for gamma in (0.5, 0.75, 1.0):
        for w0 in (1.0, 1.25, 1.5, 2.0):
            for w1 in (1.0, 1.25):
                w = np.array([w0, w1, 1.0, 1.0], dtype=float)
                adj = np.clip(p_raw_val, 1e-12, 1.0) ** gamma * w[None, :]
                den = adj.sum(axis=1, keepdims=True)
                p = adj / np.where(den <= 0, 1.0, den)
                ll = float(np.log(np.clip(
                    p[np.arange(len(bucket_val)), bucket_val], 1e-12, 1)).mean())
                if ll > best_ll:
                    best_ll = ll
                    best = {"gamma": gamma, "w0": w0, "w1": w1, "w2": 1.0, "w3": 1.0}
    return best


def apply_bucket_calibration(p_raw, cal: dict):
    w = np.array([cal[f"w{k}"] for k in range(4)], dtype=float)
    adj = np.clip(p_raw, 1e-12, 1.0) ** float(cal["gamma"]) * w[None, :]
    den = adj.sum(axis=1, keepdims=True)
    return adj / np.where(den <= 0, 1.0, den)


def _pad(pr, classes):
    """XGBoost drops absent classes; restore the full 4-column shape."""
    if pr.shape[1] >= 4:
        return pr
    o = np.zeros((len(pr), 4))
    for i, c in enumerate(classes):
        o[:, int(c)] = pr[:, i]
    return o


def compose(p60, non60, p_bucket, bucket_preds):
    """The decomposition itself, kept in one place so it cannot drift."""
    return p60 * (p_bucket * bucket_preds).sum(1) + (1.0 - p60) * non60


def fit_blend_weight(mb_val, dir_val, val_df, y_val, grid=np.arange(0.0, 0.55, 0.05)) -> float:
    """Choose the direct-model weight on validation, by the ranking objective.

    Deliberately not by RMSE: optimising a metric the model is not judged on picks a
    different weight. Scored per gameweek at player level, exactly as evaluation does.
    """
    tmp = pd.DataFrame({"gameweek": val_df["gameweek"].to_numpy(),
                        "fpl_code": val_df["fpl_code"].to_numpy(),
                        "actual": y_val})
    best_w, best = 0.0, -np.inf
    for w in grid:
        tmp["p"] = (1.0 - w) * mb_val + w * dir_val
        g = tmp.groupby(["gameweek", "fpl_code"], as_index=False).agg(
            p=("p", "sum"), actual=("actual", "sum"))
        objs = []
        for _, gg in g.groupby("gameweek"):
            if len(gg) < 20:
                continue
            r, n = spearman(gg["p"], gg["actual"]), ndcg(gg["p"], gg["actual"])
            if not (math.isnan(r) or math.isnan(n)):
                objs.append(0.5 * r + 0.5 * n)
        if objs and float(np.mean(objs)) > best:
            best, best_w = float(np.mean(objs)), float(w)
    return best_w


def fit_position(train, val, feats, pos, n_jobs=8, seed=42):
    """Fit all five heads plus the calibration and blend weight for one position.

    Returns (models, calibration, blend_weight). Positions are fitted independently:
    a goalkeeper's points come from saves and clean sheets, a forward's from goals,
    and nothing about the mapping from features to points is shared between them.
    """
    t, v = train[train["position"].eq(pos)], val[val["position"].eq(pos)]
    Xt, Xv = (d[feats].fillna(0.0).to_numpy() for d in (t, v))
    yt, yv = t["target_points"].to_numpy(), v["target_points"].to_numpy()
    clf, reg, ces, res = params(n_jobs, seed)

    sub = t["minutes"].fillna(0).lt(60).to_numpy()
    st = ~sub
    bt = t["target_bucket"].to_numpy()
    v_sub = v["minutes"].fillna(0).lt(60).to_numpy()
    v_st = ~v_sub
    v_bkt = np.digitize(yv, BUCKET_EDGES)

    m_p60 = xgb.XGBClassifier(**clf, early_stopping_rounds=ces, eval_metric="logloss")
    m_p60.fit(Xt, t["target_minutes_ge60"].to_numpy(),
              eval_set=[(Xv, v["minutes"].fillna(0).ge(60).astype(int).to_numpy())],
              verbose=False)

    m_n60 = xgb.XGBRegressor(**reg, early_stopping_rounds=res, eval_metric="rmse")
    m_n60.fit(Xt[sub], yt[sub],
              eval_set=[(Xv[v_sub], yv[v_sub])] if v_sub.sum() >= MIN_ES else [(Xv, yv)],
              verbose=False)

    m_b = xgb.XGBClassifier(**clf, early_stopping_rounds=ces, num_class=4,
                            objective="multi:softprob", eval_metric="mlogloss")
    m_b.fit(Xt[st], bt[st],
            eval_set=[(Xv[v_st], v_bkt[v_st])] if v_st.sum() >= MIN_ES else [(Xv, v_bkt)],
            verbose=False)

    cal = (fit_bucket_calibration(_pad(m_b.predict_proba(Xv[v_st]), m_b.classes_),
                                  v_bkt[v_st], pos)
           if v_st.sum() >= MIN_ES else None)

    bucket_models = []
    for k in range(4):
        in_k = st & (bt == k)
        r = xgb.XGBRegressor(**reg, early_stopping_rounds=res, eval_metric="rmse")
        fx, fy = (Xt[in_k], yt[in_k]) if in_k.sum() > 10 else (Xt[st], yt[st])
        vk = v_st & (v_bkt == k)
        r.fit(fx, fy,
              eval_set=[(Xv[vk], yv[vk])] if vk.sum() >= MIN_ES else [(Xv[v_st], yv[v_st])],
              verbose=False)
        bucket_models.append(r)

    m_dir = xgb.XGBRegressor(**reg, early_stopping_rounds=res,
                             objective="reg:squarederror", eval_metric="rmse")
    m_dir.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)

    models = {"p60": m_p60, "non60": m_n60, "bucket": m_b,
              "bucketreg": bucket_models, "direct": m_dir}
    mb_val = predict_multibucket(models, cal, Xv)
    w = fit_blend_weight(mb_val, np.clip(m_dir.predict(Xv), 0.0, None), v, yv)
    return models, cal, round(float(w), 3)


def predict_multibucket(models, cal, X) -> np.ndarray:
    p60 = models["p60"].predict_proba(X)[:, 1]
    non60 = np.clip(models["non60"].predict(X), 0.0, None)
    pb = _pad(models["bucket"].predict_proba(X), models["bucket"].classes_)
    if cal is not None:
        pb = apply_bucket_calibration(pb, cal)
    reg = np.column_stack([np.clip(r.predict(X), 0.0, None) for r in models["bucketreg"]])
    return compose(p60, non60, pb, reg)


def predict(models, cal, blend_w, X) -> np.ndarray:
    """Final xPts. Clipped at zero: FPL points can be negative, but a *projection*
    of them below zero is never the useful answer for squad selection."""
    mb = predict_multibucket(models, cal, X)
    direct = np.clip(models["direct"].predict(X), 0.0, None)
    return np.clip((1.0 - blend_w) * mb + blend_w * direct, 0.0, None)
