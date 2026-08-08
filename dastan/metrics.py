"""Scoring. Every metric is computed inside a gameweek, then averaged.

This is the single most important decision in the evaluation, so it is worth being
explicit about. Pooling all player-gameweeks in a season and computing one rank
correlation flatters a model enormously, because most of the variance is then
*between* gameweeks -- a blank week against a double gameweek, an easy fixture run
against a hard one. A model can score well on pooled Spearman while being useless at
the only question a manager actually faces: given this week, who do I pick?

So: group by gameweek, score within it, average the per-gameweek results.

Two cohorts are always reported and they answer different questions:

  all       every player in the pool. Roughly 62% recorded zero minutes, so a model
            is rewarded mostly for knowing who will not play. This is the number that
            moves most when availability information improves.
  starters  players who actually played 60+ minutes. Much harder, much noisier, and
            closer to the real decision -- these are the players you were choosing
            between. Never optimise against this; report it, so that a model which
            wins only by predicting non-players is visible rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def spearman(pred, actual) -> float:
    """Rank correlation. NaN below 3 points, where it is meaningless."""
    if len(pred) < 3:
        return float("nan")
    r, _ = stats.spearmanr(pred, actual)
    return float(r)


def ndcg(pred, actual, k: int = 10) -> float:
    """Normalised discounted cumulative gain at k.

    Rank correlation treats every pair equally, but FPL does not: being wrong about
    the 200th-best player costs nothing, and being wrong about the best costs a
    week. NDCG@10 asks how much of the achievable points mass the top 10 predictions
    actually captured, discounted by position.
    """
    g = np.asarray(actual, dtype=float)[np.argsort(-np.asarray(pred), kind="mergesort")][:k]
    d = 1.0 / np.log2(np.arange(2, len(g) + 2))
    ideal = np.sort(np.asarray(actual, dtype=float))[::-1][:k]
    idcg = float((ideal * d[: len(ideal)]).sum())
    return float((g * d).sum()) / idcg if idcg > 0 else float("nan")


def objective(pred, actual) -> float:
    """0.5 * Spearman + 0.5 * NDCG@10 -- the number the model is selected on.

    Half of it cares about the whole ordering, half only about the top. Neither
    alone is right: pure rank correlation ignores that the top matters more, and
    pure NDCG ignores that a squad is fifteen players, not ten.
    """
    return 0.5 * spearman(pred, actual) + 0.5 * ndcg(pred, actual)


def collapse_to_player_gameweek(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """One row per player per gameweek.

    A double gameweek is two fixtures but one decision, so the fixtures are summed.
    Scoring at fixture grain instead counts a double-gameweek player twice and
    quietly changes what the metric means.
    """
    return df.groupby(["season", "gameweek", "fpl_code"], as_index=False).agg(
        **{pred_col: (pred_col, "sum")},
        actual=("actual", "sum"),
        minutes=("minutes", "sum"),
    )


def score_by_gameweek(df: pd.DataFrame, col: str = "pred", cohort: str = "all",
                      min_players: int = 20) -> pd.DataFrame:
    """Return one metric row per scorable gameweek.

    Missing and non-finite predictions are removed before a gameweek is scored. This
    is important for rolling-history baselines: promoted players and early-season
    rows legitimately have no five-match history. Letting one NaN reach Spearman
    silently discarded the *whole* gameweek in earlier reports.
    """
    d = df if cohort == "all" else df[df["minutes"] >= 60]
    numeric = d[[col, "actual"]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric[col].to_numpy()) & np.isfinite(
        numeric["actual"].to_numpy()
    )
    d = d.loc[finite].copy()
    d[col] = numeric.loc[finite, col]
    d["actual"] = numeric.loc[finite, "actual"]
    rows = []
    group_keys = ["season", "gameweek"] if "season" in d.columns else ["gameweek"]
    for group, g in d.groupby(group_keys, sort=True):
        if len(g) < min_players:
            continue
        sp, nd = spearman(g[col], g["actual"]), ndcg(g[col], g["actual"])
        if np.isnan(sp) or np.isnan(nd):
            continue
        pred_top = set(g.nlargest(10, col)["fpl_code"])
        actual_top = set(g.nlargest(10, "actual")["fpl_code"])
        gameweek = group[-1] if isinstance(group, tuple) else group
        row = {
            "gameweek": int(gameweek),
            "rows": int(len(g)),
            "obj": 0.5 * sp + 0.5 * nd,
            "spearman": sp,
            "ndcg@10": nd,
            "mae": float((g[col] - g["actual"]).abs().mean()),
            "rmse": float(np.sqrt(((g[col] - g["actual"]) ** 2).mean())),
            "top10_overlap": len(pred_top & actual_top) / 10.0,
            "captain_in_top10": float(
                g.nlargest(1, col)["fpl_code"].iloc[0] in actual_top
            ),
            "mean_pred": float(g[col].mean()),
            "mean_actual": float(g["actual"].mean()),
        }
        if "season" in d.columns:
            row["season"] = str(group[0])
        rows.append(row)
    return pd.DataFrame(rows)


def score(df: pd.DataFrame, col: str = "pred", cohort: str = "all",
          min_players: int = 20) -> dict:
    """Per-gameweek metrics, averaged. `df` must be at player-gameweek grain.

    Returns objective, its two components, error metrics, and two practical ones:

      top10_overlap     how many of the actual top 10 scorers were in our top 10.
      captain_in_top10  how often our single highest pick finished in the actual
                        top 10 -- the closest thing here to a captaincy question.
    """
    d = df if cohort == "all" else df[df["minutes"] >= 60]
    numeric = d[[col, "actual"]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric[col].to_numpy()) & np.isfinite(
        numeric["actual"].to_numpy()
    )
    scored_rows = int(finite.sum())
    rows = score_by_gameweek(df, col, cohort, min_players)
    keys = ("obj", "spearman", "ndcg@10", "mae", "rmse", "top10_overlap",
            "captain_in_top10", "mean_pred", "mean_actual")
    if rows.empty:
        return {k: float("nan") for k in keys} | {
            "gameweeks": 0,
            "rows": scored_rows,
        }
    return ({k: round(float(rows[k].mean()), 4) for k in keys}
            | {"gameweeks": len(rows), "rows": scored_rows})


def auc(y_true, p) -> float:
    """Rank-based AUC, no sklearn dependency."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y_true, p) -> float:
    """Mean squared error of a probability. Lower is better; measures calibration
    and sharpness together, which AUC (rank-only) cannot see."""
    return float(((np.asarray(p, dtype=float) - np.asarray(y_true, dtype=int)) ** 2).mean())
