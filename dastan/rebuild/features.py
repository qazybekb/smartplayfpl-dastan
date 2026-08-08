"""Rebuild Dastan's deadline-anchored feature frame from canonical match tables.

The rolling feature definitions originate in OpenFPL and are used here under its
MIT licence. Dastan adds availability history, defensive-contribution history,
congestion, targets, and deadline anchoring.
"""

from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd

from .. import data

WINDOWS = [1, 3, 5, 10, 38]
POSITION_BASE_RATES = {
    "GKP": {"p_any": 0.50, "p60": 0.48, "e_minutes": 45.0},
    "DEF": {"p_any": 0.55, "p60": 0.42, "e_minutes": 42.0},
    "MID": {"p_any": 0.52, "p60": 0.38, "e_minutes": 38.0},
    "FWD": {"p_any": 0.50, "p60": 0.35, "e_minutes": 35.0},
}
FPL_TO_UNDERSTAT = {
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Spurs": "Tottenham",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
}
PLAYER_FEATURES = {
    "total_points": "fpl_points",
    "minutes": "minutes_played",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "goals_scored": "goals_scored",
    "penalties_missed": "penalties_missed",
    "assists": "assists",
    "goals_conceded": "goals_conceded",
    "own_goals": "own_goals",
    "saves": "saves",
    "penalties_saved": "penalties_saved",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "bps": "bps",
    "bonus": "fpl_bonus_points",
    "us_shots": "shots",
    "us_xG": "xg",
    "us_xGChain": "xgchain",
    "us_xGBuildup": "xgbuildup",
    "us_key_passes": "key_passes",
    "us_xA": "xa",
}
TEAM_FEATURES = {
    "scored": "goals_scored",
    "missed": "goals_conceded",
    "league_rank": "league_rank",
    "opp_league_rank": "opponent_league_rank",
    "xG": "xg",
    "deep_allowed": "deep_allowed",
    "ppda_allowed_att": "ppda_allowed_att",
    "ppda_allowed_def": "ppda_allowed_def",
    "xGA": "xga",
    "deep": "deep",
    "ppda_att": "ppda_att",
    "ppda_def": "ppda_def",
}
TEAM_NUMERIC = [
    "scored",
    "missed",
    "xG",
    "xGA",
    "deep",
    "deep_allowed",
    "ppda_att",
    "ppda_def",
    "ppda_allowed_att",
    "ppda_allowed_def",
]
PLAYER_NUMERIC = list(PLAYER_FEATURES) + [
    "us_team_xG",
    "us_team_xGA",
    "us_deep",
    "us_deep_allowed",
    "team_h_score",
    "team_a_score",
    "expected_points_pre_deadline",
    "value",
    "selected",
    "transfers_balance",
]
DEFENSIVE_STATS = [
    "defensive_contribution",
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
]
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12, "GKP": 99}


def fpl_to_understat(name: str) -> str:
    return FPL_TO_UNDERSTAT.get(str(name), str(name))


def _checked_merge(
    left: pd.DataFrame, right: pd.DataFrame, label: str, **kwargs
) -> pd.DataFrame:
    before = len(left)
    out = left.merge(right, how="left", **kwargs)
    if len(out) != before:
        raise RuntimeError(f"{label} merge fanned out: {before:,} -> {len(out):,}")
    return out


def _assign_bucket(points: pd.Series) -> pd.Series:
    values = pd.to_numeric(points, errors="coerce").fillna(0).to_numpy()
    out = np.zeros(len(values), dtype=int)
    out[values >= 1] = 1
    out[values >= 3] = 2
    out[values >= 10] = 3
    return pd.Series(out, index=points.index)


def compute_player_rolling(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.sort_values(["player_uid", "kickoff_time"], kind="mergesort").copy()
    columns: dict[str, pd.Series] = {}
    for source, feature in PLAYER_FEATURES.items():
        grouped = out.groupby("player_uid", sort=False)[source]
        for window in WINDOWS:
            name = f"player_{feature}_{window}"
            columns[name] = grouped.transform(
                lambda values, w=window: values.shift(1)
                .rolling(w, min_periods=1)
                .mean()
            )
    venue_points = out.groupby(["player_uid", "is_home"], sort=False)["total_points"]
    for window in WINDOWS:
        name = f"player_relevant_fpl_points_{window}"
        columns[name] = venue_points.transform(
            lambda values, w=window: values.shift(1).rolling(w, min_periods=1).mean()
        )
    added = pd.DataFrame(columns, index=out.index)
    return pd.concat([out, added], axis=1), list(columns)


def build_opponent_lookup(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Pair home and away team rows using score and xG within each kickoff slot."""
    out = team_matches.copy()
    out["date_str"] = out["date"].dt.strftime("%Y-%m-%d %H:%M")
    lookup: dict[tuple[str, str, str], str] = {}
    for (season, slot), group in out.groupby(["season", "date_str"], sort=False):
        homes = group[group["is_home"].eq(1.0)]
        aways = group[group["is_home"].eq(0.0)]
        used: set[str] = set()
        for _, home in homes.iterrows():
            best_name, best_difference = None, float("inf")
            for _, away in aways.iterrows():
                away_name = str(away["understat_team"])
                if away_name in used:
                    continue
                score_matches = (
                    home["scored"] == away["missed"]
                    and home["missed"] == away["scored"]
                )
                if not score_matches:
                    continue
                difference = abs(home["xG"] - away["xGA"]) + abs(
                    home["xGA"] - away["xG"]
                )
                if difference < best_difference:
                    best_name, best_difference = away_name, difference
            if best_name is not None:
                home_name = str(home["understat_team"])
                lookup[(str(season), home_name, str(slot))] = best_name
                lookup[(str(season), best_name, str(slot))] = home_name
                used.add(best_name)
    out["opponent"] = [
        lookup.get((str(row.season), str(row.understat_team), str(row.date_str)))
        for row in out.itertuples()
    ]
    return out


def compute_league_ranks(team_matches: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    ordered = team_matches.sort_values(
        ["season", "date", "understat_team"], kind="mergesort"
    )
    ranks: dict[tuple[str, str, str], int] = {}
    for season, season_rows in ordered.groupby("season", sort=False):
        teams = list(season_rows["understat_team"].dropna().unique())
        totals = {team: {"gd": 0.0, "gs": 0.0, "pts": 0.0} for team in teams}
        for kickoff, matches in season_rows.groupby("date", sort=True):
            table = sorted(
                teams,
                key=lambda team: (
                    -totals[team]["pts"],
                    -totals[team]["gd"],
                    -totals[team]["gs"],
                ),
            )
            position = {team: rank for rank, team in enumerate(table, 1)}
            date_string = pd.Timestamp(kickoff).strftime("%Y-%m-%d %H:%M")
            for row in matches.itertuples():
                ranks[(str(row.understat_team), str(season), date_string)] = (
                    position.get(row.understat_team, 10)
                )
            for row in matches.itertuples():
                current = totals[row.understat_team]
                current["gd"] += float(row.scored) - float(row.missed)
                current["gs"] += float(row.scored)
                current["pts"] += float(row.pts)
    return ranks


def compute_team_rolling(
    team_matches: pd.DataFrame, ranks: dict[tuple[str, str, str], int]
) -> tuple[pd.DataFrame, list[str]]:
    out = team_matches.sort_values(["understat_team", "date"], kind="mergesort").copy()
    date_strings = out["date"].dt.strftime("%Y-%m-%d %H:%M")
    out["league_rank"] = [
        ranks.get((str(team), str(season), str(date)), 10)
        for team, season, date in zip(
            out["understat_team"], out["season"], date_strings
        )
    ]
    out["opp_league_rank"] = [
        (
            ranks.get((str(opponent), str(season), str(date)), 10)
            if pd.notna(opponent)
            else 10
        )
        for opponent, season, date in zip(out["opponent"], out["season"], date_strings)
    ]
    rolling: dict[str, pd.Series] = {}
    for source, feature in TEAM_FEATURES.items():
        grouped = out.groupby("understat_team", sort=False)[source]
        for window in WINDOWS:
            name = f"team_{feature}_{window}"
            rolling[name] = grouped.transform(
                lambda values, w=window: values.shift(1)
                .rolling(w, min_periods=1)
                .mean()
            )
    out = pd.concat([out, pd.DataFrame(rolling, index=out.index)], axis=1)
    out["status_league_rank"] = out["league_rank"]
    out["status_opp_league_rank"] = out["opp_league_rank"]
    return out, list(rolling)


def add_availability_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(
        ["fpl_code", "season", "kickoff_time"], kind="mergesort"
    ).copy()
    out["_played"] = out["minutes"].gt(0).astype(float)
    out["_played60"] = out["minutes"].ge(60).astype(float)
    grouped = out.groupby(["fpl_code", "season"], sort=False)
    for window in [3, 5, 10]:
        for source, prefix in (
            ("_played", "p_any"),
            ("_played60", "p60"),
            ("minutes", "e_minutes"),
        ):
            out[f"{prefix}_{window}"] = grouped[source].transform(
                lambda values, w=window: values.shift(1)
                .rolling(w, min_periods=1)
                .mean()
            )
    for position, defaults in POSITION_BASE_RATES.items():
        mask = out["position"].eq(position)
        for window in [3, 5, 10]:
            for prefix in ("p_any", "p60", "e_minutes"):
                out.loc[mask, f"{prefix}_{window}"] = out.loc[
                    mask, f"{prefix}_{window}"
                ].fillna(defaults[prefix])
    return out.drop(columns=["_played", "_played60"])


def add_dastan_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["player_uid", "kickoff_time"], kind="mergesort").copy()
    grouped = out.groupby("player_uid", sort=False)
    source_names = {
        "defensive_contribution": "defensive_contribution",
        "cbit": "cbit",
        "recoveries": "recoveries",
        "tackles": "tackles",
    }
    for source, feature in source_names.items():
        for window in [3, 5, 10, 38]:
            out[f"ar_{feature}_{window}"] = grouped[source].transform(
                lambda values, w=window: values.shift(1)
                .rolling(w, min_periods=1)
                .mean()
            )

    threshold = out["position"].map(DC_THRESHOLD).astype(float)
    out["_dc_hit"] = out["defensive_contribution"].ge(threshold).astype(float)
    out["_dc_near"] = (
        out["defensive_contribution"].ge(threshold - 3)
        & out["defensive_contribution"].lt(threshold)
    ).astype(float)
    out.loc[out["defensive_contribution"].isna(), ["_dc_hit", "_dc_near"]] = np.nan
    grouped = out.groupby("player_uid", sort=False)
    for window in [3, 5, 10]:
        out[f"ar_dc_hit_rate_{window}"] = grouped["_dc_hit"].transform(
            lambda values, w=window: values.shift(1).rolling(w, min_periods=1).mean()
        )
        out[f"ar_dc_near_rate_{window}"] = grouped["_dc_near"].transform(
            lambda values, w=window: values.shift(1).rolling(w, min_periods=1).mean()
        )
        out[f"ar_starts_{window}"] = grouped["starts"].transform(
            lambda values, w=window: values.shift(1).rolling(w, min_periods=1).mean()
        )
    out["ar_dc_ppg_5"] = 2.0 * out["ar_dc_hit_rate_5"]
    out["ar_rest_days"] = (
        (out["kickoff_time"] - grouped["kickoff_time"].shift(1))
        .dt.total_seconds()
        .div(86_400)
        .clip(1, 14)
        .fillna(14)
    )

    team_fixtures = (
        out[["season", "team_name", "fixture", "kickoff_time"]]
        .drop_duplicates(["season", "team_name", "fixture"])
        .sort_values(["season", "team_name", "kickoff_time"], kind="mergesort")
    )
    density = []
    for _, rows in team_fixtures.groupby(["season", "team_name"], sort=False):
        kickoffs = rows["kickoff_time"].astype("int64").to_numpy()
        fourteen_days = pd.Timedelta(days=14).value
        values = [
            int(((kickoffs[index] - kickoffs[:index]) <= fourteen_days).sum())
            for index in range(len(kickoffs))
        ]
        density.append(pd.Series(values, index=rows.index))
    team_fixtures["ar_team_matches_14d"] = pd.concat(density).sort_index()
    out = _checked_merge(
        out,
        team_fixtures[["season", "team_name", "fixture", "ar_team_matches_14d"]],
        "team congestion",
        on=["season", "team_name", "fixture"],
        validate="many_to_one",
    )

    allowed = (
        out.groupby(["season", "fixture", "team_name"], as_index=False)
        .agg(
            dc_sum=("defensive_contribution", "sum"),
            tackle_sum=("tackles", "sum"),
            conceding_team=("opponent_team_name", "first"),
            kickoff=("kickoff_time", "first"),
        )
        .sort_values(["season", "conceding_team", "kickoff"], kind="mergesort")
    )
    grouped_allowed = allowed.groupby(["season", "conceding_team"], sort=False)
    for window in [3, 5, 10]:
        allowed[f"ar_opp_dc_allowed_{window}"] = grouped_allowed["dc_sum"].transform(
            lambda values, w=window: values.shift(1).rolling(w, min_periods=1).mean()
        )
    allowed["ar_opp_tackles_allowed_5"] = grouped_allowed["tackle_sum"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )
    opponent_columns = [
        "season",
        "fixture",
        "conceding_team",
        "ar_opp_dc_allowed_3",
        "ar_opp_dc_allowed_5",
        "ar_opp_dc_allowed_10",
        "ar_opp_tackles_allowed_5",
    ]
    out = _checked_merge(
        out,
        allowed[opponent_columns],
        "opponent defensive contributions",
        left_on=["season", "fixture", "opponent_team_name"],
        right_on=["season", "fixture", "conceding_team"],
        validate="many_to_one",
    ).drop(columns=["conceding_team", "_dc_hit", "_dc_near"], errors="ignore")
    return out


def _prepare_team_matches(team_matches: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "date",
        "understat_team",
        "is_home",
        "scored",
        "missed",
        "xG",
        "xGA",
    }
    missing = sorted(required - set(team_matches.columns))
    if missing:
        raise RuntimeError(f"team matches are missing columns: {missing}")
    out = team_matches.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    if out["date"].isna().any():
        raise RuntimeError("team matches contain invalid dates")
    out["is_home"] = (
        pd.to_numeric(out["is_home"], errors="coerce").fillna(0).astype(float)
    )
    for column in TEAM_NUMERIC:
        if column not in out:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    if "pts" not in out:
        out["pts"] = np.where(
            out["scored"] > out["missed"],
            3,
            np.where(out["scored"] == out["missed"], 1, 0),
        )
    out["pts"] = pd.to_numeric(out["pts"], errors="coerce").fillna(0.0)
    duplicates = out.duplicated(["season", "understat_team", "date"])
    if duplicates.any():
        raise RuntimeError(
            f"team matches contain {int(duplicates.sum()):,} duplicate team kickoffs"
        )
    return out


def build_feature_frame(
    player_matches: pd.DataFrame, team_matches: pd.DataFrame
) -> pd.DataFrame:
    """Build the 304-column release-schema frame from canonical raw matches."""
    required = {
        "season",
        "gameweek",
        "fixture",
        "fpl_code",
        "element",
        "position",
        "team_name",
        "opponent_team_name",
        "us_opponent",
        "kickoff_time",
        "minutes",
        "total_points",
        "is_home",
        "understat_id",
    }
    missing = sorted(required - set(player_matches.columns))
    if missing:
        raise RuntimeError(f"player matches are missing columns: {missing}")
    frame = player_matches.copy()
    frame["kickoff_time"] = pd.to_datetime(
        frame["kickoff_time"], utc=True, errors="coerce"
    )
    frame["match_date"] = pd.to_datetime(
        frame.get("match_date", frame["kickoff_time"]), utc=True, errors="coerce"
    )
    if frame[["kickoff_time", "match_date"]].isna().any().any():
        raise RuntimeError("player matches contain invalid dates")
    frame["team_us"] = frame["team_name"].map(fpl_to_understat)
    frame["opponent_us"] = frame["us_opponent"].map(fpl_to_understat)
    frame["is_home"] = (
        pd.to_numeric(frame["is_home"], errors="coerce").fillna(0).astype(bool)
    )
    frame["position"] = frame["position"].replace({"GK": "GKP"}).astype(str)
    frame["player_uid"] = (
        frame["understat_id"]
        .where(
            frame["understat_id"].notna(),
            frame["element"].astype(str) + "_" + frame["season"].astype(str),
        )
        .astype(str)
    )
    for column in PLAYER_NUMERIC:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in DEFENSIVE_STATS + ["starts"]:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    duplicate_keys = frame.duplicated(["season", "gameweek", "fixture", "fpl_code"])
    if duplicate_keys.any():
        raise RuntimeError(
            f"player matches contain {int(duplicate_keys.sum()):,} duplicate fixture keys"
        )

    print("  computing player rolling features", flush=True)
    frame, _ = compute_player_rolling(frame)
    teams = _prepare_team_matches(team_matches)
    teams = build_opponent_lookup(teams)
    ranks = compute_league_ranks(teams)
    teams, team_columns = compute_team_rolling(teams, ranks)

    join_columns = [
        "understat_team",
        "season",
        "date",
        *team_columns,
        "status_league_rank",
        "status_opp_league_rank",
    ]
    team_features = teams[join_columns].copy()
    team_features["_merge_date"] = team_features["date"].dt.date
    team_features = team_features.drop(columns=["date"])
    if team_features.duplicated(["understat_team", "season", "_merge_date"]).any():
        raise RuntimeError("team feature table is not unique by team, season, and date")
    frame["_merge_date"] = frame["kickoff_time"].dt.date
    frame = _checked_merge(
        frame,
        team_features,
        "own-team features",
        left_on=["team_us", "season", "_merge_date"],
        right_on=["understat_team", "season", "_merge_date"],
        validate="many_to_one",
    ).drop(columns=["understat_team"], errors="ignore")

    opponent_rename = {
        column: column.replace("team_", "opp_", 1) for column in team_columns
    }
    opponent_rename.update(
        {
            "status_league_rank": "opp_status_league_rank",
            "status_opp_league_rank": "opp_status_opp_league_rank",
        }
    )
    opponent_features = team_features.rename(columns=opponent_rename)
    opponent_columns = [
        column
        for column in opponent_features.columns
        if column.startswith("opp_")
        or column in {"understat_team", "season", "_merge_date"}
    ]
    frame = _checked_merge(
        frame,
        opponent_features[opponent_columns],
        "opponent features",
        left_on=["opponent_us", "season", "_merge_date"],
        right_on=["understat_team", "season", "_merge_date"],
        validate="many_to_one",
    ).drop(columns=["understat_team", "_merge_date"], errors="ignore")

    print("  computing Dastan feature families", flush=True)
    frame = add_availability_features(frame)
    frame["is_home_num"] = frame["is_home"].astype(int)
    for column in ("status_league_rank", "opp_status_league_rank"):
        if column not in frame:
            frame[column] = np.nan
    frame["target_points"] = frame["total_points"].fillna(0.0)
    frame["target_minutes_ge60"] = frame["minutes"].ge(60).astype(int)
    frame["target_bucket"] = _assign_bucket(frame["target_points"])
    frame = frame.rename(columns={"clearances_blocks_interceptions": "cbit"})
    frame = add_dastan_features(frame)
    frame = data.anchor_to_deadline(frame)
    data.assert_deadline_anchored(frame)

    release_columns = list(pd.read_parquet(data.FRAME).columns)
    core_features = json.loads(
        (data.ROOT / "models" / "core_feature_cols.json").read_text()
    )
    for column in release_columns:
        if column not in frame:
            if column in core_features:
                frame[column] = 0.0
            else:
                raise RuntimeError(f"rebuilt frame is missing release column {column}")
    # Preserve the player_uid/kickoff ordering used by the released training run.
    # XGBoost is row-order sensitive even with a fixed seed.
    return frame[release_columns].reset_index(drop=True)


def parse_ppda(value) -> tuple[float, float]:
    """Normalize Understat's dict-or-string PPDA representation."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return 0.0, 0.0
    if not isinstance(value, dict):
        return 0.0, 0.0
    return float(value.get("att", 0) or 0), float(value.get("def", 0) or 0)
