#!/usr/bin/env python3
"""Build or verify the published FPL-code-to-Understat-ID mapping.

    python -m dastan.mappings
    python -m dastan.mappings --write

The mapping is a projection of the joins already present in the published training
frame. It therefore records exactly which identities Dastan used; it is not a live
identity service or an independently re-matched third-party dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import DATA, FRAME, SEASONS

PLAYERS = DATA / "players.csv"
MAPPING = DATA / "mappings" / "fpl_understat_players.csv"
MAPPING_COLUMNS = [
    "fpl_code",
    "understat_id",
    "player_name",
    "position",
    "first_season",
    "last_season",
    "mapping_status",
]


def build() -> pd.DataFrame:
    """Derive one row per mapped FPL code from the published training data."""
    frame = pd.read_parquet(FRAME, columns=["fpl_code", "understat_id"])
    pairs = frame.dropna(subset=["understat_id"])[["fpl_code", "understat_id"]].copy()
    pairs["fpl_code"] = pairs["fpl_code"].astype("int64")
    pairs["understat_id"] = pairs["understat_id"].astype("int64")
    pairs = pairs.drop_duplicates()
    if (pairs[["fpl_code", "understat_id"]] <= 0).any().any():
        raise RuntimeError("mapping contains non-positive provider IDs")

    inconsistent = pairs.groupby("fpl_code")["understat_id"].nunique()
    inconsistent = inconsistent[inconsistent > 1]
    if len(inconsistent):
        raise RuntimeError(
            f"{len(inconsistent)} FPL codes map to multiple Understat IDs: "
            f"{inconsistent.index[:5].tolist()}"
        )

    players = pd.read_csv(PLAYERS)
    unknown_seasons = sorted(set(players["season"]) - set(SEASONS))
    if unknown_seasons:
        raise RuntimeError(f"players.csv contains unknown seasons: {unknown_seasons}")
    players["season_order"] = players["season"].map({s: i for i, s in enumerate(SEASONS)})
    players = players.sort_values(["fpl_code", "season_order"], kind="mergesort")
    identity = players.groupby("fpl_code", as_index=False).agg(
        player_name=("player_name", "last"),
        position=("position", "last"),
        first_season=("season", "first"),
        last_season=("season", "last"),
    )

    out = pairs.merge(identity, on="fpl_code", how="left", validate="one_to_one")
    if out["player_name"].isna().any():
        missing = out.loc[out["player_name"].isna(), "fpl_code"].tolist()
        raise RuntimeError(f"players.csv is missing mapped FPL codes: {missing[:5]}")

    shared = out["understat_id"].duplicated(keep=False)
    out["mapping_status"] = "mapped"
    out.loc[shared, "mapping_status"] = "shared_understat_id"
    return out[MAPPING_COLUMNS].sort_values("fpl_code", kind="mergesort").reset_index(drop=True)


def load(path: Path = MAPPING) -> pd.DataFrame:
    """Load the published mapping and enforce its public schema."""
    out = pd.read_csv(path)
    if list(out.columns) != MAPPING_COLUMNS:
        raise RuntimeError(
            f"unexpected mapping columns: expected {MAPPING_COLUMNS}, got {list(out.columns)}"
        )
    if out["fpl_code"].duplicated().any():
        raise RuntimeError("published mapping contains duplicate fpl_code values")
    if out[MAPPING_COLUMNS].isna().any().any():
        raise RuntimeError("published mapping contains missing values")
    for column in ("fpl_code", "understat_id"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")
    if (out[["fpl_code", "understat_id"]] <= 0).any().any():
        raise RuntimeError("published mapping contains non-positive provider IDs")
    expected_status = out["understat_id"].duplicated(keep=False).map(
        {False: "mapped", True: "shared_understat_id"}
    )
    if not out["mapping_status"].equals(expected_status):
        raise RuntimeError("published mapping has inconsistent mapping_status values")
    return out


def verify(path: Path = MAPPING) -> pd.DataFrame:
    """Assert that the checked-in CSV exactly matches the published frame."""
    expected = build()
    published = load(path)
    try:
        pd.testing.assert_frame_equal(published, expected, check_dtype=False)
    except AssertionError as exc:
        raise RuntimeError(
            "published mapping differs from the training frame; "
            "run `python -m dastan.mappings --write`"
        ) from exc
    return published


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="regenerate the checked-in CSV")
    args = ap.parse_args()

    if args.write:
        mapping = build()
        MAPPING.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(MAPPING, index=False)
        action = "wrote"
    else:
        mapping = verify()
        action = "verified"

    shared_ids = mapping.loc[
        mapping["mapping_status"].eq("shared_understat_id"), "understat_id"
    ].nunique()
    print(
        f"{action} {len(mapping):,} FPL-to-Understat mappings | "
        f"{mapping['understat_id'].nunique():,} Understat IDs | "
        f"{shared_ids} shared IDs flagged"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
