#!/usr/bin/env python3
"""Build and verify Dastan's FPL-to-Understat identity artifacts.

There are deliberately two historical mappings:

``training``
    The immutable IDs present in the released feature frame. Use this to reproduce
    the released model exactly, including two historical identity mistakes.

``corrected``
    The same population after applying the checked-in, evidence-backed identity
    audit. Use this when joining new data.

The current-season map is built separately from a captured FPL roster. It retains
unresolved players instead of guessing an Understat ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import DATA, FRAME, SEASONS

PLAYERS = DATA / "players.csv"
MAPPINGS = DATA / "mappings"
TRAINING_MAPPING = MAPPINGS / "fpl_understat_training_snapshot.csv"
TRAINING_ASSIGNMENTS = MAPPINGS / "fpl_understat_training_assignments.csv"
CORRECTED_MAPPING = MAPPINGS / "fpl_understat_players.csv"
IDENTITY_AUDIT = MAPPINGS / "fpl_understat_identity_audit.csv"
CURRENT_ROSTER = MAPPINGS / "fpl_players_current.csv"
CURRENT_MAPPING = MAPPINGS / "fpl_understat_current.csv"
CURRENT_META = MAPPINGS / "fpl_understat_current.json"
OPERATIONAL_PLAYERS = MAPPINGS / "current_fpl_understat_players.csv"
OPERATIONAL_CLUBS = MAPPINGS / "current_fpl_understat_clubs.csv"
OPERATIONAL_MANIFEST = MAPPINGS / "current_manifest.json"

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

MAPPING_COLUMNS = [
    "fpl_code",
    "understat_id",
    "player_name",
    "position",
    "first_season",
    "last_season",
    "mapping_status",
]
ASSIGNMENT_COLUMNS = [
    "season",
    "fpl_code",
    "understat_id",
    "first_gameweek",
    "last_gameweek",
    "mapped_rows",
]
AUDIT_COLUMNS = [
    "fpl_code",
    "release_understat_id",
    "decision_understat_id",
    "decision",
    "verified_at",
    "release_identity",
    "decision_identity",
    "release_evidence_url",
    "decision_evidence_url",
    "note",
]
ROSTER_COLUMNS = [
    "season",
    "fpl_code",
    "element",
    "player_name",
    "position",
    "team_name",
]
OPERATIONAL_PLAYER_COLUMNS = [
    "fpl_code",
    "understat_player_id",
    "fpl_player_name",
    "fpl_position",
    "confidence_level",
    "mapping_status",
]
OPERATIONAL_CLUB_COLUMNS = [
    "club_name",
    "club_short",
    "fpl_team_code",
    "understat_name",
    "understat_team_id",
    "mapping_status",
]
CURRENT_COLUMNS = ROSTER_COLUMNS + [
    "understat_id",
    "understat_player_name",
    "mapping_status",
    "confidence",
    "source",
]


def _shared_status(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["understat_id"]
        .duplicated(keep=False)
        .map({False: "mapped", True: "shared_understat_id"})
    )


def build_training() -> pd.DataFrame:
    """Derive the exact release mapping from the published training frame."""
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
    players["season_order"] = players["season"].map(
        {s: i for i, s in enumerate(SEASONS)}
    )
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

    out["mapping_status"] = _shared_status(out)
    return (
        out[MAPPING_COLUMNS]
        .sort_values("fpl_code", kind="mergesort")
        .reset_index(drop=True)
    )


def build_training_assignments() -> pd.DataFrame:
    """Derive the exact season/gameweek intervals used by the release frame."""
    frame = pd.read_parquet(
        FRAME, columns=["season", "gameweek", "fpl_code", "understat_id"]
    )
    mapped = frame.dropna(subset=["understat_id"]).copy()
    for column in ("gameweek", "fpl_code", "understat_id"):
        mapped[column] = pd.to_numeric(mapped[column], errors="raise").astype("int64")
    out = mapped.groupby(["season", "fpl_code", "understat_id"], as_index=False).agg(
        first_gameweek=("gameweek", "min"),
        last_gameweek=("gameweek", "max"),
        mapped_rows=("gameweek", "size"),
    )
    if out.duplicated(["season", "fpl_code"]).any():
        raise RuntimeError("release assignments contain multiple IDs per player-season")

    expected = frame.merge(
        out,
        on=["season", "fpl_code"],
        how="left",
        suffixes=("_release", "_assignment"),
        validate="many_to_one",
    )
    active = expected["gameweek"].between(
        expected["first_gameweek"], expected["last_gameweek"]
    )
    assigned = expected["understat_id_assignment"].where(active)
    release = expected["understat_id_release"]
    same = (release.isna() & assigned.isna()) | release.eq(assigned)
    if not same.all():
        raise RuntimeError(
            "release identity presence is not a single interval per player-season"
        )
    return (
        out[ASSIGNMENT_COLUMNS]
        .sort_values(["season", "fpl_code"], kind="mergesort")
        .reset_index(drop=True)
    )


def load_mapping(path: Path) -> pd.DataFrame:
    """Load a complete historical map and enforce its public schema."""
    out = pd.read_csv(path)
    if list(out.columns) != MAPPING_COLUMNS:
        raise RuntimeError(
            f"unexpected columns in {path.name}: expected {MAPPING_COLUMNS}, got {list(out.columns)}"
        )
    if out["fpl_code"].duplicated().any():
        raise RuntimeError(f"{path.name} contains duplicate fpl_code values")
    if out[MAPPING_COLUMNS].isna().any().any():
        raise RuntimeError(f"{path.name} contains missing values")
    for column in ("fpl_code", "understat_id"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")
    if (out[["fpl_code", "understat_id"]] <= 0).any().any():
        raise RuntimeError(f"{path.name} contains non-positive provider IDs")
    if not out["mapping_status"].equals(_shared_status(out)):
        raise RuntimeError(f"{path.name} has inconsistent mapping_status values")
    return out


def load_training(path: Path = TRAINING_MAPPING) -> pd.DataFrame:
    """Load the immutable mapping used by the released feature frame."""
    return load_mapping(path)


def load_training_assignments(
    path: Path = TRAINING_ASSIGNMENTS,
) -> pd.DataFrame:
    """Load the immutable season/gameweek mapping intervals used by the release."""
    out = pd.read_csv(path)
    if list(out.columns) != ASSIGNMENT_COLUMNS:
        raise RuntimeError(
            f"unexpected training-assignment columns: {list(out.columns)}"
        )
    if out[ASSIGNMENT_COLUMNS].isna().any().any():
        raise RuntimeError("training assignments contain missing values")
    for column in ASSIGNMENT_COLUMNS[1:]:
        out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")
    if out.duplicated(["season", "fpl_code"]).any():
        raise RuntimeError("training assignments contain duplicate player-seasons")
    if (
        (out[["fpl_code", "understat_id", "first_gameweek", "mapped_rows"]] <= 0)
        .any()
        .any()
    ):
        raise RuntimeError("training assignments contain non-positive values")
    if out["first_gameweek"].gt(out["last_gameweek"]).any():
        raise RuntimeError("training assignments contain reversed intervals")
    return out


def assert_operational_clubs_ready(season: str) -> None:
    """Fail before new-season data joins while a live club lacks Understat identity."""
    roster = load_roster()
    roster_season = str(roster["season"].iat[0])
    if season != roster_season:
        raise RuntimeError(
            f"new season {season} does not match the checked-in FPL roster {roster_season}"
        )
    clubs = load_operational_clubs().set_index("club_name")
    current = sorted(set(roster["team_name"].astype(str)))
    missing_rows = sorted(set(current) - set(clubs.index))
    unmapped = [
        name
        for name in current
        if name in clubs.index and pd.isna(clubs.loc[name, "understat_team_id"])
    ]
    if missing_rows or unmapped:
        raise RuntimeError(
            "current club mappings are not retraining-ready: "
            f"missing={missing_rows}, awaiting_understat={unmapped}"
        )


def assignments_for_seasons(seasons: list[str]) -> pd.DataFrame:
    """Use frozen assignments for released seasons and operational IDs for the next.

    The immutable release assignments are never rewritten.  For the current active
    season, stable FPL codes are joined to the accepted operational mapping release;
    observed source rows determine the actual appearance interval downstream.
    """
    frozen = load_training_assignments()
    registry = json.loads(
        (DATA / "season_registry.json").read_text(encoding="utf-8")
    )
    completed = set(map(str, registry["completed_seasons"]))
    requested = set(seasons)
    active_seasons = sorted(requested - completed)
    pieces = [frozen[frozen["season"].isin(requested & completed)].copy()]
    if active_seasons:
        roster = load_roster()
        active = str(roster["season"].iat[0])
        if active_seasons != [active]:
            raise RuntimeError(
                f"only checked-in current season {active} may extend frozen assignments; "
                f"got {active_seasons}"
            )
        assert_operational_clubs_ready(active)
        operational = load_operational_players().dropna(
            subset=["understat_player_id"]
        )[["fpl_code", "understat_player_id"]]
        current = roster[["fpl_code"]].merge(
            operational, on="fpl_code", how="inner", validate="one_to_one"
        ).rename(columns={"understat_player_id": "understat_id"})
        current.insert(0, "season", active)
        current["first_gameweek"] = 1
        current["last_gameweek"] = 38
        current["mapped_rows"] = 0
        current["understat_id"] = current["understat_id"].astype("int64")
        pieces.append(current[ASSIGNMENT_COLUMNS])
    out = pd.concat(pieces, ignore_index=True)
    if out.duplicated(["season", "fpl_code"]).any():
        raise RuntimeError("resolved assignments contain duplicate player-seasons")
    return out.sort_values(["season", "fpl_code"], kind="mergesort").reset_index(drop=True)


def load(path: Path = CORRECTED_MAPPING) -> pd.DataFrame:
    """Load the corrected historical mapping recommended for new joins."""
    return load_mapping(path)


def load_audit(path: Path = IDENTITY_AUDIT) -> pd.DataFrame:
    out = pd.read_csv(path, keep_default_na=False)
    if list(out.columns) != AUDIT_COLUMNS:
        raise RuntimeError(f"unexpected identity-audit columns: {list(out.columns)}")
    if out["fpl_code"].duplicated().any():
        raise RuntimeError("identity audit contains duplicate fpl_code values")
    if out[AUDIT_COLUMNS].eq("").any().any():
        raise RuntimeError("identity audit contains blank values")
    for column in ("fpl_code", "release_understat_id", "decision_understat_id"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")
    allowed = {"keep", "replace"}
    if not set(out["decision"]).issubset(allowed):
        raise RuntimeError(f"identity audit decision must be one of {sorted(allowed)}")
    inconsistent = out[
        (
            out["decision"].eq("keep")
            & out["release_understat_id"].ne(out["decision_understat_id"])
        )
        | (
            out["decision"].eq("replace")
            & out["release_understat_id"].eq(out["decision_understat_id"])
        )
    ]
    if len(inconsistent):
        raise RuntimeError("identity audit has inconsistent keep/replace decisions")
    return out


def build_corrected() -> pd.DataFrame:
    """Apply the evidence-backed audit without changing the release snapshot."""
    out = build_training()
    audit = load_audit()
    release = out.set_index("fpl_code")["understat_id"]
    missing = sorted(set(audit["fpl_code"]) - set(release.index))
    if missing:
        raise RuntimeError(
            f"identity audit references unknown release codes: {missing}"
        )
    mismatches = audit[
        audit.apply(
            lambda row: int(release.loc[int(row["fpl_code"])])
            != int(row["release_understat_id"]),
            axis=1,
        )
    ]
    if len(mismatches):
        raise RuntimeError("identity audit no longer matches the immutable release IDs")

    decisions = audit.set_index("fpl_code")["decision_understat_id"]
    mask = out["fpl_code"].isin(decisions.index)
    out.loc[mask, "understat_id"] = (
        out.loc[mask, "fpl_code"].map(decisions).astype("int64")
    )
    out["mapping_status"] = _shared_status(out)
    return (
        out[MAPPING_COLUMNS]
        .sort_values("fpl_code", kind="mergesort")
        .reset_index(drop=True)
    )


def _season_label(events: list[dict]) -> str:
    deadlines = pd.to_datetime(
        [event.get("deadline_time") for event in events if event.get("deadline_time")],
        utc=True,
    )
    if deadlines.empty:
        raise RuntimeError("FPL bootstrap has no event deadlines")
    start = int(deadlines.min().year)
    return f"{start}-{str(start + 1)[-2:]}"


def roster_from_bootstrap(payload: dict) -> pd.DataFrame:
    """Project a bootstrap response into the stable current-roster schema."""
    season = _season_label(payload.get("events", []))
    teams = {int(team["id"]): str(team["name"]) for team in payload.get("teams", [])}
    positions = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    rows = []
    for player in payload.get("elements", []):
        position = positions.get(int(player["element_type"]))
        if position is None:
            continue
        rows.append(
            {
                "season": season,
                "fpl_code": int(player["code"]),
                "element": int(player["id"]),
                "player_name": str(player["web_name"]),
                "position": position,
                "team_name": teams[int(player["team"])],
            }
        )
    out = pd.DataFrame(rows, columns=ROSTER_COLUMNS)
    if (
        out.empty
        or out["fpl_code"].duplicated().any()
        or out["element"].duplicated().any()
    ):
        raise RuntimeError(
            "FPL bootstrap did not produce a unique, non-empty player roster"
        )
    return out.sort_values("fpl_code", kind="mergesort").reset_index(drop=True)


def load_roster(path: Path = CURRENT_ROSTER) -> pd.DataFrame:
    out = pd.read_csv(path, keep_default_na=False)
    if list(out.columns) != ROSTER_COLUMNS:
        raise RuntimeError(f"unexpected current-roster columns: {list(out.columns)}")
    if out[ROSTER_COLUMNS].eq("").any().any():
        raise RuntimeError("current roster contains blank values")
    for column in ("fpl_code", "element"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")
    if out["fpl_code"].duplicated().any() or out["element"].duplicated().any():
        raise RuntimeError("current roster contains duplicate provider IDs")
    if out["season"].nunique() != 1:
        raise RuntimeError("current roster must contain exactly one season")
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_operational_players(path: Path = OPERATIONAL_PLAYERS) -> pd.DataFrame:
    """Load the generated projection of SmartPlayFPL's canonical registry."""
    out = pd.read_csv(path, keep_default_na=False)
    if list(out.columns) != OPERATIONAL_PLAYER_COLUMNS:
        raise RuntimeError(
            f"unexpected operational mapping columns: {list(out.columns)}"
        )
    out["fpl_code"] = pd.to_numeric(out["fpl_code"], errors="raise").astype("int64")
    out["understat_player_id"] = pd.to_numeric(
        out["understat_player_id"], errors="coerce"
    ).astype("Int64")
    if out["fpl_code"].duplicated().any() or (out["fpl_code"] <= 0).any():
        raise RuntimeError("operational mapping contains invalid/duplicate FPL codes")
    mapped = out["understat_player_id"].notna()
    if (out.loc[mapped, "understat_player_id"] <= 0).any():
        raise RuntimeError("operational mapping contains non-positive Understat IDs")
    if not out.loc[~mapped, "mapping_status"].eq("unmapped").all():
        raise RuntimeError("operational mapping labels missing IDs as mapped")
    if out.loc[mapped, "mapping_status"].eq("unmapped").any():
        raise RuntimeError("operational mapping labels populated IDs as unmapped")
    if not out.loc[~mapped, "confidence_level"].eq("NONE").all():
        raise RuntimeError("operational unmapped rows must have NONE confidence")
    if out.loc[mapped, "confidence_level"].eq("NONE").any():
        raise RuntimeError("operational mapped rows cannot have NONE confidence")
    return out


def load_operational_clubs(path: Path = OPERATIONAL_CLUBS) -> pd.DataFrame:
    out = pd.read_csv(path, keep_default_na=False)
    if list(out.columns) != OPERATIONAL_CLUB_COLUMNS:
        raise RuntimeError(
            f"unexpected operational club columns: {list(out.columns)}"
        )
    out["fpl_team_code"] = pd.to_numeric(
        out["fpl_team_code"], errors="raise"
    ).astype("int64")
    out["understat_team_id"] = pd.to_numeric(
        out["understat_team_id"], errors="coerce"
    ).astype("Int64")
    if out["fpl_team_code"].duplicated().any() or (out["fpl_team_code"] <= 0).any():
        raise RuntimeError("operational clubs contain invalid/duplicate FPL codes")
    mapped = out["understat_team_id"].notna()
    if (out.loc[mapped, "understat_team_id"] <= 0).any():
        raise RuntimeError("operational clubs contain non-positive Understat IDs")
    if not out.loc[mapped, "mapping_status"].eq("mapped").all() or not out.loc[
        ~mapped, "mapping_status"
    ].eq("unmapped").all():
        raise RuntimeError("operational club mapping states are inconsistent")
    return out


def verify_operational_release() -> dict:
    """Verify generated public files against their canonical release manifest."""
    meta = json.loads(OPERATIONAL_MANIFEST.read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1:
        raise RuntimeError("unsupported operational mapping manifest schema")
    release_id = str(meta.get("release_id", ""))
    canonical_sha = str(meta.get("canonical_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", canonical_sha) or not release_id.endswith(
        canonical_sha[:12]
    ):
        raise RuntimeError("operational release ID/hash is malformed")
    expected_files = {
        OPERATIONAL_PLAYERS.name: OPERATIONAL_PLAYERS,
        OPERATIONAL_CLUBS.name: OPERATIONAL_CLUBS,
    }
    if set(meta.get("files", {})) != set(expected_files):
        raise RuntimeError("operational manifest has an unexpected file set")
    for name, path in expected_files.items():
        expected = meta["files"][name]
        if path.stat().st_size != expected.get("bytes") or _sha256(path) != expected.get(
            "sha256"
        ):
            raise RuntimeError(f"{name} differs from the accepted operational release")
    players = load_operational_players()
    clubs = load_operational_clubs()
    player_coverage = meta.get("coverage", {}).get("players", {})
    club_coverage = meta.get("coverage", {}).get("clubs", {})
    if player_coverage.get("total") != len(players) or player_coverage.get(
        "mapped"
    ) != int(
        players["understat_player_id"].notna().sum()
    ):
        raise RuntimeError("operational manifest player coverage has drifted")
    if club_coverage.get("total") != len(clubs) or club_coverage.get("mapped") != int(
        clubs["understat_team_id"].notna().sum()
    ):
        raise RuntimeError("operational manifest club coverage has drifted")
    return meta


def build_current() -> pd.DataFrame:
    """Build the current roster from the versioned operational projection."""
    roster = load_roster()
    operational = load_operational_players()
    out = roster.merge(operational, on="fpl_code", how="left", validate="one_to_one")
    missing_registry = out["mapping_status"].isna()
    if missing_registry.any():
        raise RuntimeError(
            "operational registry is missing current FPL codes: "
            f"{out.loc[missing_registry, 'fpl_code'].astype(int).tolist()}"
        )
    out["understat_id"] = pd.to_numeric(
        out.pop("understat_player_id"), errors="coerce"
    ).astype(
        "Int64"
    )
    out["understat_player_name"] = ""
    out["confidence"] = out.pop("confidence_level")
    out["source"] = "smartplayfpl_mapping_release"
    return (
        out[CURRENT_COLUMNS]
        .sort_values("fpl_code", kind="mergesort")
        .reset_index(drop=True)
    )


def load_current(path: Path = CURRENT_MAPPING) -> pd.DataFrame:
    out = pd.read_csv(path, keep_default_na=False)
    if list(out.columns) != CURRENT_COLUMNS:
        raise RuntimeError(f"unexpected current-mapping columns: {list(out.columns)}")
    required = [column for column in CURRENT_COLUMNS if column != "understat_id"]
    if out[required].eq("").any().any():
        # An Understat display name is optional for historical mappings, but every
        # operational/status field must be explicit.
        bad = out[required].drop(columns=["understat_player_name"]).eq("").any().any()
        if bad:
            raise RuntimeError("current mapping contains blank required values")
    for column in ("fpl_code", "element"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")
    out["understat_id"] = pd.to_numeric(out["understat_id"], errors="coerce").astype(
        "Int64"
    )
    if out["fpl_code"].duplicated().any() or out["element"].duplicated().any():
        raise RuntimeError("current mapping contains duplicate FPL IDs")
    mapped = out["understat_id"].notna()
    duplicate = out.loc[mapped, "understat_id"].duplicated(keep=False)
    if duplicate.any() and not out.loc[mapped].loc[
        duplicate, "mapping_status"
    ].eq("shared_understat_id").all():
        raise RuntimeError("current mapping contains an unreviewed shared Understat ID")
    if not out.loc[~mapped, "mapping_status"].eq("unmapped").all():
        raise RuntimeError("current mapping labels missing Understat IDs as mapped")
    if out.loc[mapped, "mapping_status"].eq("unmapped").any():
        raise RuntimeError("current mapping labels populated Understat IDs as unmapped")
    if not out.loc[~mapped, "confidence"].eq("NONE").all():
        raise RuntimeError("current unmapped rows must have NONE confidence")
    if out.loc[mapped, "confidence"].eq("NONE").any():
        raise RuntimeError("current mapped rows cannot have NONE confidence")
    return out


def _assert_equal(published: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(published, expected, check_dtype=False)
    except AssertionError as exc:
        raise RuntimeError(
            f"{label} differs from its checked-in inputs; run --write"
        ) from exc


def verify() -> dict[str, pd.DataFrame]:
    """Verify exact training, corrected historical, and current-season maps."""
    verify_operational_release()
    training = load_training()
    assignments = load_training_assignments()
    corrected = load()
    current = load_current()
    _assert_equal(training, build_training(), "training mapping")
    _assert_equal(assignments, build_training_assignments(), "training assignments")
    _assert_equal(corrected, build_corrected(), "corrected mapping")
    _assert_equal(current, build_current(), "current mapping")

    meta = json.loads(CURRENT_META.read_text(encoding="utf-8"))
    roster = load_roster()
    if meta.get("season") != roster["season"].iat[0] or meta.get("players") != len(
        roster
    ):
        raise RuntimeError(
            "current mapping metadata does not match the roster snapshot"
        )
    if meta.get("roster_sha256") != _sha256(CURRENT_ROSTER):
        raise RuntimeError("current roster differs from its captured SHA256")
    return {
        "training": training,
        "assignments": assignments,
        "corrected": corrected,
        "current": current,
    }


def write_all() -> dict[str, pd.DataFrame]:
    MAPPINGS.mkdir(parents=True, exist_ok=True)
    frames = {
        "training": build_training(),
        "assignments": build_training_assignments(),
        "corrected": build_corrected(),
        "current": build_current(),
    }
    frames["training"].to_csv(TRAINING_MAPPING, index=False, lineterminator="\n")
    frames["assignments"].to_csv(TRAINING_ASSIGNMENTS, index=False, lineterminator="\n")
    frames["corrected"].to_csv(CORRECTED_MAPPING, index=False, lineterminator="\n")
    frames["current"].to_csv(CURRENT_MAPPING, index=False, lineterminator="\n")
    return frames


def refresh_current(bootstrap_path: Path | None = None) -> None:
    """Capture the live FPL roster, then regenerate the current identity map."""
    captured_at = datetime.now(timezone.utc).isoformat()
    if bootstrap_path is None:
        request = urllib.request.Request(
            FPL_BOOTSTRAP_URL, headers={"User-Agent": "dastan-data-rebuild/1.0"}
        )
        raw = urllib.request.urlopen(request, timeout=60).read()
    else:
        raw = bootstrap_path.read_bytes()
    payload = json.loads(raw)
    roster = roster_from_bootstrap(payload)
    roster.to_csv(CURRENT_ROSTER, index=False, lineterminator="\n")
    CURRENT_META.write_text(
        json.dumps(
            {
                "season": roster["season"].iat[0],
                "captured_at": captured_at,
                "source": FPL_BOOTSTRAP_URL,
                "bootstrap_sha256": hashlib.sha256(raw).hexdigest(),
                "roster_sha256": _sha256(CURRENT_ROSTER),
                "players": len(roster),
                "note": "Roster snapshot only; identities are joined from the versioned SmartPlayFPL operational mapping release.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_all()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate all mappings from checked-in inputs",
    )
    parser.add_argument(
        "--refresh-current",
        action="store_true",
        help="capture the live FPL roster before regenerating all mappings",
    )
    parser.add_argument(
        "--bootstrap",
        type=Path,
        help="offline bootstrap-static JSON to use with --refresh-current",
    )
    args = parser.parse_args()
    if args.bootstrap and not args.refresh_current:
        parser.error("--bootstrap requires --refresh-current")

    if args.refresh_current:
        refresh_current(args.bootstrap)
        action = "refreshed"
    elif args.write:
        write_all()
        action = "wrote"
    else:
        action = "verified"

    frames = verify()
    training = frames["training"]
    corrected = frames["corrected"]
    current = frames["current"]
    corrected_ids = int(
        (
            training.set_index("fpl_code")["understat_id"]
            != corrected.set_index("fpl_code")["understat_id"]
        ).sum()
    )
    mapped_current = int(current["understat_id"].notna().sum())
    print(
        f"{action} identity artifacts | training {len(training):,} mappings | "
        f"{corrected_ids} audited corrections | current {mapped_current:,}/{len(current):,} mapped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
