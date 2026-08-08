"""Versioned annual and mid-season retraining plans for the public release.

Season state lives in ``data/season_registry.json`` so adding a completed season
does not require editing Python constants.  An active season may be prepared before
registration only when it is the immediate successor and matches the checked-in
current FPL roster.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REGISTRY_PATH = DATA / "season_registry.json"
CURRENT_ROSTER = DATA / "mappings" / "fpl_players_current.csv"
SEASON_RE = re.compile(r"^(\d{4})-(\d{2})$")

GAMEWEEKS = 38
ANNUAL_HOLDOUT = (31, 38)
MIDSEASON_DEFAULT_THROUGH_GW = 19
MIDSEASON_HOLDOUT_GAMEWEEKS = 4
MIDSEASON_MIN_THROUGH_GW = 12


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported Dastan season-registry schema")
    completed = payload.get("completed_seasons")
    released = payload.get("release_seasons", completed)
    latest = payload.get("latest_completed_season")
    if not isinstance(completed, list) or not completed or completed[-1] != latest:
        raise RuntimeError("season registry has an inconsistent latest season")
    if completed != sorted(set(completed)):
        raise RuntimeError("completed seasons must be unique and chronological")
    if not isinstance(released, list) or released != sorted(set(released)):
        raise RuntimeError("release seasons must be unique and chronological")
    if not set(completed) <= set(released):
        raise RuntimeError("every completed season must be present in the public release")
    for season in completed:
        validate_label(str(season))
    if int(payload.get("release_rows", 0)) <= 0:
        raise RuntimeError("season registry has no positive release row guard")
    if set(payload.get("season_coverage", {})) != set(released):
        raise RuntimeError("season registry coverage does not match release seasons")
    return payload


def validate_label(season: str) -> None:
    match = SEASON_RE.fullmatch(season)
    if not match:
        raise ValueError(f"season must use YYYY-YY, got {season!r}")
    start = int(match.group(1))
    if int(match.group(2)) != (start + 1) % 100:
        raise ValueError(f"season years are not consecutive: {season!r}")


def next_season(season: str) -> str:
    validate_label(season)
    start = int(season[:4]) + 1
    return f"{start}-{str(start + 1)[-2:]}"


def current_roster_season(path: Path = CURRENT_ROSTER) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        seasons = {str(row["season"]) for row in csv.DictReader(handle)}
    if len(seasons) != 1:
        raise RuntimeError("current FPL roster must contain exactly one season")
    season = seasons.pop()
    validate_label(season)
    return season


@dataclass(frozen=True)
class RetrainingPlan:
    mode: str
    data_season: str
    seasons: tuple[str, ...]
    available_through_gw: int
    train_max_gw: int
    holdout: tuple[int, int]
    target_season: str
    expected_rows: int | None

    @property
    def run_id(self) -> str:
        suffix = f"-gw{self.available_through_gw}" if self.mode == "midseason" else ""
        return f"{self.mode}-{self.data_season}{suffix}"

    def to_dict(self) -> dict:
        value = asdict(self)
        value["seasons"] = list(self.seasons)
        value["holdout"] = list(self.holdout)
        value["run_id"] = self.run_id
        return value


def published_plan() -> RetrainingPlan:
    """The exact plan that produced the checked-in public frame and model."""
    payload = load_registry().get("published_retraining_plan")
    if not isinstance(payload, dict):
        raise RuntimeError("season registry has no published retraining plan")
    plan = RetrainingPlan(
        mode=str(payload["mode"]),
        data_season=str(payload["data_season"]),
        seasons=tuple(map(str, payload["seasons"])),
        available_through_gw=int(payload["available_through_gw"]),
        train_max_gw=int(payload["train_max_gw"]),
        holdout=tuple(map(int, payload["holdout"])),
        target_season=str(payload["target_season"]),
        expected_rows=(
            int(payload["expected_rows"])
            if payload.get("expected_rows") is not None
            else None
        ),
    )
    if payload.get("run_id") != plan.run_id:
        raise RuntimeError("published retraining plan has an inconsistent run ID")
    if tuple(load_registry()["release_seasons"]) != plan.seasons:
        raise RuntimeError("published plan seasons differ from the released frame")
    return plan


def make_plan(
    mode: str = "annual",
    *,
    season: str | None = None,
    through_gw: int | None = None,
) -> RetrainingPlan:
    registry = load_registry()
    completed = tuple(str(value) for value in registry["completed_seasons"])
    latest = str(registry["latest_completed_season"])
    successor = next_season(latest)
    if mode == "annual":
        data_season = season or latest
        if data_season not in {latest, successor}:
            raise ValueError(
                f"annual season must be registered {latest} or its successor {successor}"
            )
        if through_gw not in (None, GAMEWEEKS):
            raise ValueError("annual retraining requires a completed GW38 season")
        registered = data_season == latest
        seasons = completed if registered else (*completed, data_season)
        plan = RetrainingPlan(
            mode=mode,
            data_season=data_season,
            seasons=seasons,
            available_through_gw=GAMEWEEKS,
            train_max_gw=ANNUAL_HOLDOUT[0] - 1,
            holdout=ANNUAL_HOLDOUT,
            target_season=next_season(data_season),
            expected_rows=int(registry["release_rows"]) if registered else None,
        )
    elif mode == "midseason":
        data_season = season or current_roster_season()
        if data_season != successor:
            raise ValueError(
                f"mid-season data must be the immediate successor {successor}, got {data_season}"
            )
        cutoff = through_gw or MIDSEASON_DEFAULT_THROUGH_GW
        if not MIDSEASON_MIN_THROUGH_GW <= cutoff < GAMEWEEKS:
            raise ValueError(
                f"mid-season cutoff must be GW{MIDSEASON_MIN_THROUGH_GW}-GW{GAMEWEEKS - 1}"
            )
        holdout = (cutoff - MIDSEASON_HOLDOUT_GAMEWEEKS + 1, cutoff)
        plan = RetrainingPlan(
            mode=mode,
            data_season=data_season,
            seasons=(*completed, data_season),
            available_through_gw=cutoff,
            train_max_gw=holdout[0] - 1,
            holdout=holdout,
            target_season=data_season,
            expected_rows=None,
        )
    else:
        raise ValueError(f"mode must be annual or midseason, got {mode!r}")
    return plan


def source_seasons() -> tuple[str, ...]:
    """Completed release seasons plus the checked-in immediate active season."""
    registry = load_registry()
    released = tuple(str(value) for value in registry["release_seasons"])
    active = current_roster_season()
    latest = str(registry["latest_completed_season"])
    if active not in released and active == next_season(latest):
        return (*released, active)
    return released
