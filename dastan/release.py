"""Prepare, accept, and promote public Dastan retraining releases safely.

Data preparation and approval are intentionally separate commands.  A candidate
manifest binds every input by SHA256; training refuses a candidate without the
matching accepted manifest; promotion is preview-only unless ``--apply`` is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import artifacts, data, mappings, seasons

CANDIDATE_NAME = "dastan_release_candidate.json"
ACCEPTED_NAME = "dastan_release_accepted.json"
SNAPSHOT_FILES = [
    "pre_deadline_ep_next.parquet",
    "pre_deadline_signals.parquet",
]
DATA_FILES = ["features.parquet", *SNAPSHOT_FILES, "players.csv"]
INPUT_FILES = {
    "source_pins": data.DATA / "source_pins.json",
    "season_registry": data.DATA / "season_registry.json",
    "mapping_release": data.DATA / "mappings" / "current_manifest.json",
    "feature_contract": artifacts.MODELS / "feature_cols.json",
    "hyperparameters": artifacts.MODELS / "hyperparameters.json",
    "tuning_decision": artifacts.MODELS / "tuned_params.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def inspect_candidate(data_dir: Path, plan: seasons.RetrainingPlan) -> dict:
    data_dir = data_dir.resolve()
    missing = [name for name in DATA_FILES if not (data_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"candidate data is missing: {missing}")
    mappings.verify()
    registered = set(seasons.load_registry()["completed_seasons"])
    if plan.data_season not in registered:
        mappings.assert_operational_clubs_ready(plan.data_season)

    frame_path = data_dir / "features.parquet"
    frame = pd.read_parquet(
        frame_path,
        columns=["season", "gameweek", "fixture", "fpl_code", "position"],
    )
    key = ["season", "gameweek", "fixture", "fpl_code"]
    if frame.empty or frame.duplicated(key).any():
        raise RuntimeError("candidate frame is empty or has duplicate player-fixture keys")
    actual_seasons = set(frame["season"].astype(str))
    if actual_seasons != set(plan.seasons):
        raise RuntimeError(
            f"candidate seasons differ from plan: expected={list(plan.seasons)}, "
            f"actual={sorted(actual_seasons)}"
        )
    active = frame[frame["season"].eq(plan.data_season)]
    gameweeks = sorted(active["gameweek"].dropna().astype(int).unique())
    if not gameweeks or gameweeks[0] != 1 or gameweeks[-1] != plan.available_through_gw:
        raise RuntimeError(
            f"{plan.data_season} must span GW1-GW{plan.available_through_gw}; "
            f"got first={gameweeks[:1]}, last={gameweeks[-1:]}"
        )
    minimum_gameweeks = plan.available_through_gw if plan.mode == "midseason" else 37
    if len(gameweeks) < minimum_gameweeks:
        raise RuntimeError(
            f"{plan.data_season} has only {len(gameweeks)} distinct gameweeks"
        )
    holdout = active[active["gameweek"].between(*plan.holdout)]
    positions = set(holdout["position"].dropna().astype(str))
    if positions != {"GKP", "DEF", "MID", "FWD"}:
        raise RuntimeError(f"candidate holdout positions are incomplete: {positions}")

    snapshot_coverage = {}
    for name in SNAPSHOT_FILES:
        snapshot = pd.read_parquet(data_dir / name)
        snapshot_key = ["season", "gameweek", "fpl_code"]
        if snapshot.duplicated(snapshot_key).any():
            raise RuntimeError(f"{name} has duplicate player-gameweek keys")
        selected = snapshot[snapshot["season"].eq(plan.data_season)]
        selected_gameweeks = set(selected["gameweek"].dropna().astype(int))
        expected_gameweeks = set(range(1, plan.available_through_gw + 1))
        if selected_gameweeks != expected_gameweeks:
            missing_gameweeks = sorted(expected_gameweeks - selected_gameweeks)
            raise RuntimeError(
                f"{name} is missing {plan.data_season} gameweeks: {missing_gameweeks}"
            )
        max_gw = int(selected["gameweek"].max()) if not selected.empty else 0
        if max_gw != plan.available_through_gw:
            raise RuntimeError(
                f"{name} ends at GW{max_gw}, expected GW{plan.available_through_gw}"
            )
        if {"snapshot_at", "deadline_time"} <= set(snapshot.columns):
            captured = pd.to_datetime(snapshot["snapshot_at"], utc=True, errors="coerce")
            deadline = pd.to_datetime(snapshot["deadline_time"], utc=True, errors="coerce")
            if captured.isna().any() or deadline.isna().any() or captured.ge(deadline).any():
                raise RuntimeError(f"{name} contains invalid or post-deadline snapshots")
        snapshot_coverage[name] = {
            "rows": len(selected),
            "gameweeks": len(selected_gameweeks),
            "max_gameweek": max_gw,
        }

    players = pd.read_csv(data_dir / "players.csv")
    if players.duplicated(["season", "fpl_code"]).any():
        raise RuntimeError("players.csv has duplicate season/FPL-code identities")
    if set(players["season"].astype(str)) != set(plan.seasons):
        raise RuntimeError("players.csv seasons differ from the candidate plan")

    coverage = {}
    for season, group in frame.groupby("season", sort=True):
        coverage[str(season)] = {
            "rows": len(group),
            "gameweeks": int(group["gameweek"].nunique()),
            "min_gameweek": int(group["gameweek"].min()),
            "max_gameweek": int(group["gameweek"].max()),
            "fixtures": int(group["fixture"].nunique()),
        }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate",
        "plan": plan.to_dict(),
        "rows": len(frame),
        "columns": len(pd.read_parquet(frame_path).columns),
        "coverage": coverage,
        "holdout_rows_by_position": {
            str(key): int(value)
            for key, value in holdout["position"].value_counts().sort_index().items()
        },
        "snapshot_coverage": snapshot_coverage,
        "data_files": {
            name: _fingerprint(data_dir / name) for name in DATA_FILES
        },
        "release_inputs": {
            name: _fingerprint(path) for name, path in INPUT_FILES.items()
        },
    }


def _without_times(payload: dict) -> dict:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "accepted_at", "status"}
    }


def accept_candidate(
    manifest_path: Path, *, apply: bool = False
) -> Path:
    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan_data = candidate.get("plan", {})
    plan = seasons.make_plan(
        str(plan_data.get("mode")),
        season=str(plan_data.get("data_season")),
        through_gw=int(plan_data.get("available_through_gw")),
    )
    current = inspect_candidate(manifest_path.parent, plan)
    if _without_times(candidate) != _without_times(current):
        raise RuntimeError("candidate data or release inputs changed after preparation")
    accepted_path = manifest_path.parent / ACCEPTED_NAME
    if not apply:
        print(
            f"validated candidate {plan.run_id}: {candidate['rows']:,} rows; "
            "preview only, rerun accept with --apply"
        )
        return accepted_path
    accepted = dict(candidate)
    accepted["status"] = "accepted"
    accepted["accepted_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(accepted_path, accepted)
    print(f"accepted exact candidate -> {accepted_path}")
    return accepted_path


def verify_accepted_manifest(
    manifest_path: Path, data_dir: Path, plan: seasons.RetrainingPlan
) -> dict:
    accepted = json.loads(manifest_path.read_text(encoding="utf-8"))
    if accepted.get("status") != "accepted" or not accepted.get("accepted_at"):
        raise RuntimeError("training manifest has not been explicitly accepted")
    current = inspect_candidate(data_dir, plan)
    if accepted.get("plan") != plan.to_dict():
        raise RuntimeError("accepted manifest belongs to a different retraining plan")
    if _without_times(accepted) != _without_times(current):
        raise RuntimeError("accepted data or release inputs changed before training")
    return accepted


def register_release(
    manifest_path: Path, data_dir: Path, *, apply: bool = False
) -> dict:
    """Advance release/completed season state from one accepted data manifest."""
    accepted = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan_data = accepted.get("plan", {})
    plan = seasons.make_plan(
        str(plan_data.get("mode")),
        season=str(plan_data.get("data_season")),
        through_gw=int(plan_data.get("available_through_gw")),
    )
    verify_accepted_manifest(manifest_path, data_dir, plan)
    current = seasons.load_registry()
    completed = list(current["completed_seasons"])
    latest = str(current["latest_completed_season"])
    if plan.mode == "annual" and plan.data_season != latest:
        if plan.data_season != seasons.next_season(latest):
            raise RuntimeError("annual registration is not the immediate successor")
        completed = list(plan.seasons)
        latest = plan.data_season
    published_plan = plan.to_dict()
    published_plan["expected_rows"] = int(accepted["rows"])
    registry = {
        "schema_version": 1,
        "completed_seasons": completed,
        "release_seasons": list(plan.seasons),
        "latest_completed_season": latest,
        "release_rows": int(accepted["rows"]),
        "published_retraining_plan": published_plan,
        "season_coverage": accepted["coverage"],
        "release_data_contract": {
            name: value["sha256"]
            for name, value in accepted["data_files"].items()
        },
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    action = "complete" if plan.mode == "annual" else "partial"
    print(
        f"validated {action} {plan.data_season} registry: {accepted['rows']:,} rows, "
        f"{len(plan.seasons)} release seasons"
    )
    if not apply:
        print("preview only; rerun register with --apply in the data-release commit")
        return registry
    _atomic_json(seasons.REGISTRY_PATH, registry)
    print(f"registered published plan -> {seasons.REGISTRY_PATH}")
    return registry


def promote(source: Path, production_commit: str, *, apply: bool = False) -> None:
    source = source.resolve()
    summary = artifacts.verify_directory(source)
    changes = []
    for name in artifacts.RELEASE_FILES:
        target = artifacts.MODELS / name
        if not target.exists() or _sha256(source / name) != _sha256(target):
            changes.append(name)
    print(
        f"validated {summary['files']}-file candidate for {summary['target_season']}; "
        f"{len(changes)} files differ from the public release"
    )
    if not apply:
        print("preview only; rerun promote with --apply after reviewing evaluation evidence")
        return

    stage = Path(tempfile.mkdtemp(prefix=".models-stage-", dir=artifacts.ROOT))
    backup = artifacts.ROOT / ".models-backup"
    if backup.exists():
        raise RuntimeError(f"stale promotion backup exists: {backup}")
    try:
        for name in artifacts.RELEASE_FILES:
            shutil.copy2(source / name, stage / name)
        manifest = artifacts.build_manifest(stage, production_commit=production_commit)
        (stage / artifacts.MANIFEST.name).write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(artifacts.MODELS, backup)
        try:
            os.replace(stage, artifacts.MODELS)
        except Exception:
            os.replace(backup, artifacts.MODELS)
            raise
        shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    artifacts.verify_manifest()
    print(f"promoted public Dastan contract from production commit {production_commit}")


def _plan_from_args(args: argparse.Namespace) -> seasons.RetrainingPlan:
    return seasons.make_plan(
        args.mode, season=args.season, through_gw=args.through_gw
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare"):
        command = sub.add_parser(name)
        command.add_argument("--mode", choices=("annual", "midseason"), default="annual")
        command.add_argument("--season")
        command.add_argument("--through-gw", type=int)
        if name == "prepare":
            command.add_argument("--candidate-dir", type=Path, required=True)
    accept = sub.add_parser("accept")
    accept.add_argument("--manifest", type=Path, required=True)
    accept.add_argument("--apply", action="store_true")
    check = sub.add_parser("check")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--data-dir", type=Path, required=True)
    register = sub.add_parser("register")
    register.add_argument("--manifest", type=Path, required=True)
    register.add_argument("--data-dir", type=Path, required=True)
    register.add_argument("--apply", action="store_true")
    promote_parser = sub.add_parser("promote")
    promote_parser.add_argument("--source", type=Path, required=True)
    promote_parser.add_argument("--production-commit", required=True)
    promote_parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.command == "plan":
        print(json.dumps(_plan_from_args(args).to_dict(), indent=2))
        return 0
    if args.command == "prepare":
        plan = _plan_from_args(args)
        payload = inspect_candidate(args.candidate_dir, plan)
        path = args.candidate_dir.resolve() / CANDIDATE_NAME
        _atomic_json(path, payload)
        print(
            f"prepared {plan.run_id}: {payload['rows']:,} rows -> {path}\n"
            "Review coverage and hashes, then run the separate accept command."
        )
        return 0
    if args.command == "accept":
        accept_candidate(args.manifest, apply=args.apply)
        return 0
    if args.command == "check":
        accepted = json.loads(args.manifest.read_text(encoding="utf-8"))
        plan_data = accepted.get("plan", {})
        plan = seasons.make_plan(
            str(plan_data.get("mode")),
            season=str(plan_data.get("data_season")),
            through_gw=int(plan_data.get("available_through_gw")),
        )
        verify_accepted_manifest(args.manifest, args.data_dir, plan)
        print(f"accepted data contract is intact for {plan.run_id}")
        return 0
    if args.command == "register":
        register_release(args.manifest, args.data_dir, apply=args.apply)
        return 0
    promote(
        args.source,
        args.production_commit,
        apply=args.apply,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
