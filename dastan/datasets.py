#!/usr/bin/env python3
"""Verify, download, rebuild, and compare Dastan's published datasets.

Examples:

    python -m dastan.datasets verify
    python -m dastan.datasets download
    python -m dastan.datasets build
    python -m dastan.datasets all
    python -m dastan.datasets compare .cache/rebuilt-data

Downloads are cached under ``.cache/dastan-raw``. Rebuilds are written under
``.cache/rebuilt-data`` and never overwrite the checked-in release implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, mappings, seasons

ROOT = data.ROOT
RELEASE_MANIFEST = data.DATA / "release_manifest.json"
DEFAULT_RAW = ROOT / ".cache" / "dastan-raw"
DEFAULT_OUTPUT = ROOT / ".cache" / "rebuilt-data"

RELEASE_ARTIFACTS = [
    "season_registry.json",
    "source_pins.json",
    "features.parquet",
    "pre_deadline_ep_next.parquet",
    "pre_deadline_signals.parquet",
    "openfpl_predictions.csv",
    "openfpl_row_keys.csv",
    "players.csv",
    "mappings/fpl_understat_training_snapshot.csv",
    "mappings/fpl_understat_training_assignments.csv",
    "mappings/fpl_understat_players.csv",
    "mappings/fpl_understat_identity_audit.csv",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(path: Path) -> tuple[int | None, int | None]:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        return len(frame), len(frame.columns)
    if path.suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
        return len(frame), len(frame.columns)
    return None, None


def build_release_manifest() -> dict:
    files = {}
    for relative in RELEASE_ARTIFACTS:
        path = data.DATA / relative
        if not path.exists():
            raise RuntimeError(f"release artifact is missing: {path}")
        rows, columns = _shape(path)
        entry = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if rows is not None:
            entry.update({"rows": rows, "columns": columns})
        files[relative] = entry
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "note": (
            "Checksums describe the immutable release. Raw-source rebuilds may differ "
            "after upstream statistical corrections; retain downloads.json for an exact "
            "record of a reconstruction run. Operational/current mappings are versioned "
            "separately by data/mappings/current_manifest.json."
        ),
    }


def write_release_manifest() -> dict:
    manifest = build_release_manifest()
    RELEASE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_release() -> dict:
    if not RELEASE_MANIFEST.exists():
        raise RuntimeError("release manifest is missing; run --write-manifest")
    manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    expected_files = manifest.get("files", {})
    if set(expected_files) != set(RELEASE_ARTIFACTS):
        raise RuntimeError("release manifest artifact list is out of date")
    for relative, expected in expected_files.items():
        path = data.DATA / relative
        if not path.exists():
            raise RuntimeError(f"release artifact is missing: {relative}")
        actual_hash = _sha256(path)
        if actual_hash != expected["sha256"]:
            raise RuntimeError(
                f"release artifact changed: {relative} ({actual_hash} != {expected['sha256']})"
            )
        rows, columns = _shape(path)
        if rows is not None and (rows, columns) != (
            expected.get("rows"),
            expected.get("columns"),
        ):
            raise RuntimeError(
                f"release shape changed for {relative}: {(rows, columns)}"
            )

    frame = pd.read_parquet(data.FRAME)
    if len(frame) != data.EXPECTED_ROWS:
        raise RuntimeError(
            f"expected {data.EXPECTED_ROWS:,} feature rows, got {len(frame):,}"
        )
    key = ["season", "gameweek", "fixture", "fpl_code"]
    if frame.duplicated(key).any():
        raise RuntimeError(
            "released feature frame contains duplicate player-fixture keys"
        )
    data.assert_deadline_anchored(frame)

    for artifact in (data.EP_NEXT, data.SIGNALS):
        snapshot = pd.read_parquet(artifact)
        if snapshot.duplicated(["season", "gameweek", "fpl_code"]).any():
            raise RuntimeError(f"{artifact.name} contains duplicate gameweek keys")
        if snapshot["snapshot_at"].ge(snapshot["deadline_time"]).any():
            raise RuntimeError(f"{artifact.name} contains post-deadline rows")
    mappings.verify()
    return manifest


def rebuild(
    raw_dir: Path,
    output_dir: Path,
    seasons: list[str],
    *,
    skip_predeadline: bool,
) -> dict:
    from .rebuild.features import build_feature_frame
    from .rebuild.fplcache import build_predeadline_artifacts
    from .rebuild.sources import build_canonical_matches, write_download_manifest

    if not (raw_dir / "downloads.json").exists():
        raise RuntimeError(
            f"raw source manifest is missing under {raw_dir}; run download first"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print("  loading and joining raw player/team histories", flush=True)
    players, teams, lookup = build_canonical_matches(raw_dir, seasons)
    print(
        f"  canonical rows: {len(players):,}; team matches: {len(teams):,}", flush=True
    )
    feature_frame = build_feature_frame(players, teams)
    feature_path = output_dir / "features.parquet"
    feature_frame.to_parquet(feature_path, index=False)
    lookup.to_csv(output_dir / "players.csv", index=False, lineterminator="\n")

    if not skip_predeadline:
        build_predeadline_artifacts(raw_dir, output_dir, seasons)
        # Include selected fplcache snapshots in the immutable per-run source record.
        write_download_manifest(raw_dir, seasons)

    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "rebuild_manifest.json":
            continue
        rows, columns = _shape(path)
        entry = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if rows is not None:
            entry.update({"rows": rows, "columns": columns})
        outputs[path.name] = entry
    source_manifest = raw_dir / "downloads.json"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": _sha256(source_manifest),
        "outputs": outputs,
    }
    (output_dir / "rebuild_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  rebuilt feature frame -> {feature_path}", flush=True)
    return manifest


def compare(candidate_dir: Path, strict: bool = False) -> dict:
    candidate_path = candidate_dir / "features.parquet"
    if not candidate_path.exists():
        raise RuntimeError(f"candidate feature frame is missing: {candidate_path}")
    released = pd.read_parquet(data.FRAME)
    candidate = pd.read_parquet(candidate_path)
    key = ["season", "gameweek", "fixture", "fpl_code"]
    if candidate.duplicated(key).any():
        raise RuntimeError("candidate feature frame contains duplicate keys")
    missing_columns = sorted(set(released.columns) - set(candidate.columns))
    extra_columns = sorted(set(candidate.columns) - set(released.columns))
    common = released[key].merge(candidate[key], on=key, how="inner")

    joined = released.merge(
        candidate,
        on=key,
        how="inner",
        suffixes=("_release", "_candidate"),
        validate="one_to_one",
    )
    feature_columns = json.loads(data.FEATURE_COLS.read_text(encoding="utf-8"))
    numeric_differences = []
    for column in feature_columns:
        left = f"{column}_release"
        right = f"{column}_candidate"
        if left not in joined or right not in joined:
            continue
        a = pd.to_numeric(joined[left], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(joined[right], errors="coerce").to_numpy(dtype=float)
        both_missing = np.isnan(a) & np.isnan(b)
        difference = np.abs(a - b)
        difference[both_missing] = 0.0
        exact_different = ~((a == b) | both_missing)
        different = ~np.isclose(a, b, rtol=1e-6, atol=1e-7, equal_nan=True)
        finite = difference[np.isfinite(difference)]
        numeric_differences.append(
            {
                "column": column,
                "exact_different_rows": int(exact_different.sum()),
                "different_rows": int(different.sum()),
                "mean_abs_difference": float(finite.mean()) if len(finite) else 0.0,
                "max_abs_difference": float(finite.max()) if len(finite) else 0.0,
            }
        )
    numeric_differences.sort(
        key=lambda item: (item["different_rows"], item["max_abs_difference"]),
        reverse=True,
    )

    reconstructed_artifacts = {}
    for filename, artifact_key in (
        ("pre_deadline_ep_next.parquet", ["season", "gameweek", "fpl_code"]),
        ("pre_deadline_signals.parquet", ["season", "gameweek", "fpl_code"]),
    ):
        candidate_artifact = candidate_dir / filename
        if not candidate_artifact.exists():
            continue
        expected = (
            pd.read_parquet(data.DATA / filename)
            .sort_values(artifact_key, kind="mergesort")
            .reset_index(drop=True)
        )
        rebuilt = (
            pd.read_parquet(candidate_artifact)
            .sort_values(artifact_key, kind="mergesort")
            .reset_index(drop=True)
        )
        semantically_equal = False
        try:
            pd.testing.assert_frame_equal(expected, rebuilt, check_dtype=False)
            semantically_equal = True
        except AssertionError:
            pass
        reconstructed_artifacts[filename] = {
            "release_rows": len(expected),
            "candidate_rows": len(rebuilt),
            "semantic_equal": semantically_equal,
            "byte_equal": _sha256(data.DATA / filename) == _sha256(candidate_artifact),
        }
    report = {
        "release_rows": len(released),
        "candidate_rows": len(candidate),
        "common_keys": len(common),
        "release_only_keys": len(released) - len(common),
        "candidate_only_keys": len(candidate) - len(common),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "numeric_tolerance": {"rtol": 1e-6, "atol": 1e-7},
        "feature_columns_with_exact_differences": sum(
            item["exact_different_rows"] > 0 for item in numeric_differences
        ),
        "feature_columns_with_differences": sum(
            item["different_rows"] > 0 for item in numeric_differences
        ),
        "largest_differences": numeric_differences[:20],
        "reconstructed_artifacts": reconstructed_artifacts,
        "candidate_sha256": _sha256(candidate_path),
        "release_sha256": _sha256(data.FRAME),
    }
    print(json.dumps(report, indent=2))
    exact = report["candidate_sha256"] == report["release_sha256"]
    if strict and not exact:
        raise RuntimeError(
            "candidate is not byte-identical to the released feature frame"
        )
    return report


def _seasons(value: str) -> list[str]:
    requested = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(requested) - set(seasons.source_seasons()))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"seasons are not registered/current: {unknown}; update the roster or registry first"
        )
    return requested


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    verify_parser = subparsers.add_parser(
        "verify", help="verify release hashes and invariants"
    )
    verify_parser.add_argument(
        "--write-manifest", action="store_true", help="replace release_manifest.json"
    )

    for name in ("download", "build", "all"):
        command = subparsers.add_parser(name)
        command.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
        command.add_argument("--seasons", type=_seasons, default=list(data.SEASONS))
        if name in {"download", "all"}:
            command.add_argument("--workers", type=int, default=12)
            command.add_argument("--force", action="store_true")
            command.add_argument("--allow-missing-understat", action="store_true")
        if name in {"build", "all"}:
            command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
            command.add_argument("--skip-predeadline", action="store_true")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("candidate_dir", type=Path)
    compare_parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    command = args.command or "verify"

    if command == "verify":
        if getattr(args, "write_manifest", False):
            write_release_manifest()
        manifest = verify_release()
        total_bytes = sum(entry["bytes"] for entry in manifest["files"].values())
        print(
            f"verified {len(manifest['files'])} release artifacts | "
            f"{total_bytes / 1024**2:.1f} MiB | deadline and identity invariants pass"
        )
        return 0
    if command == "compare":
        compare(args.candidate_dir, strict=args.strict)
        return 0

    if command in {"download", "all"}:
        from .rebuild.sources import download_sources

        download_sources(
            args.raw_dir,
            args.seasons,
            workers=args.workers,
            force=args.force,
            allow_missing_understat=args.allow_missing_understat,
        )
    if command in {"build", "all"}:
        rebuild(
            args.raw_dir,
            args.output_dir,
            args.seasons,
            skip_predeadline=args.skip_predeadline,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
