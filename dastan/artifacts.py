"""Validate and fingerprint the complete public Dastan model contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .model import POSITIONS

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MANIFEST = MODELS / "artifact_manifest.json"


def head_files() -> list[str]:
    names: list[str] = []
    for pos in POSITIONS:
        names.extend(
            [
                f"p60_{pos}.json",
                f"non60_{pos}.json",
                f"bucket_{pos}.json",
                *(f"bucketreg_{pos}_{bucket}.json" for bucket in range(4)),
                f"direct_{pos}.json",
            ]
        )
    return names


RUNTIME_FILES = [
    *head_files(),
    "feature_cols.json",
    "bucket_calibration.json",
    "minutes_calibration.json",
    "blend.json",
    "train_metadata.json",
    "tuned_params.json",
]
PUBLIC_TRAINING_FILES = ["core_feature_cols.json", "hyperparameters.json"]
RELEASE_FILES = sorted([*RUNTIME_FILES, *PUBLIC_TRAINING_FILES])
PRODUCTION_REPOSITORY = "qazybekb/smartplayfplv2"
PRODUCTION_ARTIFACT_DIRECTORY = "python/ml/model/artifacts/dastan_production"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_production_commit(value: object) -> str:
    commit = str(value)
    if not COMMIT_RE.fullmatch(commit):
        raise RuntimeError("production commit must be a full lowercase 40-character SHA")
    return commit


def verify_directory(model_dir: Path = MODELS) -> dict:
    missing = [name for name in RELEASE_FILES if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Dastan artifact contract is missing: {missing}")
    features = json.loads((model_dir / "feature_cols.json").read_text(encoding="utf-8"))
    if len(features) != 286 or len(features) != len(set(features)):
        raise RuntimeError("feature_cols.json must contain 286 unique ordered features")
    blend = json.loads((model_dir / "blend.json").read_text(encoding="utf-8"))
    weights = blend.get("per_position_direct_weight", {})
    if set(weights) != set(POSITIONS) or any(
        not 0.0 <= float(value) <= 1.0 for value in weights.values()
    ):
        raise RuntimeError("blend.json has invalid per-position weights")
    minutes = json.loads(
        (model_dir / "minutes_calibration.json").read_text(encoding="utf-8")
    )
    if set(minutes.get("curve", {})) != set(POSITIONS):
        raise RuntimeError("minutes calibration does not cover all positions")
    for pos, curve in minutes["curve"].items():
        p60 = list(map(float, curve.get("p60", [])))
        expected = list(map(float, curve.get("e_minutes", [])))
        p_any = list(map(float, curve.get("p_any", [])))
        if not p60 or not (len(p60) == len(expected) == len(p_any)):
            raise RuntimeError(f"{pos} minutes calibration has inconsistent knots")
        if p60 != sorted(p60) or expected != sorted(expected) or p_any != sorted(p_any):
            raise RuntimeError(f"{pos} minutes calibration is not monotone")
        if any(m < 60.0 * p - 1e-3 for p, m in zip(p60, expected)):
            raise RuntimeError(f"{pos} expected minutes contradict p60")
        if any(a + 1e-3 < max(p, m / 90.0) for p, m, a in zip(p60, expected, p_any)):
            raise RuntimeError(f"{pos} p_any contradicts p60/expected minutes")
    metadata = json.loads((model_dir / "train_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_version") != "dastan":
        raise RuntimeError("train metadata must declare model_version=dastan")
    if metadata.get("scenario", {}).get("base_model") != "dastan_production":
        raise RuntimeError("train metadata is not bound to the promoted base model")
    return {
        "files": len(RELEASE_FILES),
        "runtime_files": len(RUNTIME_FILES),
        "features": len(features),
        "target_season": metadata.get("target_season"),
    }


def build_manifest(
    model_dir: Path = MODELS, *, production_commit: str | None = None
) -> dict:
    summary = verify_directory(model_dir)
    if production_commit is None:
        if MANIFEST.exists():
            production_commit = str(
                json.loads(MANIFEST.read_text(encoding="utf-8"))
                .get("production_source", {})
                .get("commit", "")
            )
        if not production_commit:
            raise RuntimeError("a production commit is required for the first manifest")
    production_commit = validate_production_commit(production_commit)
    mapping = json.loads(
        (ROOT / "data" / "mappings" / "current_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "dastan",
        "public_version": "1.1.0",
        "production_source": {
            "repository": PRODUCTION_REPOSITORY,
            "commit": production_commit,
            "artifact_directory": PRODUCTION_ARTIFACT_DIRECTORY,
            "equivalence": (
                "all 38 runtime files are byte-for-byte copies of the production "
                "artifact contract at this commit"
            ),
        },
        "mapping_release": {
            "release_id": mapping["release_id"],
            "canonical_sha256": mapping["canonical_sha256"],
        },
        "contract": summary,
        "files": {
            name: {
                "bytes": (model_dir / name).stat().st_size,
                "sha256": sha256(model_dir / name),
                "role": "runtime" if name in RUNTIME_FILES else "public_retraining",
            }
            for name in RELEASE_FILES
        },
    }


def verify_manifest(model_dir: Path = MODELS, manifest_path: Path = MANIFEST) -> dict:
    summary = verify_directory(model_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported model artifact manifest schema")
    if manifest.get("model_version") != "dastan" or manifest.get(
        "public_version"
    ) != "1.1.0":
        raise RuntimeError("artifact manifest has an unexpected release version")
    production = manifest.get("production_source", {})
    if production.get("repository") != PRODUCTION_REPOSITORY or production.get(
        "artifact_directory"
    ) != PRODUCTION_ARTIFACT_DIRECTORY:
        raise RuntimeError("artifact manifest points at an unexpected production source")
    validate_production_commit(production.get("commit"))
    if manifest.get("contract") != summary:
        raise RuntimeError("artifact manifest summary differs from the model contract")
    if set(manifest.get("files", {})) != set(RELEASE_FILES):
        raise RuntimeError("artifact manifest file set is out of date")
    for name, expected in manifest["files"].items():
        expected_role = "runtime" if name in RUNTIME_FILES else "public_retraining"
        if expected.get("role") != expected_role:
            raise RuntimeError(f"artifact manifest has the wrong role for {name}")
        path = model_dir / name
        if path.stat().st_size != expected.get("bytes") or sha256(path) != expected.get(
            "sha256"
        ):
            raise RuntimeError(f"released model artifact changed: {name}")
    mapping = json.loads(
        (ROOT / "data" / "mappings" / "current_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.get("mapping_release") != {
        "release_id": mapping.get("release_id"),
        "canonical_sha256": mapping.get("canonical_sha256"),
    }:
        raise RuntimeError("model manifest points at a stale operational mapping release")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--production-commit")
    args = parser.parse_args()
    if args.write_manifest:
        MANIFEST.write_text(
            json.dumps(
                build_manifest(production_commit=args.production_commit),
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {MANIFEST}")
    summary = verify_manifest()
    print(
        f"verified Dastan {summary['files']}-file public contract | "
        f"{summary['runtime_files']} production files | target {summary['target_season']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
