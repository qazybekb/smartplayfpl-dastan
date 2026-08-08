from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from dastan import artifacts, mappings, seasons
from dastan.minutes import apply_curve, payload_equivalent


class SeasonPlanTests(unittest.TestCase):
    def test_registered_annual_plan_targets_following_season(self) -> None:
        plan = seasons.published_plan()
        self.assertEqual(plan.data_season, "2025-26")
        self.assertEqual(plan.holdout, (31, 38))
        self.assertEqual(plan.target_season, "2026-27")
        self.assertEqual(plan.expected_rows, 163_072)

    def test_midseason_plan_is_derived_without_python_edits(self) -> None:
        plan = seasons.make_plan("midseason", season="2026-27", through_gw=19)
        self.assertEqual(plan.train_max_gw, 15)
        self.assertEqual(plan.holdout, (16, 19))
        self.assertEqual(plan.target_season, "2026-27")
        self.assertEqual(plan.seasons[-1], "2026-27")

    def test_midseason_rejects_non_successor(self) -> None:
        with self.assertRaisesRegex(ValueError, "immediate successor"):
            seasons.make_plan("midseason", season="2027-28", through_gw=19)


class ArtifactContractTests(unittest.TestCase):
    def test_manifest_binds_complete_public_and_runtime_contracts(self) -> None:
        summary = artifacts.verify_manifest()
        self.assertEqual(summary["files"], 40)
        self.assertEqual(summary["runtime_files"], 38)
        self.assertEqual(summary["features"], 286)
        self.assertEqual(summary["target_season"], "2026-27")

    def test_production_commit_requires_full_sha(self) -> None:
        self.assertEqual(artifacts.validate_production_commit("a" * 40), "a" * 40)
        with self.assertRaisesRegex(RuntimeError, "40-character SHA"):
            artifacts.validate_production_commit("a" * 12)

    def test_minutes_curve_enforces_probability_invariants(self) -> None:
        payload = json.loads(
            (artifacts.MODELS / "minutes_calibration.json").read_text(encoding="utf-8")
        )
        probabilities = np.linspace(0.0, 1.0, 501)
        for position, curve in payload["curve"].items():
            minutes, p_any = apply_curve(probabilities, curve)
            self.assertTrue(np.isfinite(minutes).all(), position)
            self.assertTrue(np.isfinite(p_any).all(), position)
            self.assertTrue((minutes >= 60.0 * probabilities - 1e-9).all(), position)
            self.assertTrue((p_any >= probabilities - 1e-9).all(), position)
            self.assertTrue((p_any >= minutes / 90.0 - 1e-9).all(), position)

    def test_minutes_contract_tolerates_only_roundoff(self) -> None:
        expected = {
            "note": "test",
            "method": "test",
            "source": "test",
            "bins": 1,
            "source_rows": 1,
            "curve": {
                "FWD": {
                    "p60": [0.25],
                    "e_minutes": [30.0],
                    "p_any": [0.5],
                    "n": 1,
                }
            },
        }
        rounded = json.loads(json.dumps(expected))
        rounded["curve"]["FWD"]["p60"] = [0.250002]
        stale = json.loads(json.dumps(expected))
        stale["curve"]["FWD"]["p60"] = [0.26]
        self.assertTrue(payload_equivalent(expected, rounded))
        self.assertFalse(payload_equivalent(expected, stale))


class NewSeasonMappingTests(unittest.TestCase):
    def test_new_season_assignments_derive_from_operational_release(self) -> None:
        with patch.object(mappings, "assert_operational_clubs_ready"):
            assignments = mappings.assignments_for_seasons(["2025-26", "2026-27"])
        current = assignments[assignments["season"].eq("2026-27")]
        self.assertEqual(len(current), 517)
        self.assertTrue(current["first_gameweek"].eq(1).all())
        self.assertTrue(current["last_gameweek"].eq(38).all())
        self.assertFalse(current["fpl_code"].duplicated().any())


if __name__ == "__main__":
    unittest.main()
