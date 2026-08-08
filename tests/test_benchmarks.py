from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dastan.benchmark_openfpl import (
    _eligible,
    _paired_bootstrap,
    _return_band_errors,
)
from dastan.metrics import score

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_frame() -> pd.DataFrame:
    rows = []
    for gameweek in (1, 2):
        for player in range(25):
            actual = float((player * 3 + gameweek) % 11)
            rows.append(
                {
                    "season": "2025-26",
                    "gameweek": gameweek,
                    "fpl_code": gameweek * 100 + player,
                    "minutes": 90,
                    "actual": actual,
                    "dastan": actual * 0.8 + player / 100,
                    "openfpl": actual * 0.6 - player / 100,
                }
            )
    return pd.DataFrame(rows)


class MetricCoverageTests(unittest.TestCase):
    def test_one_missing_baseline_value_does_not_discard_its_gameweek(self) -> None:
        frame = metric_frame()
        frame.loc[0, "dastan"] = np.nan

        result = score(frame, "dastan")

        self.assertEqual(result["gameweeks"], 2)
        self.assertEqual(result["rows"], 49)

    def test_missing_sentinel_is_exact_and_negative_forecasts_remain_valid(self) -> None:
        frame = pd.DataFrame({"ep_next": [-1.0, -1.5, 0.0, np.nan, np.inf]})

        eligible = _eligible(frame, "ep_next", missing_value=-1.0)

        self.assertEqual(eligible["ep_next"].tolist(), [-1.5, 0.0])


class OpenFplProtocolTests(unittest.TestCase):
    def test_return_bands_are_mutually_exclusive_and_exhaustive(self) -> None:
        frame = pd.DataFrame(
            {
                "season": ["2025-26"] * 4,
                "gameweek": [1] * 4,
                "fpl_code": [1, 2, 3, 4],
                "minutes": [0, 45, 90, 90],
                # A retrospective -1 on a zero-minute row exists in the released
                # benchmark. Participation, not a perfect-zero assumption, defines
                # OpenFPL's non-player band here.
                "actual": [-1, 2, 4, 8],
                "dastan": [0.2, 1.4, 3.2, 5.0],
                "openfpl": [0.4, 1.6, 2.8, 4.5],
            }
        )

        report = _return_band_errors(frame, ("dastan", "openfpl"))

        self.assertEqual({band: value["rows"] for band, value in report.items()}, {
            "zeros": 1,
            "blanks": 1,
            "tickers": 1,
            "haulers": 1,
        })

    def test_bootstrap_is_paired_by_gameweek_and_deterministic(self) -> None:
        frame = metric_frame()

        first = _paired_bootstrap(
            frame, "dastan", "openfpl", "all", samples=500, seed=7
        )
        second = _paired_bootstrap(
            frame, "dastan", "openfpl", "all", samples=500, seed=7
        )

        self.assertEqual(first, second)
        self.assertEqual(first["gameweeks"], 2)
        self.assertLessEqual(first["obj"]["lower"], first["obj"]["delta"])
        self.assertGreaterEqual(first["obj"]["upper"], first["obj"]["delta"])

    def test_published_report_is_paired_and_bound_to_release_inputs(self) -> None:
        report = json.loads((ROOT / "docs" / "openfpl_benchmark.json").read_text())

        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["model"]["features"], 286)
        self.assertEqual(report["model"]["seeds"], [42, 7, 2026])
        self.assertEqual(report["regime"]["fixture_rows"], 18_173)
        self.assertEqual(report["regime"]["player_gameweeks"], 18_173)
        self.assertEqual(report["paired_baselines"]["fpl_ep_next"]["gameweeks"], 24)

        provenance = report["provenance"]
        self.assertEqual(
            provenance["openfpl_predictions"]["sha256"],
            sha256(ROOT / provenance["openfpl_predictions"]["path"]),
        )
        self.assertEqual(
            provenance["openfpl_row_keys"]["sha256"],
            sha256(ROOT / provenance["openfpl_row_keys"]["path"]),
        )
        self.assertEqual(
            provenance["data_release_manifest_sha256"],
            sha256(ROOT / "data" / "release_manifest.json"),
        )
        self.assertEqual(
            provenance["model_artifact_manifest_sha256"],
            sha256(ROOT / "models" / "artifact_manifest.json"),
        )
        for source in provenance["code"].values():
            self.assertEqual(source["sha256"], sha256(ROOT / source["path"]))


class WalkForwardReportTests(unittest.TestCase):
    def test_published_clean_report_uses_the_current_feature_contract(self) -> None:
        report = json.loads((ROOT / "docs" / "evaluation.json").read_text())

        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["model"], {
            "features": 286,
            "seeds": [42, 7, 2026],
        })
        self.assertEqual(report["protocol"]["scope"], "clean")
        self.assertEqual(len(report["blocks"]), 3)
        self.assertTrue(all(block["tag"] == "clean" for block in report["blocks"].values()))

        blocks = list(report["blocks"].values())
        self.assertEqual(sum(block["test_rows"] for block in blocks), 17_986)
        self.assertEqual(sum(block["player_gameweeks"] for block in blocks), 17_622)
        self.assertEqual(
            sum(block["paired_baselines"]["FPL ep_next"]["rows"] for block in blocks),
            17_307,
        )
        self.assertTrue(
            all(
                block["paired_baselines"]["rolling-5 points"]["baseline"]["all"]["gameweeks"]
                == 8
                for block in blocks
            )
        )

        provenance = report["provenance"]
        self.assertEqual(
            provenance["data_release_manifest_sha256"],
            sha256(ROOT / "data" / "release_manifest.json"),
        )
        self.assertEqual(
            provenance["model_artifact_manifest_sha256"],
            sha256(ROOT / "models" / "artifact_manifest.json"),
        )
        for source in provenance["code"].values():
            self.assertEqual(source["sha256"], sha256(ROOT / source["path"]))

    def test_human_readable_accuracy_numbers_match_the_reports(self) -> None:
        accuracy = (ROOT / "docs" / "ACCURACY.md").read_text()
        readme = (ROOT / "README.md").read_text()

        for expected in ("0.6090", "0.6097", "0.5321", "0.5641", "0.5044"):
            self.assertIn(expected, accuracy)
            self.assertIn(expected, readme)
        for stale in ("0.6072", "0.5602", "0.0558"):
            self.assertNotIn(stale, accuracy)
            self.assertNotIn(stale, readme)


if __name__ == "__main__":
    unittest.main()
