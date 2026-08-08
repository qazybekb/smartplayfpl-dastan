from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from dastan import mappings
from dastan.rebuild.features import compute_league_ranks
from dastan.rebuild.fplcache import (
    extract_ep_next,
    extract_signals,
    pick_snapshot,
)
from dastan.rebuild.sources import _get


UTC = dt.timezone.utc


class MappingTests(unittest.TestCase):
    def test_published_identity_artifacts_are_derived_from_release(self) -> None:
        frames = mappings.verify()
        self.assertEqual(len(frames["training"]), 1_564)
        self.assertEqual(len(frames["assignments"]), 4_199)
        self.assertEqual(int(frames["current"]["understat_id"].notna().sum()), 517)
        training = frames["training"].set_index("fpl_code")["understat_id"]
        corrected = frames["corrected"].set_index("fpl_code")["understat_id"]
        self.assertEqual(int(training.loc[437688]), 9082)
        self.assertEqual(int(corrected.loc[437688]), 9216)
        self.assertEqual(int(training.loc[515501]), 5191)
        self.assertEqual(int(corrected.loc[515501]), 10576)

    def test_operational_projection_fails_closed_after_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / mappings.OPERATIONAL_PLAYERS.name
            changed.write_bytes(mappings.OPERATIONAL_PLAYERS.read_bytes() + b"\n")
            with patch.object(mappings, "OPERATIONAL_PLAYERS", changed):
                with self.assertRaisesRegex(RuntimeError, "accepted operational release"):
                    mappings.verify_operational_release()


class FeatureRebuildTests(unittest.TestCase):
    def test_league_rank_ties_preserve_first_appearance_order(self) -> None:
        rows = [
            ("Yankee", "2024-08-01", 1, 1, 1),
            ("Zulu", "2024-08-01", 1, 1, 1),
            ("Alpha", "2024-08-02", 0, 0, 1),
            ("Omega", "2024-08-02", 0, 0, 1),
            ("Aardvark", "2024-08-03", 0, 0, 1),
            ("Beta", "2024-08-03", 0, 0, 1),
        ]
        frame = pd.DataFrame(
            rows, columns=["understat_team", "date", "scored", "missed", "pts"]
        )
        frame["season"] = "2024-25"
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        ranks = compute_league_ranks(frame)

        self.assertEqual(ranks[("Alpha", "2024-25", "2024-08-02 00:00")], 3)
        self.assertEqual(ranks[("Omega", "2024-25", "2024-08-02 00:00")], 4)


class FplcacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.deadline = dt.datetime(2026, 8, 15, 10, tzinfo=UTC)
        self.captured = self.deadline - dt.timedelta(hours=4)
        self.payload = {
            "events": [
                {
                    "id": 1,
                    "is_next": True,
                    "deadline_time": self.deadline.isoformat(),
                }
            ],
            "elements": [
                {
                    "id": 7,
                    "code": 700,
                    "ep_next": "5.5",
                    "status": "d",
                    "chance_of_playing_next_round": 75,
                    "news": "Knock",
                    "penalties_order": 1,
                    "direct_freekicks_order": 2,
                    "corners_and_indirect_freekicks_order": 3,
                    "birth_date": "2000-01-01",
                    "team_join_date": "2024-07-01",
                }
            ],
        }

    def test_snapshot_selection_keeps_five_minute_safety_buffer(self) -> None:
        index = {
            self.deadline - dt.timedelta(minutes=6): "accepted",
            self.deadline - dt.timedelta(minutes=4): "too-close",
        }
        self.assertEqual(
            pick_snapshot(index, self.deadline),
            (self.deadline - dt.timedelta(minutes=6), "accepted"),
        )

    def test_extractors_enforce_next_gameweek_and_deadline(self) -> None:
        kwargs = {
            "season": "2026-27",
            "gameweek": 1,
            "deadline": self.deadline,
            "captured_at": self.captured,
            "source_path": "cache/example.json.xz",
        }
        ep = extract_ep_next(self.payload, **kwargs)
        signals = extract_signals(self.payload, **kwargs)
        self.assertEqual(ep[0]["ep_next"], 5.5)
        self.assertEqual(signals[0]["sig_status_risk"], 1)
        self.assertEqual(signals[0]["sig_chance_playing"], 75.0)

        self.payload["events"][0]["is_next"] = False
        self.assertEqual(extract_ep_next(self.payload, **kwargs), [])
        self.assertEqual(extract_signals(self.payload, **kwargs), [])


class SourceDownloadTests(unittest.TestCase):
    @patch("dastan.rebuild.sources.urllib.request.urlopen")
    def test_non_ascii_source_paths_are_percent_encoded(self, urlopen) -> None:
        urlopen.return_value = SimpleNamespace(read=lambda: b"ok")
        self.assertEqual(_get("https://example.test/Højbjerg.csv"), b"ok")
        request = urlopen.call_args.args[0]
        self.assertIn("H%C3%B8jbjerg.csv", request.full_url)


if __name__ == "__main__":
    unittest.main()
