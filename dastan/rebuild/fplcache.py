"""Rebuild provenance-checked pre-deadline artifacts from Randdalf/fplcache."""

from __future__ import annotations

import datetime as dt
import json
import lzma
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd

SOURCE_PINS = Path(__file__).resolve().parents[2] / "data" / "source_pins.json"
_PINS = json.loads(SOURCE_PINS.read_text(encoding="utf-8"))
FPLCACHE_COMMIT = str(_PINS["fplcache"]["commit"])
FPLCACHE_REPO = str(_PINS["fplcache"]["repository"])
TREE_URL = (
    f"https://api.github.com/repos/{FPLCACHE_REPO}/git/trees/"
    f"{FPLCACHE_COMMIT}?recursive=1"
)
RAW_URL = f"https://raw.githubusercontent.com/{FPLCACHE_REPO}/" f"{FPLCACHE_COMMIT}/"
USER_AGENT = {"User-Agent": "dastan-data-rebuild/1.0"}
SAFETY_BUFFER = dt.timedelta(minutes=5)
PATH_PATTERN = re.compile(
    r"^cache/(\d{4})/(\d{1,2})/(\d{1,2})/(\d{2})(\d{2})\.json\.xz$"
)
STATUS_RISK = {"a": 0, "d": 1, "s": 2, "i": 3, "u": 4, "n": 4}
SIGNALS_FIRST_SEASON = "2021-22"


def _get(url: str, timeout: int = 90, retries: int = 4) -> bytes:
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=USER_AGENT)
            return urllib.request.urlopen(request, timeout=timeout).read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url} ({last_error})")


def _timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_index(raw_dir: Path, refresh: bool = False) -> dict[dt.datetime, str]:
    cache_dir = raw_dir / "fplcache"
    tree_path = cache_dir / "tree.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if refresh or not tree_path.exists():
        tree_path.write_bytes(_get(TREE_URL))
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    if tree.get("truncated"):
        raise RuntimeError("fplcache Git tree is truncated")
    index: dict[dt.datetime, str] = {}
    for entry in tree.get("tree", []):
        path = entry.get("path", "")
        match = PATH_PATTERN.match(path)
        if entry.get("type") != "blob" or not match:
            continue
        year, month, day, hour, minute = map(int, match.groups())
        try:
            captured = dt.datetime(
                year, month, day, hour, minute, tzinfo=dt.timezone.utc
            )
        except ValueError:
            continue
        index[captured] = path
    if not index:
        raise RuntimeError("fplcache index is empty")
    return index


def fetch_snapshot(raw_dir: Path, source_path: str) -> dict:
    local = raw_dir / "fplcache" / "snapshots" / source_path
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_name(f".{local.name}.tmp")
        temporary.write_bytes(_get(RAW_URL + source_path))
        temporary.replace(local)
    return json.loads(lzma.decompress(local.read_bytes()).decode("utf-8"))


def season_label(events: list[dict]) -> str:
    first = min(_timestamp(event["deadline_time"]) for event in events)
    start = first.year if first.month >= 6 else first.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def probe_date(season: str) -> dt.date:
    start = int(season.split("-")[0])
    if season == "2020-21":
        return dt.date(2021, 5, 1)
    return dt.date(start + 1, 1, 15)


def season_deadlines(
    raw_dir: Path, index: dict[dt.datetime, str], season: str
) -> dict[int, dt.datetime]:
    probe = probe_date(season)
    same_day = sorted(captured for captured in index if captured.date() == probe)
    if not same_day:
        nearby = sorted(index, key=lambda captured: abs((captured.date() - probe).days))
        if not nearby or abs((nearby[0].date() - probe).days) > 21:
            raise RuntimeError(f"no fplcache snapshot near {probe}")
        same_day = [nearby[0]]
    payload = fetch_snapshot(raw_dir, index[same_day[0]])
    events = [
        event for event in payload.get("events", []) if event.get("deadline_time")
    ]
    if not events or season_label(events) != season:
        raise RuntimeError(f"fplcache probe did not resolve season {season}")
    return {int(event["id"]): _timestamp(event["deadline_time"]) for event in events}


def pick_snapshot(
    index: dict[dt.datetime, str], deadline: dt.datetime
) -> tuple[dt.datetime, str] | None:
    cutoff = deadline - SAFETY_BUFFER
    eligible = [captured for captured in index if captured <= cutoff]
    if not eligible:
        return None
    captured = max(eligible)
    return captured, index[captured]


def snapshot_deadline(payload: dict, gameweek: int) -> dt.datetime | None:
    for event in payload.get("events", []):
        if int(event["id"]) == gameweek and event.get("deadline_time"):
            return _timestamp(event["deadline_time"])
    return None


def _accepted_event(payload: dict, gameweek: int, deadline: dt.datetime) -> bool:
    target = next(
        (event for event in payload.get("events", []) if int(event["id"]) == gameweek),
        None,
    )
    return bool(
        target
        and target.get("is_next")
        and _timestamp(target["deadline_time"]) == deadline
    )


def _date(value) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def extract_ep_next(
    payload: dict,
    *,
    season: str,
    gameweek: int,
    deadline: dt.datetime,
    captured_at: dt.datetime,
    source_path: str,
) -> list[dict]:
    if not _accepted_event(payload, gameweek, deadline):
        return []
    if captured_at >= deadline:
        raise RuntimeError("post-deadline fplcache snapshot was selected")
    age_hours = (deadline - captured_at).total_seconds() / 3600
    rows = []
    for player in payload.get("elements", []):
        raw = player.get("ep_next")
        rows.append(
            {
                "season": season,
                "gameweek": gameweek,
                "fpl_code": int(player["code"]),
                "element_id": int(player["id"]),
                "ep_next": None if raw in (None, "") else float(raw),
                "snapshot_at": captured_at,
                "deadline_time": deadline,
                "age_hours": age_hours,
                "source": FPLCACHE_REPO,
                "snapshot_path": source_path,
            }
        )
    return rows


def extract_signals(
    payload: dict,
    *,
    season: str,
    gameweek: int,
    deadline: dt.datetime,
    captured_at: dt.datetime,
    source_path: str,
) -> list[dict]:
    if not _accepted_event(payload, gameweek, deadline):
        return []
    if captured_at >= deadline:
        raise RuntimeError("post-deadline fplcache snapshot was selected")
    deadline_date = deadline.date()
    age_hours = (deadline - captured_at).total_seconds() / 3600
    rows = []
    for player in payload.get("elements", []):
        chance = player.get("chance_of_playing_next_round")
        birth = _date(player.get("birth_date"))
        joined = _date(player.get("team_join_date"))
        rows.append(
            {
                "season": season,
                "gameweek": gameweek,
                "fpl_code": int(player["code"]),
                "sig_status_risk": STATUS_RISK.get(str(player.get("status") or "a"), 0),
                "sig_chance_playing": -1.0 if chance is None else float(chance),
                "sig_has_news": float(bool((player.get("news") or "").strip())),
                "sig_pens_order": float(player.get("penalties_order") or 0),
                "sig_fk_order": float(player.get("direct_freekicks_order") or 0),
                "sig_corners_order": float(
                    player.get("corners_and_indirect_freekicks_order") or 0
                ),
                "sig_age_years": (
                    round((deadline_date - birth).days / 365.25, 2) if birth else -1.0
                ),
                "sig_days_at_club": (
                    float((deadline_date - joined).days) if joined else -1.0
                ),
                "snapshot_at": captured_at,
                "deadline_time": deadline,
                "age_hours": age_hours,
                "source": FPLCACHE_REPO,
                "snapshot_path": source_path,
            }
        )
    return rows


Extractor = Callable[..., list[dict]]


def resolve_gameweek(
    raw_dir: Path,
    index: dict[dt.datetime, str],
    *,
    season: str,
    gameweek: int,
    deadline: dt.datetime,
    max_age_hours: float,
    extractor: Extractor,
    attempts: int = 4,
) -> tuple[list[dict], str | None]:
    seen = set()
    for _ in range(attempts):
        selected = pick_snapshot(index, deadline)
        if selected is None:
            return [], "no snapshot before deadline"
        captured_at, source_path = selected
        age_hours = (deadline - captured_at).total_seconds() / 3600
        if age_hours > max_age_hours:
            return [], f"stale ({age_hours:.1f}h)"
        payload = fetch_snapshot(raw_dir, source_path)
        stated = snapshot_deadline(payload, gameweek)
        if stated is None:
            return [], "gameweek absent from snapshot calendar"
        if stated != deadline:
            if stated in seen:
                return [], "deadline did not converge"
            seen.add(deadline)
            deadline = stated
            continue
        rows = extractor(
            payload,
            season=season,
            gameweek=gameweek,
            deadline=deadline,
            captured_at=captured_at,
            source_path=source_path,
        )
        return (rows, None) if rows else ([], "snapshot does not name gameweek as next")
    return [], "deadline did not converge"


def _build_one(
    raw_dir: Path,
    seasons: list[str],
    *,
    extractor: Extractor,
    max_age_hours: float,
    refresh_index: bool,
) -> tuple[pd.DataFrame, dict]:
    index = load_index(raw_dir, refresh=refresh_index)
    rows: list[dict] = []
    report = {}
    for season in seasons:
        try:
            deadlines = season_deadlines(raw_dir, index, season)
        except RuntimeError as exc:
            report[season] = {"status": "skipped", "reason": str(exc)}
            continue
        refused = {}
        resolved = 0
        for gameweek, deadline in sorted(deadlines.items()):
            if deadline < min(index):
                refused[str(gameweek)] = "predates archive"
                continue
            extracted, reason = resolve_gameweek(
                raw_dir,
                index,
                season=season,
                gameweek=gameweek,
                deadline=deadline,
                max_age_hours=max_age_hours,
                extractor=extractor,
            )
            if reason:
                refused[str(gameweek)] = reason
                continue
            rows.extend(extracted)
            resolved += 1
        report[season] = {
            "status": "ok",
            "gameweeks_resolved": resolved,
            "gameweeks_total": len(deadlines),
            "refused": refused,
        }
        print(f"    {season}: {resolved}/{len(deadlines)} gameweeks", flush=True)
    if not rows:
        raise RuntimeError("no pre-deadline rows were reconstructed")
    frame = (
        pd.DataFrame(rows)
        .sort_values(["season", "gameweek", "fpl_code"], kind="mergesort")
        .reset_index(drop=True)
    )
    duplicate_keys = frame.duplicated(["season", "gameweek", "fpl_code"])
    if duplicate_keys.any():
        raise RuntimeError(
            f"pre-deadline artifact has {int(duplicate_keys.sum())} duplicate keys"
        )
    if frame["snapshot_at"].ge(frame["deadline_time"]).any():
        raise RuntimeError("pre-deadline artifact contains a post-deadline row")
    return frame, report


def build_predeadline_artifacts(
    raw_dir: Path,
    output_dir: Path,
    seasons: list[str],
    *,
    refresh_index: bool = False,
) -> tuple[Path, Path]:
    """Build both published fplcache-derived parquet artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("  rebuilding pre-deadline ep_next", flush=True)
    ep_next, ep_report = _build_one(
        raw_dir,
        seasons,
        extractor=extract_ep_next,
        max_age_hours=30.0,
        refresh_index=refresh_index,
    )
    ep_path = output_dir / "pre_deadline_ep_next.parquet"
    ep_next.to_parquet(ep_path, index=False)
    (output_dir / "pre_deadline_ep_next.json").write_text(
        json.dumps(
            {
                "source": FPLCACHE_REPO,
                "commit": FPLCACHE_COMMIT,
                "rows": len(ep_next),
                "seasons": ep_report,
                "acceptance": "is_next + matching deadline + captured before deadline",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("  rebuilding pre-deadline signals", flush=True)
    signal_seasons = [season for season in seasons if season >= SIGNALS_FIRST_SEASON]
    if not signal_seasons:
        raise RuntimeError(
            f"the released signals artifact starts in {SIGNALS_FIRST_SEASON}"
        )
    signals, signal_report = _build_one(
        raw_dir,
        signal_seasons,
        extractor=extract_signals,
        max_age_hours=12.0,
        refresh_index=False,
    )
    signal_path = output_dir / "pre_deadline_signals.parquet"
    signals.to_parquet(signal_path, index=False)
    (output_dir / "pre_deadline_signals.json").write_text(
        json.dumps(
            {
                "source": FPLCACHE_REPO,
                "commit": FPLCACHE_COMMIT,
                "rows": len(signals),
                "seasons": signal_report,
                "acceptance": "is_next + matching deadline + captured before deadline",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ep_path, signal_path
