"""Download and normalize the raw FPL and Understat inputs used by Dastan."""

from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import mappings
from .features import fpl_to_understat, parse_ppda

SOURCE_PINS = Path(__file__).resolve().parents[2] / "data" / "source_pins.json"
_PINS = json.loads(SOURCE_PINS.read_text(encoding="utf-8"))
VAASTAV_COMMIT = str(_PINS["vaastav"]["commit"])
VAASTAV_RAW = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    f"{VAASTAV_COMMIT}/"
)
VAASTAV_TREE = (
    "https://api.github.com/repos/vaastav/Fantasy-Premier-League/git/trees/"
    f"{VAASTAV_COMMIT}?recursive=1"
)
USER_AGENT = {"User-Agent": "dastan-data-rebuild/1.0"}
CORE_FILES = ("gws/merged_gw.csv", "players_raw.csv", "teams.csv")
POSITION_VALUES = {"GK", "GKP", "DEF", "MID", "FWD"}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(url: str, timeout: int = 90, retries: int = 4) -> bytes:
    url = urllib.parse.quote(url, safe=":/?&=%")
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


def _download(url: str, path: Path, force: bool = False) -> Path:
    if path.exists() and path.stat().st_size and not force:
        return path
    payload = _get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def _download_many(
    jobs: list[tuple[str, Path]], workers: int, force: bool
) -> list[Path]:
    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_download, url, path, force): (url, path) for url, path in jobs
        }
        for index, future in enumerate(as_completed(futures), 1):
            url, path = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"failed to download {url} -> {path}") from exc
            if index % 100 == 0:
                print(f"    downloaded {index:,}/{len(jobs):,}", flush=True)
    return results


def _tree(raw_dir: Path, force: bool) -> list[str]:
    path = raw_dir / "vaastav" / "tree.json"
    if force or not path.exists():
        _download(VAASTAV_TREE, path, force=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("truncated"):
        raise RuntimeError(
            "Vaastav Git tree is truncated; source discovery is incomplete"
        )
    return [entry["path"] for entry in payload["tree"] if entry["type"] == "blob"]


def _core_path(raw_dir: Path, season: str, filename: str) -> Path:
    return raw_dir / "vaastav" / season / Path(filename).name


def _needed_understat_ids(raw_dir: Path, seasons: list[str]) -> set[int]:
    del raw_dir
    assignments = mappings.assignments_for_seasons(seasons)
    return set(
        assignments.loc[assignments["season"].isin(seasons), "understat_id"].astype(int)
    )


def _expected_understat_player_dates(
    raw_dir: Path, seasons: list[str]
) -> dict[int, set[object]]:
    """Played FPL dates that should have a matching Understat player row."""
    assignments = mappings.assignments_for_seasons(seasons)
    expected: dict[int, set[object]] = {}
    for season in seasons:
        players = pd.read_csv(
            _core_path(raw_dir, season, "players_raw.csv"), usecols=["id", "code"]
        ).rename(columns={"id": "element", "code": "fpl_code"})
        gameweeks = pd.read_csv(
            _core_path(raw_dir, season, "merged_gw.csv"), low_memory=False
        )
        gameweeks["gameweek"] = pd.to_numeric(
            gameweeks["GW"] if "GW" in gameweeks else gameweeks["round"],
            errors="raise",
        ).astype(int)
        gameweeks = gameweeks[
            gameweeks["position"].isin(POSITION_VALUES)
            & pd.to_numeric(gameweeks["minutes"], errors="coerce").gt(0)
        ]
        season_assignments = assignments[assignments["season"].eq(season)].drop(
            columns=["season", "mapped_rows"]
        )
        gameweeks = gameweeks.merge(
            players, on="element", how="left", validate="many_to_one"
        ).merge(season_assignments, on="fpl_code", how="left", validate="many_to_one")
        active = gameweeks["gameweek"].between(
            gameweeks["first_gameweek"], gameweeks["last_gameweek"]
        )
        gameweeks["understat_id"] = gameweeks["understat_id"].where(active)
        gameweeks["match_date"] = pd.to_datetime(
            gameweeks["kickoff_time"], utc=True, errors="coerce"
        ).dt.date
        for player_id, rows in gameweeks.dropna(
            subset=["understat_id", "match_date"]
        ).groupby("understat_id"):
            expected.setdefault(int(player_id), set()).update(rows["match_date"])
    return expected


def _incomplete_pinned_players(
    raw_dir: Path, seasons: list[str], player_ids: set[int]
) -> set[int]:
    expected = _expected_understat_player_dates(raw_dir, seasons)
    incomplete: set[int] = set()
    root = raw_dir / "vaastav" / "understat" / "players"
    for player_id in player_ids:
        path = root / f"{player_id}.csv"
        if not path.exists():
            incomplete.add(player_id)
            continue
        history = pd.read_csv(path, usecols=["date"], low_memory=False)
        actual = set(pd.to_datetime(history["date"], errors="coerce").dt.date.dropna())
        if expected.get(player_id, set()) - actual:
            incomplete.add(player_id)
    return incomplete


def _incomplete_team_seasons(
    raw_dir: Path, seasons: list[str], team_paths: dict[str, list[Path]]
) -> list[str]:
    incomplete = []
    for season in seasons:
        gameweeks = pd.read_csv(
            _core_path(raw_dir, season, "merged_gw.csv"), usecols=["fixture"]
        )
        expected_rows = 2 * gameweeks["fixture"].nunique()
        epl_teams = _epl_teams(raw_dir, [season])[season]
        actual_rows = sum(
            len(_load_vaastav_team_file(path, season))
            for path in team_paths.get(season, [])
            if fpl_to_understat(_team_name_from_path(path)) in epl_teams
        )
        if actual_rows < expected_rows:
            incomplete.append(season)
    return incomplete


def _latest_player_paths(paths: list[str], wanted: set[int]) -> dict[int, str]:
    latest: dict[int, tuple[str, str]] = {}
    pattern = re.compile(r"^data/(20\d\d-\d\d)/understat/.+_(\d+)\.csv$")
    for path in paths:
        match = pattern.match(path)
        if not match:
            continue
        season, player_id = match.group(1), int(match.group(2))
        if player_id not in wanted:
            continue
        if player_id not in latest or season > latest[player_id][0]:
            latest[player_id] = (season, path)
    return {player_id: path for player_id, (_, path) in latest.items()}


def _team_paths(paths: list[str], season: str) -> list[str]:
    prefix = f"data/{season}/understat/understat_"
    return [
        path
        for path in paths
        if path.startswith(prefix)
        and path.endswith(".csv")
        and not path.endswith("understat_player.csv")
    ]


def _fetch_understat_fallbacks(
    raw_dir: Path,
    missing_players: list[int],
    missing_team_seasons: list[str],
    *,
    force: bool,
    allow_missing: bool,
) -> None:
    if not missing_players and not missing_team_seasons:
        return
    try:
        from understatapi import UnderstatClient
    except ImportError as exc:
        raise RuntimeError(
            "Understat fallbacks are required; install requirements-data.txt"
        ) from exc

    failures: list[str] = []
    with UnderstatClient() as client:
        for index, player_id in enumerate(missing_players, 1):
            path = raw_dir / "understat" / "players" / f"{player_id}.json"
            if path.exists() and path.stat().st_size and not force:
                continue
            payload = None
            for attempt in range(3):
                try:
                    payload = client.player(player=str(player_id)).get_match_data()
                    break
                except Exception:
                    time.sleep(2**attempt)
            if payload is None:
                failures.append(f"player:{player_id}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            if index % 25 == 0:
                print(
                    f"    Understat fallback players {index:,}/{len(missing_players):,}",
                    flush=True,
                )
            time.sleep(0.15)

        for season in missing_team_seasons:
            path = raw_dir / "understat" / "teams" / f"{season}.json"
            if path.exists() and path.stat().st_size and not force:
                continue
            payload = None
            for attempt in range(3):
                try:
                    payload = client.league(league="EPL").get_team_data(
                        season=season.split("-")[0]
                    )
                    break
                except Exception:
                    time.sleep(2**attempt)
            if not payload:
                failures.append(f"teams:{season}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    if failures and not allow_missing:
        raise RuntimeError(
            f"Understat could not resolve {len(failures)} sources: {failures[:10]}; "
            "rerun or pass --allow-missing-understat"
        )
    if failures:
        print(f"  WARNING: {len(failures)} Understat fallbacks remain missing")


def write_download_manifest(raw_dir: Path, seasons: list[str]) -> Path:
    files = {}
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name == "downloads.json":
            continue
        relative = str(path.relative_to(raw_dir))
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "sources": {
            "vaastav": {
                "repository": "https://github.com/vaastav/Fantasy-Premier-League",
                "commit": VAASTAV_COMMIT,
            },
            "fplcache": {
                "repository": f"https://github.com/{_PINS['fplcache']['repository']}",
                "commit": str(_PINS["fplcache"]["commit"]),
            },
            "understat": {
                "repository_cache": "Vaastav where available",
                "live_fallback": "Understat via understatapi for IDs absent from the pinned cache",
            },
        },
        "files": files,
    }
    path = raw_dir / "downloads.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def download_sources(
    raw_dir: Path,
    seasons: list[str],
    *,
    workers: int = 12,
    force: bool = False,
    allow_missing_understat: bool = False,
) -> Path:
    """Download pinned FPL/history inputs and cache mutable Understat fallbacks."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("  downloading pinned FPL season files", flush=True)
    core_jobs = []
    for season in seasons:
        for filename in CORE_FILES:
            remote = f"data/{season}/{filename}"
            core_jobs.append(
                (VAASTAV_RAW + remote, _core_path(raw_dir, season, filename))
            )
    _download_many(core_jobs, workers=min(workers, len(core_jobs)), force=force)

    paths = _tree(raw_dir, force)
    wanted = _needed_understat_ids(raw_dir, seasons)
    player_paths = _latest_player_paths(paths, wanted)
    print(
        f"  downloading {len(player_paths):,} pinned Understat player histories",
        flush=True,
    )
    player_jobs = [
        (
            VAASTAV_RAW + source,
            raw_dir / "vaastav" / "understat" / "players" / f"{player_id}.csv",
        )
        for player_id, source in sorted(player_paths.items())
    ]
    _download_many(player_jobs, workers=workers, force=force)

    team_jobs: list[tuple[str, Path]] = []
    local_team_paths: dict[str, list[Path]] = {}
    missing_team_seasons: list[str] = []
    for season in seasons:
        found = _team_paths(paths, season)
        if not found:
            missing_team_seasons.append(season)
            continue
        for source in found:
            destination = (
                raw_dir / "vaastav" / "understat" / "teams" / season / Path(source).name
            )
            local_team_paths.setdefault(season, []).append(destination)
            team_jobs.append(
                (
                    VAASTAV_RAW + source,
                    destination,
                )
            )
    print(
        f"  downloading {len(team_jobs):,} pinned Understat team histories", flush=True
    )
    _download_many(team_jobs, workers=min(workers, max(1, len(team_jobs))), force=force)

    incomplete_players = _incomplete_pinned_players(raw_dir, seasons, wanted)
    missing_players = sorted((wanted - set(player_paths)) | incomplete_players)
    incomplete_teams = _incomplete_team_seasons(raw_dir, seasons, local_team_paths)
    missing_team_seasons = sorted(set(missing_team_seasons) | set(incomplete_teams))
    print(
        f"  caching {len(missing_players):,} player and "
        f"{len(missing_team_seasons):,} team-season Understat fallbacks",
        flush=True,
    )
    _fetch_understat_fallbacks(
        raw_dir,
        missing_players,
        missing_team_seasons,
        force=force,
        allow_missing=allow_missing_understat,
    )
    manifest = write_download_manifest(raw_dir, seasons)
    print(f"  source manifest -> {manifest}", flush=True)
    return manifest


def _epl_teams(raw_dir: Path, seasons: list[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for season in seasons:
        teams = pd.read_csv(_core_path(raw_dir, season, "teams.csv"))
        result[season] = {fpl_to_understat(name) for name in teams["name"].astype(str)}
    return result


def _load_player_history(path: Path, player_id: int) -> pd.DataFrame:
    if path.suffix == ".csv":
        out = pd.read_csv(path, low_memory=False)
    else:
        out = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if out.empty:
        return out
    out["understat_id"] = int(player_id)
    out["understat_season"] = out["season"].astype(str)
    out["season"] = out["understat_season"].map(
        lambda value: f"{int(value)}-{str(int(value) + 1)[-2:]}"
    )
    return out


def load_understat_player_matches(raw_dir: Path, seasons: list[str]) -> pd.DataFrame:
    """Load cached player histories and filter out non-EPL matches."""
    team_sets = _epl_teams(raw_dir, seasons)
    paths: dict[int, Path] = {}
    for path in (raw_dir / "vaastav" / "understat" / "players").glob("*.csv"):
        paths[int(path.stem)] = path
    for path in (raw_dir / "understat" / "players").glob("*.json"):
        # A fallback is a complete provider snapshot and supersedes the partial
        # pinned history whose missing dates caused it to be fetched.
        paths[int(path.stem)] = path

    frames = []
    for player_id, path in sorted(paths.items()):
        history = _load_player_history(path, player_id)
        if history.empty:
            continue
        history = history[history["season"].isin(seasons)].copy()
        if history.empty:
            continue
        keep = [
            home in team_sets.get(season, set())
            and away in team_sets.get(season, set())
            for season, home, away in zip(
                history["season"], history["h_team"], history["a_team"]
            )
        ]
        history = history.loc[keep]
        if len(history):
            frames.append(history)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date_only"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.sort_values(["understat_id", "date"], kind="mergesort")
    duplicates = out.duplicated(["understat_id", "date_only"], keep=False)
    if duplicates.any():
        sample = out.loc[
            duplicates, ["understat_id", "date", "h_team", "a_team"]
        ].head()
        raise RuntimeError(f"Understat player/date keys are not unique:\n{sample}")
    return out.reset_index(drop=True)


def _team_name_from_path(path: Path) -> str:
    value = path.stem.removeprefix("understat_").replace("_", " ")
    return html.unescape(value)


def _load_vaastav_team_file(path: Path, season: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    start_year = int(season.split("-")[0])
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[
        parsed_dates.ge(pd.Timestamp(start_year, 7, 1))
        & parsed_dates.lt(pd.Timestamp(start_year + 1, 7, 1))
    ].copy()
    frame["season"] = season
    frame["understat_team"] = _team_name_from_path(path)
    frame["is_home"] = frame["h_a"].astype(str).str.lower().eq("h").astype(float)
    for source, prefix in (("ppda", "ppda"), ("ppda_allowed", "ppda_allowed")):
        values = frame[source].map(parse_ppda)
        frame[f"{prefix}_att"] = values.map(lambda pair: pair[0])
        frame[f"{prefix}_def"] = values.map(lambda pair: pair[1])
    return frame


def _load_live_team_file(path: Path, season: str) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    values = payload.values() if isinstance(payload, dict) else payload
    for team in values:
        title = team.get("title", "")
        for match in team.get("history", []):
            ppda_att, ppda_def = parse_ppda(match.get("ppda"))
            allowed_att, allowed_def = parse_ppda(match.get("ppda_allowed"))
            rows.append(
                {
                    **match,
                    "season": season,
                    "understat_team": title,
                    "is_home": float(str(match.get("h_a", "")).lower() == "h"),
                    "ppda_att": ppda_att,
                    "ppda_def": ppda_def,
                    "ppda_allowed_att": allowed_att,
                    "ppda_allowed_def": allowed_def,
                }
            )
    return pd.DataFrame(rows)


def load_understat_team_matches(raw_dir: Path, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        live = raw_dir / "understat" / "teams" / f"{season}.json"
        pinned = sorted(
            (raw_dir / "vaastav" / "understat" / "teams" / season).glob("*.csv")
        )
        if live.exists():
            frames.append(_load_live_team_file(live, season))
            continue
        if pinned:
            frames.extend(_load_vaastav_team_file(path, season) for path in pinned)
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("no Understat team histories are available")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    keep = pd.Series(False, index=out.index)
    team_sets = _epl_teams(raw_dir, seasons)
    for season in seasons:
        gameweeks = pd.read_csv(
            _core_path(raw_dir, season, "merged_gw.csv"), usecols=["kickoff_time"]
        )
        kickoffs = pd.to_datetime(gameweeks["kickoff_time"], utc=True, errors="coerce")
        naive_kickoffs = kickoffs.dt.tz_localize(None)
        first_date, last_date = naive_kickoffs.min(), naive_kickoffs.max()
        keep |= (
            out["season"].eq(season)
            & out["understat_team"].map(fpl_to_understat).isin(team_sets[season])
            & out["date"].between(first_date, last_date)
        )
    out = out.loc[keep].copy()
    required = [
        "season",
        "date",
        "understat_team",
        "is_home",
        "scored",
        "missed",
        "xG",
        "xGA",
        "deep",
        "deep_allowed",
        "ppda_att",
        "ppda_def",
        "ppda_allowed_att",
        "ppda_allowed_def",
        "pts",
    ]
    for column in required:
        if column not in out:
            out[column] = np.nan
    return out[required].reset_index(drop=True)


def _merge_understat_players(
    fpl: pd.DataFrame, understat: pd.DataFrame
) -> pd.DataFrame:
    output_columns = {
        "time": "us_minutes",
        "goals": "us_goals",
        "assists": "us_assists",
        "shots": "us_shots",
        "key_passes": "us_key_passes",
        "xG": "us_xG",
        "xA": "us_xA",
        "npg": "us_npg",
        "npxG": "us_npxG",
        "xGChain": "us_xGChain",
        "xGBuildup": "us_xGBuildup",
    }
    if understat.empty:
        out = fpl.copy()
        for target in output_columns.values():
            out[target] = np.nan
        return out
    available = [source for source in output_columns if source in understat]
    right = understat[["understat_id", "date_only", *available]].rename(
        columns={source: output_columns[source] for source in available}
    )
    out = fpl.copy()
    out["date_only"] = out["kickoff_time"].dt.date
    out = out.merge(
        right,
        on=["understat_id", "date_only"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["date_only"])
    for target in output_columns.values():
        if target not in out:
            out[target] = np.nan
    return out


def build_canonical_matches(
    raw_dir: Path, seasons: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return canonical player matches, team matches, and player lookup."""
    assignments = mappings.assignments_for_seasons(seasons)
    all_rows = []
    for season in seasons:
        gameweeks = pd.read_csv(
            _core_path(raw_dir, season, "merged_gw.csv"), low_memory=False
        )
        players = pd.read_csv(
            _core_path(raw_dir, season, "players_raw.csv"), low_memory=False
        )[["id", "code"]].rename(columns={"id": "element", "code": "fpl_code"})
        teams = pd.read_csv(_core_path(raw_dir, season, "teams.csv"))
        team_names = teams.set_index("id")["name"].to_dict()

        gameweeks = gameweeks[gameweeks["position"].isin(POSITION_VALUES)].copy()
        gameweeks["position"] = gameweeks["position"].replace({"GK": "GKP"})
        gameweeks["gameweek"] = pd.to_numeric(
            gameweeks["GW"] if "GW" in gameweeks else gameweeks["round"],
            errors="raise",
        ).astype(int)
        gameweeks["season"] = season
        gameweeks = gameweeks.merge(
            players, on="element", how="left", validate="many_to_one"
        )
        if gameweeks["fpl_code"].isna().any():
            raise RuntimeError(f"{season} has FPL rows without a stable player code")
        gameweeks["fpl_code"] = gameweeks["fpl_code"].astype(int)
        gameweeks = gameweeks.rename(
            columns={
                "name": "player_name",
                "team": "team_name",
                "was_home": "is_home",
            }
        )
        gameweeks["opponent_team_name"] = gameweeks["opponent_team"].map(team_names)
        gameweeks["us_opponent"] = gameweeks["opponent_team_name"].map(fpl_to_understat)
        gameweeks["kickoff_time"] = pd.to_datetime(
            gameweeks["kickoff_time"], utc=True, errors="coerce"
        )
        gameweeks["match_date"] = gameweeks["kickoff_time"].dt.date.astype(str)
        season_assignments = assignments[assignments["season"].eq(season)].drop(
            columns=["season", "mapped_rows"]
        )
        gameweeks = gameweeks.merge(
            season_assignments, on="fpl_code", how="left", validate="many_to_one"
        )
        active = gameweeks["gameweek"].between(
            gameweeks["first_gameweek"], gameweeks["last_gameweek"]
        )
        gameweeks["understat_id"] = gameweeks["understat_id"].where(active)
        gameweeks = gameweeks.drop(columns=["first_gameweek", "last_gameweek"])
        # Vaastav's xP is not provenance-safe and can be post-match. The verified
        # fplcache artifact is joined separately by dastan.data.load().
        gameweeks["expected_points_pre_deadline"] = 0.0
        for column in [
            "starts",
            "clearances_blocks_interceptions",
            "defensive_contribution",
            "recoveries",
            "tackles",
        ]:
            if column not in gameweeks:
                gameweeks[column] = np.nan
        gameweeks = gameweeks.drop_duplicates(
            ["season", "fixture", "fpl_code"], keep="last"
        )
        all_rows.append(gameweeks)

    fpl = pd.concat(all_rows, ignore_index=True)
    understat_players = load_understat_player_matches(raw_dir, seasons)
    fpl = _merge_understat_players(fpl, understat_players)
    teams = load_understat_team_matches(raw_dir, seasons)
    player_lookup = (
        fpl[["season", "fpl_code", "element", "player_name", "position"]]
        .drop_duplicates(["season", "fpl_code"])
        .sort_values(["season", "player_name", "fpl_code"], kind="mergesort")
        .reset_index(drop=True)
    )
    return fpl, teams, player_lookup
