# Rebuilding the datasets

Dastan supports two different reproducibility checks:

1. **Release verification** hashes the checked-in artifacts and validates their keys,
   shapes, deadline rules, and identity derivations. It is offline and exact.
2. **Source reconstruction** downloads provider-level inputs, rebuilds the canonical
   joins and 304-column frame, and compares the candidate with the release. Raw
   Understat responses can change after statistical corrections, so this check reports
   drift instead of pretending a later download is byte-identical.

Model retraining itself remains `python -m dastan.reproduce` and starts from the exact
checked-in frame.

## Setup

Use Python 3.12. The optional data environment adds only the Understat client:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt
```

The default full run downloads about 115 MB. Downloads are resumable and cached under
`.cache/dastan-raw`; candidate artifacts go under `.cache/rebuilt-data`. Neither command
overwrites the release under `data/`.

## Commands

```bash
# Exact, offline verification of the published artifacts.
python -m dastan.datasets verify

# Download pinned FPL/Understat archives and required provider fallbacks.
python -m dastan.datasets download

# Build the feature frame and both pre-deadline artifacts from that cache.
python -m dastan.datasets build

# Report key, schema, numeric, hash, and semantic artifact differences.
python -m dastan.datasets compare .cache/rebuilt-data

# Download and build in one command.
python -m dastan.datasets all
```

For a smaller development run, seasons are one comma-separated argument:

```bash
python -m dastan.datasets all \
  --seasons 2024-25 \
  --raw-dir .cache/dastan-raw-2024 \
  --output-dir .cache/rebuilt-2024 \
  --skip-predeadline
```

A one-season feature build does not carry prior-season rolling history, so compare its
keys and schema rather than expecting its long windows to equal the six-season release.

## Inputs and pins

| input | reconstruction source | stability |
|---|---|---|
| FPL gameweek history, rosters, teams | `vaastav/Fantasy-Premier-League` at commit `8c97b2adb123863c3dd581e730f1360e89815ac2` | pinned |
| Understat player/team archives | the same pinned repository where coverage is complete | pinned |
| incomplete Understat histories | Understat through `understatapi==0.7.1` | cached but provider-mutable |
| pre-deadline FPL bootstrap | `Randdalf/fplcache` at commit `6c364bfbd9914649dc5dec016e544be3ae4fe767` | pinned |
| release identity behavior | checked-in training snapshot and assignment timeline | immutable |

The downloader percent-encodes non-ASCII archive paths, checks every mapped FPL
appearance date against cached Understat history, and detects incomplete team seasons.
A complete Understat fallback supersedes a partial archive for that player or season.
Passing `--allow-missing-understat` permits an incomplete exploratory build; do not use
it for a reproduction claim.

Every run writes `downloads.json` with source commits, byte sizes, and SHA-256 hashes.
The build writes `rebuild_manifest.json` with output hashes and the exact source-manifest
hash. Keep both manifests with any published reconstruction result.

## Pipeline

The reconstruction performs these steps:

1. Download each season's `merged_gw.csv`, `players_raw.csv`, and `teams.csv`.
2. Filter obsolete `AM` rows and remove duplicate `(season, fixture, fpl_code)` records.
3. Apply the immutable training assignment intervals, preserving the release's exact
   historical `player_uid` behavior.
4. Join Understat player matches by numeric ID and match date, and team histories by
   normalized club name and date.
5. Recreate OpenFPL player, team, opponent, venue, and league-rank rolling features.
6. Add Dastan availability, defensive-contribution, start-rate, rest, congestion, and
   opponent-allowed families.
7. Shift every historical feature and anchor double-gameweek histories to the common
   FPL deadline.
8. Rebuild `ep_next` and signal tables only from snapshots that name the gameweek as
   `is_next`, agree on its deadline, and were captured before the deadline with a
   five-minute safety buffer.

The full smoke-tested reconstruction produces all 163,072 release keys, all 304 frame
columns, 4,560 EPL team-match rows, 142,173 `ep_next` rows, and 137,982 signal rows. The
two pre-deadline tables are semantically identical to the release after sorting by
`(season, gameweek, fpl_code)`. Feature-value differences can remain where Understat's
current response differs from the historical release capture; `compare` quantifies
those rows and columns with an explicit numeric tolerance.

## Exactness boundaries

`data/release_manifest.json` is the byte-level record of the immutable public release.
Use it when exact bytes matter. Source reconstruction answers a different question:
whether an independent researcher can recover the same population, joins, feature
definitions, and deadline-safe inputs from public sources.

Parquet bytes may differ even for equal tables because writer metadata and row-group
encoding can vary. `compare` therefore reports byte equality and semantic equality
separately. `compare --strict` requires the candidate feature Parquet to have the exact
release hash and is intended only for a controlled release environment.

## Attribution

The feature definitions originate in OpenFPL and are used under MIT. Vaastav's
repository is MIT; `fplcache` is Unlicensed; `understatAPI` is MIT. FPL and Understat
retain rights in their source data and their respective terms still apply.
