# Data

Everything needed to retrain Dastan is in this repository. No API keys or database are
required. Optional public-source reconstruction code is documented in
[`REBUILDING_DATA.md`](REBUILDING_DATA.md).

| file | rows | size |
|---|---|---|
| `data/features.parquet` | 163,072 | 37 MB |
| `data/pre_deadline_ep_next.parquet` | 142,173 | 362 KB |
| `data/pre_deadline_signals.parquet` | 137,982 | 438 KB |
| `data/openfpl_predictions.csv` | 18,173 | 1.3 MB |
| `data/openfpl_row_keys.csv` | 18,173 | 444 KB |
| `data/players.csv` | 4,717 | 130 KB |
| `data/mappings/fpl_understat_training_snapshot.csv` | 1,564 | 84 KB |
| `data/mappings/fpl_understat_training_assignments.csv` | 4,199 | 115 KB |
| `data/mappings/fpl_understat_players.csv` | 1,564 | 84 KB |
| `data/mappings/fpl_understat_current.csv` | 577 | 51 KB |
| `docs/walkforward_predictions.parquet` | 17,622 | 539 KB |

The walk-forward file is an evaluation artifact rather than training input. It retains
the three-seed averaged Dastan prediction, the separately fitted no-`ep_next` arm,
`p60`, actual points/minutes, position, and each public baseline for every clean
out-of-sample player-gameweek. It is the source for
`docs/baseline_benchmark.json` and can regenerate the published tables without fitting
the models again:

```bash
python -m dastan.benchmark_baselines \
  --from-predictions docs/walkforward_predictions.parquet
```

---

## 1. `features.parquet` — the training frame

One row per **player per fixture**, six seasons (2020-21 through 2025-26).

Identity columns: `season`, `gameweek`, `fixture`, `fpl_code`, `position`,
`team_name`, `kickoff_time`. **Player names are not in the frame** — join
`data/players.csv` on `(season, fpl_code)`.
Targets: `target_points`, `target_bucket`, `target_minutes_ge60`, `minutes`.
Features: 286 columns, listed in `models/feature_cols.json`.

### Sources

| block | source |
|---|---|
| FPL scoring, minutes, price, ownership, transfers | the official FPL API |
| Shots, xG, xA, xGChain, xGBuildup, deep passes, PPDA | Understat |
| Defensive contributions, CBIT, recoveries, tackles | FPL API (2025-26 onward) |
| Rolling feature construction | [OpenFPL](https://github.com/daniegr/OpenFPL) (MIT) |

### `fpl_code`, not `element`

**FPL renumbers `element` IDs every season.** Player 233 in 2023-24 is a different human
from player 233 in 2024-25. Joining on `element` across seasons silently mixes players
together and produces rolling features that are an average of two careers.

`fpl_code` is stable across seasons and is the join key everywhere in this repo.
**1,130 of the 1,959 players in this dataset had their `element` id change between
seasons** — joining on it merges careers together.

`data/players.csv` maps `(season, fpl_code)` to a name and position. It deliberately
carries **no club column**: 3,683 rows belong to players who changed club mid-season, and
the frame's own per-gameweek `team_name` is correct for those where a season-level
lookup would not be. Use `team_name` from the frame.

### FPL to Understat identities

`data/mappings/fpl_understat_training_snapshot.csv` publishes the exact 1,564-player
identity snapshot present in the release. It intentionally retains two identities later
confirmed to be wrong. `fpl_understat_training_assignments.csv` records the precise
season/gameweek intervals in which those IDs were attached to rows; use both artifacts
for release reconstruction.

`data/mappings/fpl_understat_players.csv` applies the evidence-backed audit and is the
map recommended for new joins. Load it with `dastan.mappings.load()`. Use
`load_training()` and `load_training_assignments()` only when reproducing the historical
release. The 2026-27 roster map covers 518 of 577 players and keeps 59 unresolved rather
than guessing by name.

The exact snapshot contains three Understat IDs associated with six FPL-code rows. Two
shared IDs are the audited mistakes; the remaining pair is two FPL spellings of Kaine
Kesler-Hayden. See [`data/mappings/README.md`](../data/mappings/README.md) for the audit,
confidence levels, current-roster provenance, and reverse-join rules. `understat_id` is
an identity key, not a model feature.

### Deadline anchoring

The frame is **already anchored** — every fixture in a gameweek carries the history that
existed at that gameweek's deadline. `data.assert_deadline_anchored` enforces it and is
called on every load.

If you rebuild the frame from other sources, this is the property to get right. Rolling
features shifted by kickoff let the second fixture of a double gameweek see the first
fixture's result, which nobody had when the squad was locked. On this data, **6,958
player-gameweeks have more than one fixture and 40.6% of them carried different rolling
values across those fixtures** before the fix.

### Columns present but deliberately unused

`starts`, `defensive_contribution`, `cbit`, `recoveries`, `tackles` describe **the
gameweek being predicted**. They are 100% zero whenever the player recorded zero minutes
and correlate 0.60–0.90 with minutes. They are in the frame so the rolling versions can
be rebuilt; **using them directly is leakage.**

---

## 2. The pre-deadline artefacts

These are the reason Dastan can use FPL's own projection and availability flags without
leaking.

Both are derived from [Randdalf/fplcache](https://github.com/Randdalf/fplcache)
(Unlicense — public domain), a daily archive of the FPL bootstrap endpoint.

### The acceptance test

A snapshot is used for a gameweek **only if all three hold**:

1. it names that gameweek as `is_next`,
2. its recorded deadline for that gameweek matches ours,
3. it was captured **strictly before** that deadline.

Anything else is discarded. There is no backfill and no interpolation. This is the whole
reason these columns are features rather than answers.

Resulting capture ages:

| artefact | rows | median age before deadline | max | after deadline |
|---|---|---|---|---|
| `pre_deadline_ep_next` | 142,173 | 4.4 h | — | **0** |
| `pre_deadline_signals` | 137,982 | 4.3 h | 6.2 h | **0** |

### `pre_deadline_ep_next.parquet`

FPL's own published projection for the coming gameweek — the number shown in the app.
It is both a feature and the most meaningful external benchmark.

Coverage is 2020-21 partially and ~99.4% for later seasons.

### `pre_deadline_signals.parquet`

Eight fields from the same snapshots. The published artifact starts in 2021-22; the six
2020-21 gameweeks available in the archive are deliberately excluded from this release.

| column | meaning | informative on |
|---|---|---|
| `sig_status_risk` | FPL status as an ordinal: available 0 → unavailable 4 | 32.3% |
| `sig_chance_playing` | `chance_of_playing_next_round` | 63.0% |
| `sig_has_news` | whether news text is attached | 32.2% |
| `sig_pens_order` | declared penalty-taker order | 7.6% |
| `sig_fk_order` | declared free-kick order | 8.8% |
| `sig_corners_order` | declared corner order | 10.7% |
| `sig_age_years` | age at the deadline | 27.2% |
| `sig_days_at_club` | days since joining | 31.4% |

Only the first three are in the model. The rest were measured and rejected — see
[FEATURES.md](FEATURES.md).

`sig_has_news` is **99.97% identical** to `sig_status_risk > 0` (38 differing rows in
137,982). It is retained because it was part of the family that was measured, but it
carries essentially no independent information.

### Missing is `-1`, never `0`

54,430 rows carry a genuine `ep_next` of exactly zero — FPL expecting nothing from a
player. Filling absent snapshots with `0` would make those indistinguishable, teaching
the model that "we have no data" and "FPL expects nothing" are the same event.

`data.load()` fills every joined column with `-1`. If you add a column, do the same.

---

## 3. `openfpl_predictions.csv`

OpenFPL's predictions for 2025-26 GW1-24, produced by running the published OpenFPL
model under its published regime (train 2020-21..2023-24, early stop 2024-25). Used by
`dastan.benchmark_openfpl` so both models are scored on identical rows.

`openfpl_row_keys.csv` maps `(element, gameweek)` to
`(season, gameweek, fixture, fpl_code)`, because of the ID-renumbering issue above.

**The `ep_next` column in `openfpl_predictions.csv` is not provenance-checked** and some
of it was captured after the deadline it describes. Use
`pre_deadline_ep_next.parquet` for any honest FPL comparison. This is not hypothetical:
the unchecked version scores an NDCG@10 of 0.4062 against the checked version's 0.2939.

---

## 4. Rebuilding

The raw-source builders are included under `dastan/rebuild`. A full reconstruction uses
pinned Vaastav and fplcache commits plus cached Understat fallbacks for incomplete
histories:

```bash
python -m pip install -r requirements-data.txt
python -m dastan.datasets verify
python -m dastan.datasets all
python -m dastan.datasets compare .cache/rebuilt-data
```

`verify` is the exact offline release check. `all` reconstructs a candidate without
overwriting `data/`. Understat can apply corrections after the release, so the comparison
reports key/schema agreement, numeric drift, semantic equality, and byte equality
separately. Source and output manifests contain SHA-256 hashes for every run.

Commands, source pins, the 304-column pipeline, cache behavior, measured smoke-test
counts, and exactness boundaries are in
[`REBUILDING_DATA.md`](REBUILDING_DATA.md).

---

## 5. Licensing

| component | licence |
|---|---|
| this repository (code, weights, derived data) | MIT |
| OpenFPL feature engineering | MIT |
| Vaastav FPL archive code/repository | MIT |
| fplcache snapshot archive | Unlicense (public domain) |
| understatAPI client | MIT |
| FPL and Understat source data | see their respective terms |

Not affiliated with the Premier League or Fantasy Premier League. The derived data here
is aggregated and transformed, not a redistribution of any provider's raw feed.
