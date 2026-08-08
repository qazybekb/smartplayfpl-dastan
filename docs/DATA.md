# Data

Everything needed to retrain Dastan is in this repository. No API keys, no scraping, no
database.

| file | rows | size |
|---|---|---|
| `data/features.parquet` | 163,072 | 37 MB |
| `data/pre_deadline_ep_next.parquet` | 137,610 | 362 KB |
| `data/pre_deadline_signals.parquet` | 137,982 | 438 KB |
| `data/openfpl_predictions.csv` | 18,173 | 1.3 MB |
| `data/openfpl_row_keys.csv` | 18,173 | 444 KB |
| `data/players.csv` | 4,717 | 130 KB |
| `data/mappings/fpl_understat_players.csv` | 1,564 | 84 KB |

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

`data/mappings/fpl_understat_players.csv` publishes the exact `fpl_code` to
`understat_id` joins used by the training frame. It covers 1,564 players, representing
90.8% of player-fixture rows. Load it with `dastan.mappings.load()` or verify that it
still matches the frame with `python -m dastan.mappings`. The ID is a join key, not a
model feature.

This is a snapshot of the model's historical joins, not a claim that every identity is
globally one-to-one. Three Understat IDs are associated with multiple historical FPL
codes; all six affected rows are flagged `shared_understat_id`. See
[`data/mappings/README.md`](../data/mappings/README.md) before reverse-joining from
Understat to FPL.

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
| `pre_deadline_ep_next` | 137,610 | 4.4 h | — | **0** |
| `pre_deadline_signals` | 137,982 | 4.3 h | 6.2 h | **0** |

### `pre_deadline_ep_next.parquet`

FPL's own published projection for the coming gameweek — the number shown in the app.
It is both a feature and the most meaningful external benchmark.

Coverage is 2020-21 partially and ~99.4% for later seasons.

### `pre_deadline_signals.parquet`

Eight fields from the same snapshots. Seasons 2021-22 onward (the archive does not reach
2020-21).

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

The builders that produced the pre-deadline artefacts are not in this repo, because they
depend on a local mirror of the fplcache archive. The logic is short and fully described
by §2 above — the acceptance test is the entire algorithm.

To extend to a new season:

1. Pull the fplcache snapshots for that season.
2. For each gameweek, find snapshots naming it `is_next`, agreeing on the deadline, and
   predating it. Take the latest such snapshot.
3. Extract the fields in §2, keyed by `(season, gameweek, fpl_code)`.
4. Assert that no accepted snapshot is post-deadline. **Do not relax this.** If a season
   resolves few gameweeks, drop those columns for that season rather than weakening the
   test.
5. After extending `features.parquet`, regenerate the identity snapshot with
   `python -m dastan.mappings --write` and review every `shared_understat_id` row.

---

## 5. Licensing

| component | licence |
|---|---|
| this repository (code, weights, derived data) | MIT |
| OpenFPL feature engineering | MIT |
| fplcache snapshot archive | Unlicense (public domain) |
| FPL and Understat source data | see their respective terms |

Not affiliated with the Premier League or Fantasy Premier League. The derived data here
is aggregated and transformed, not a redistribution of any provider's raw feed.
