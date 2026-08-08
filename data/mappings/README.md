# FPL to Understat mapping

`fpl_understat_players.csv` is the player identity crosswalk used by the published
Dastan training frame. It maps FPL's cross-season `code` field to Understat's numeric
player ID for 1,564 players across 2020-21 through 2025-26.

This is a release snapshot, not a live identity service. It is derived from
`data/features.parquet`, with display names and season coverage added from
`data/players.csv`. That makes the artifact reproducible entirely from files in this
repository and records the joins the model actually used. `understat_id` is an identity
join key, not one of the 286 model features.

## Coverage

- 1,564 of the 1,960 FPL codes in the frame are mapped (79.8%).
- Mapped players account for 148,118 of 163,072 player-fixture rows (90.8%).
- Three Understat IDs are shared by multiple historical FPL codes. Their six rows are
  marked `shared_understat_id`; do not assume the reverse Understat-to-FPL join is
  one-to-one.

Unmapped players are not included in the CSV. Their `understat_id` remains null in the
training frame, and their Understat-derived rolling features follow the frame's normal
missing-data policy.

## Columns

| column | meaning |
|---|---|
| `fpl_code` | FPL's cross-season player code; use this, not the season-local `element` ID |
| `understat_id` | numeric Understat player ID |
| `player_name` | latest FPL display name in this release |
| `position` | latest FPL position in this release |
| `first_season` | first season for this FPL code in the published frame |
| `last_season` | last season for this FPL code in the published frame |
| `mapping_status` | `mapped` or `shared_understat_id` |

## Usage

```python
import pandas as pd

mapping = pd.read_csv("data/mappings/fpl_understat_players.csv")
external_fpl_rows = external_fpl_rows.merge(
    mapping[["fpl_code", "understat_id"]],
    on="fpl_code",
    how="left",
    validate="many_to_one",
)
```

The training frame already contains `understat_id`; this CSV is for joining other FPL
data or auditing identities. Join from FPL to Understat. If you need to join in the
opposite direction, handle rows marked `shared_understat_id` explicitly.

## Rebuild and verify

```bash
python -m dastan.mappings --write  # regenerate from the published frame
python -m dastan.mappings          # fail if the checked-in snapshot has drifted
```
