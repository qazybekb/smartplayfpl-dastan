# FPL to Understat identities

Identity is versioned here because "the IDs used by the released model" and "the IDs
recommended for a new join" are not the same artifact.

## Files

| file | purpose |
|---|---|
| `fpl_understat_training_snapshot.csv` | immutable 1,564-player identity snapshot present in `features.parquet` |
| `fpl_understat_training_assignments.csv` | exact season/gameweek intervals in which those IDs were attached to training rows |
| `fpl_understat_identity_audit.csv` | evidence and decisions for three manually checked identities |
| `fpl_understat_players.csv` | corrected historical map recommended for new joins |
| `fpl_players_current.csv` | captured 2026-27 FPL roster, including unresolved players |
| `fpl_understat_current_supplements.csv` | reviewed current-only identities not present in training history |
| `fpl_understat_current.csv` | complete current roster plus mapping status, confidence, and source |
| `fpl_understat_current.json` | roster capture time, source URL, and bootstrap SHA-256 |

## Which map to use

Use `dastan.mappings.load()` for new data. It returns the corrected historical map.
Two release identities are replaced by evidence-backed Understat IDs:

| FPL code | player | release ID | corrected ID |
|---:|---|---:|---:|
| 437688 | Lewis Richards | 9082 | 9216 |
| 515501 | Álvaro Fernández Carreras | 5191 | 10576 |

The audit also records why FPL code 431248 remains mapped to Understat 12721. Evidence
URLs and review notes are in `fpl_understat_identity_audit.csv`.

Use `dastan.mappings.load_training()` only to reproduce the published model. It retains
the two mistakes because silently correcting them would change historical player
grouping. Use `load_training_assignments()` when rebuilding raw data; the release did
not attach every known identity to every historical row. The assignment table captures
that timeline exactly.

Use `dastan.mappings.load_current()` for the captured 2026-27 roster. It maps 517 of
573 players. The remaining 56 rows are explicit `unmapped` records; absence is safer
than a name-based guess. Of the 29 current-only supplements, 4 are `HIGH` confidence
and 25 are `MEDIUM` confidence.

## Coverage and reverse joins

The immutable training snapshot contains 1,564 FPL codes and 1,561 unique Understat
IDs. Three IDs are shared across six historical FPL-code rows, including the two errors
above. The corrected historical map has 1,563 unique Understat IDs; its remaining shared
ID represents two FPL spellings of Kaine Kesler-Hayden.

Join from FPL to Understat on `fpl_code`. A reverse join is not globally one-to-one;
handle `mapping_status == "shared_understat_id"` explicitly.

`understat_id` is an identity key, not one of the model's 286 features. Unmapped rows
remain null and follow the normal missing-data policy.

## Verify and refresh

```bash
python -m dastan.mappings
python -m dastan.mappings --write
python -m dastan.mappings --refresh-current
python -m dastan.mappings --refresh-current --bootstrap bootstrap-static.json
```

`--write` regenerates derived maps from checked-in inputs. `--refresh-current` first
captures a new official FPL roster; review unresolved and supplemental identities before
committing the result.
