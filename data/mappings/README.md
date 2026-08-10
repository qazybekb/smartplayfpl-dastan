# FPL to Understat identities

The repository deliberately separates three identity contracts:

1. the immutable IDs that reproduced the released Dastan model;
2. the audited correction of that historical model population; and
3. the latest versioned SmartPlayFPL operational registry used for new data.

Conflating those contracts would either make Dastan irreproducible or let an old
training snapshot overwrite newer production identities.

## Files

| file | ownership and purpose |
|---|---|
| `fpl_understat_training_snapshot.csv` | immutable 1,564-player snapshot present in `features.parquet` |
| `fpl_understat_training_assignments.csv` | exact season/gameweek intervals in which those IDs were attached to training rows |
| `fpl_understat_identity_audit.csv` | public evidence for three manually checked historical identities |
| `fpl_understat_players.csv` | audited historical Dastan population recommended when reproducing/rebuilding model-era data |
| `current_fpl_understat_players.csv` | generated projection of the latest full operational player registry |
| `current_fpl_understat_clubs.csv` | generated projection of the latest full operational club registry |
| `current_manifest.json` | canonical release ID, SHA256, byte count, and coverage for both operational projections |
| `fpl_players_current.csv` | captured current-season FPL roster, including unresolved players |
| `fpl_understat_current.csv` | current roster joined to the versioned operational player registry |
| `fpl_understat_current.json` | roster capture time, source URL, and bootstrap SHA256 |

The three `current_*` release files are generated from SmartPlayFPL's reviewed
canonical registry. Do not hand-edit them in this repository. Corrections are reviewed
at the canonical source, assigned a new manifest, synchronized transactionally to
production, and then exported here.

## Which map to use

Use `dastan.mappings.load_training()` only to reproduce the published model. It
retains two historical mistakes because silently correcting them would change the
released player grouping. Use `load_training_assignments()` when rebuilding raw data;
the release did not attach every known identity to every historical row.

Use `dastan.mappings.load()` for corrected model-era data. Two release identities are
replaced by evidence-backed Understat IDs:

| FPL code | player | release ID | corrected ID |
|---:|---|---:|---:|
| 437688 | Lewis Richards | 9082 | 9216 |
| 515501 | Álvaro Fernández Carreras | 5191 | 10576 |

The audit also records why FPL code 431248 remains mapped to Understat 12721. Evidence
URLs and review notes are in `fpl_understat_identity_audit.csv`.

Use `dastan.mappings.load_operational_players()` for the latest complete registry and
`load_current()` for the captured 2026-27 roster. The roster currently maps 518 of
577 players; the other 59 are explicit `unmapped`/`NONE` records. Absence is safer
than a name-based guess.

For a newly active season, `assignments_for_seasons()` combines immutable assignments
for released seasons with stable-code identities from the accepted operational
release. It filters the operational registry to the captured FPL roster and refuses
to proceed until all current clubs have Understat identities. This is the only public
data-builder path for adding a season; do not extend the frozen assignment CSV by hand.

## Coverage and reverse joins

The immutable training snapshot contains 1,564 FPL codes and 1,561 unique Understat
IDs. Three IDs are shared across six historical FPL-code rows, including the two
release errors above. The corrected historical map has 1,563 unique Understat IDs;
its remaining shared ID represents two FPL spellings of Kaine Kesler-Hayden.

The operational release retains historical registry rows so old seasons remain
rebuildable. Filter/join its current-season projection via `fpl_understat_current.csv`.
Join from FPL to Understat on `fpl_code`; reverse joins are not globally one-to-one,
so handle `mapping_status == "shared_understat_id"` explicitly.

`understat_id` is an identity key, not one of the model's 286 features. Unmapped rows
remain null and follow the normal missing-data policy.

## Verify and refresh

```bash
# Verify frozen, audited, operational-manifest, and current-roster artifacts.
python -m dastan.mappings

# Rebuild all derived maps from their checked-in inputs.
python -m dastan.mappings --write

# Capture a fresh official FPL roster and join the accepted operational release.
python -m dastan.mappings --refresh-current
python -m dastan.mappings --refresh-current --bootstrap bootstrap-static.json
```

Verification recalculates the immutable model artifacts, checks both operational file
hashes against `current_manifest.json`, validates coverage and mapping states, and
rebuilds the current roster in memory. Any post-release edit fails closed.
