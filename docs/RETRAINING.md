# Retraining and releasing Dastan

This is the public release checklist for both a mid-season update and an annual
rollover. It complements the production SmartPlayFPL runbook: production source-data
and mapping synchronization happen first, then this repository receives the exact
reviewed data/model release.

Do not retrain from memory. Do not overwrite `data/` or `models/` while exploring.
Candidate data belongs under `.cache/`; candidate models belong under `experiments/`.

## Release modes

| release | data cutoff | training | fitting holdout | target |
|---|---:|---:|---:|---|
| mid-season 2026-27 | GW19 | through GW15 | GW16-19 | 2026-27 |
| end of 2026-27 | GW38 | through GW30 | GW31-38 | 2027-28 |

The plan is generated from `data/season_registry.json`; these windows are not spread
through Python constants. Check them at any time:

```bash
python -m dastan.release plan
python -m dastan.release plan --mode midseason --season 2026-27 --through-gw 19
python -m dastan.release plan --mode annual --season 2026-27
```

## One-time setup

Use the pinned Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-data.txt

python -m unittest discover -v
python -m dastan.datasets verify
python -m dastan.mappings
python -m dastan.artifacts
python -m dastan.verify
```

## Gate 1: mappings

The editable mapping source is maintained in SmartPlayFPL. This repository contains
two generated contracts:

- `data/mappings/current_*` is the latest operational mapping release used for new
  data;
- `fpl_understat_training_snapshot.csv` and
  `fpl_understat_training_assignments.csv` remain frozen with the published frame.

Never hand-edit a generated current file and never copy the frozen mapping back over
the operational release. After the canonical mapping release is accepted and synced
to production, export it here and run:

```bash
python -m dastan.mappings --refresh-current
python -m dastan.mappings
```

New-season data preparation fails closed until every current club has an Understat
identity. It is normal for promoted clubs to be `awaiting_understat` before Understat
publishes the season, but it is not permissible to train through that state. As of
this release Coventry City and Hull City are the two expected pending clubs.

Frozen assignments are used for all released seasons. For the immediate active
season, `dastan.mappings.assignments_for_seasons()` derives assignments from stable
FPL codes in the accepted operational release; no name-based fallback is used.

## Gate 2: source pins

`data/source_pins.json` is the only place that identifies the Vaastav and fplcache
commits used by the public builder. Before adding a season:

1. select commits that contain every completed gameweek through the cutoff;
2. review both commit hashes rather than using a moving branch;
3. update `data/source_pins.json` in the release PR;
4. retain the generated `downloads.json`, which fingerprints every downloaded byte.

Do not use `--allow-missing-understat` for a release candidate.

## Mid-season release

Start only after GW19 is finished and the production mapping/source pipeline has
completed. Build all released seasons plus the active season into an isolated
directory:

```bash
python -m dastan.datasets all \
  --seasons 2020-21,2021-22,2022-23,2023-24,2024-25,2025-26,2026-27 \
  --raw-dir .cache/dastan-raw-2026-27-gw19 \
  --output-dir .cache/releases/midseason-2026-27-gw19
```

Prepare a fingerprinted candidate. Preparation writes a manifest but neither accepts
the data nor trains a model:

```bash
python -m dastan.release prepare \
  --mode midseason --season 2026-27 --through-gw 19 \
  --candidate-dir .cache/releases/midseason-2026-27-gw19
```

Review `dastan_release_candidate.json`, including per-season row/GW/fixture coverage,
the four-position holdout counts, snapshot coverage, mapping release, source pins, and
every SHA256. Then preview and explicitly accept those exact bytes:

```bash
python -m dastan.release accept \
  --manifest .cache/releases/midseason-2026-27-gw19/dastan_release_candidate.json

python -m dastan.release accept \
  --manifest .cache/releases/midseason-2026-27-gw19/dastan_release_candidate.json \
  --apply
```

Train into an isolated model directory:

```bash
python -m dastan.train \
  --mode midseason --season 2026-27 --through-gw 19 \
  --data-dir .cache/releases/midseason-2026-27-gw19 \
  --accepted-manifest .cache/releases/midseason-2026-27-gw19/dastan_release_accepted.json \
  --out experiments/midseason-2026-27-gw19 \
  --n-jobs 8
```

If any accepted data, mapping, source pin, feature contract, hyperparameter, or tuning
record changes after acceptance, training refuses to start.

When publishing the mid-season frame, preview and apply its registry update:

```bash
python -m dastan.release register \
  --manifest .cache/releases/midseason-2026-27-gw19/dastan_release_accepted.json \
  --data-dir .cache/releases/midseason-2026-27-gw19
python -m dastan.release register \
  --manifest .cache/releases/midseason-2026-27-gw19/dastan_release_accepted.json \
  --data-dir .cache/releases/midseason-2026-27-gw19 --apply
```

The command appends `2026-27` to `release_seasons`, updates its coverage/row guard,
and records the accepted GW19 plan. It does **not** add it to `completed_seasons` or
advance `latest_completed_season`, so reproduction uses the exact partial-season plan
without pretending the season is complete.

After registration, promote the accepted data bytes into the release directory and
regenerate the mappings derived from that exact frame. Do this as one release change;
do not commit the temporarily inconsistent state between commands:

```bash
cp .cache/releases/midseason-2026-27-gw19/features.parquet data/
cp .cache/releases/midseason-2026-27-gw19/pre_deadline_ep_next.parquet data/
cp .cache/releases/midseason-2026-27-gw19/pre_deadline_signals.parquet data/
cp .cache/releases/midseason-2026-27-gw19/players.csv data/
python -m dastan.mappings --write
python -m dastan.datasets verify --write-manifest
```

`mappings --write` freezes the exact player snapshot and season/gameweek assignment
intervals present in the newly published frame. The operational `current_*` mapping
release remains separate and continues to drive later active-season additions.

## Annual rollover

Build the just-completed season before the official FPL API rolls to the next roster:

```bash
python -m dastan.datasets all \
  --seasons 2020-21,2021-22,2022-23,2023-24,2024-25,2025-26,2026-27 \
  --raw-dir .cache/dastan-raw-2026-27-final \
  --output-dir .cache/releases/annual-2026-27

python -m dastan.release prepare \
  --mode annual --season 2026-27 \
  --candidate-dir .cache/releases/annual-2026-27

python -m dastan.release accept \
  --manifest .cache/releases/annual-2026-27/dastan_release_candidate.json
python -m dastan.release accept \
  --manifest .cache/releases/annual-2026-27/dastan_release_candidate.json --apply

python -m dastan.train \
  --mode annual --season 2026-27 \
  --data-dir .cache/releases/annual-2026-27 \
  --accepted-manifest .cache/releases/annual-2026-27/dastan_release_accepted.json \
  --out experiments/annual-2026-27 \
  --n-jobs 8
```

When the reviewed data is published, preview and apply `python -m dastan.release
register` with the annual accepted manifest and data directory, exactly as above.
For an annual plan it moves `2026-27` into `completed_seasons`, advances
`latest_completed_season`, and records the accepted annual plan. Commit the registry
with the new immutable data manifest. Copy the four accepted data files and regenerate
the frozen mappings/release manifest with the same commands shown in the mid-season
section, substituting `.cache/releases/annual-2026-27`. Archived coverage is preserved
and only the immediate successor can be registered.

## Candidate verification and promotion

Reload the candidate artifacts against the candidate holdout:

```bash
python -m dastan.verify \
  --model-dir experiments/midseason-2026-27-gw19 \
  --data-dir .cache/releases/midseason-2026-27-gw19 \
  --season 2026-27 --gw 16 19
```

Review both the all-player and starter cohorts. Run the stable multi-seed evaluation
before promotion; do not tune on a release holdout after inspecting it. The documented
three-seed keep margin is 0.0061.

Review every `paired_baselines` result too. FPL `ep_next`, previous-five form, and
last-match points are only valid when Dastan is rescored on the exact same eligible
player-gameweek rows; price is ranking-only. Missing forecasts must be excluded from
both sides, never imputed as zero. Do not import FPL Review or any other published
score from a different season, horizon, or row set and present it as a head-to-head.

Before promotion, regenerate the checked evidence when its inputs changed:

```bash
python -m dastan.evaluate --scope clean --seeds 3
python -m dastan.benchmark_openfpl --seeds 3
```

The OpenFPL benchmark is required when the recipe, feature contract, training data,
or benchmark protocol changes. A weights-only release under an unchanged recipe may
retain the existing controlled report, but reviewers must confirm its provenance and
limitations. Commit `docs/evaluation.json`, `docs/openfpl_benchmark.json`, and the
matching prose together; the documentation tests reject stale headline numbers.

Public promotion is an atomic, explicit operation. Supply the exact SmartPlayFPL
production commit that promoted the same model:

```bash
python -m dastan.release promote \
  --source experiments/midseason-2026-27-gw19 \
  --production-commit FULL_40_CHARACTER_COMMIT

python -m dastan.release promote \
  --source experiments/midseason-2026-27-gw19 \
  --production-commit FULL_40_CHARACTER_COMMIT \
  --apply
```

Promotion validates all 40 public files, stages a complete replacement, swaps the
directory atomically, and writes `models/artifact_manifest.json`. The public contract
is a safe superset of production: the 38-file production contract plus two public-only files
needed to retrain (`hyperparameters.json` and `core_feature_cols.json`).

## Final release checks

Run all of these from a clean checkout:

```bash
python -m unittest discover -v
python -m dastan.datasets verify
python -m dastan.mappings
python -m dastan.artifacts
python -m dastan.verify
python -m dastan.reproduce --n-jobs 2
```

The release PR must contain together:

- new immutable data and `data/release_manifest.json`, if the training frame changed;
- the advanced season registry, if a season completed;
- the exact operational mapping projection and manifest;
- all 40 model-contract files and their artifact manifest;
- updated accuracy/evaluation evidence and documentation;
- the full production commit linking public artifacts to what SmartPlayFPL serves.

CI repeats the offline tests, dataset/mapping/model hash checks, coherent-minutes
checks, artifact reload, and full retraining reproduction. Merge only when all pass.

## Recovery rules

- A failed preparation changes no released data or model.
- A changed candidate must be prepared and accepted again; never edit an accepted
  manifest.
- A failed model promotion restores the prior `models/` directory.
- Old seasons are append-only. Provider corrections are reported as drift, not
  silently folded into the immutable release.
- A missing promoted-club mapping is a hard stop, not permission to guess an ID.
- The public frozen mapping changes only with a new model/data release; current
  operational mappings may update independently between model releases.
