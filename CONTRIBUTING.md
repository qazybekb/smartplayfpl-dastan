# Contributing to Dastan

Dastan welcomes model, evaluation, documentation, and data-provenance improvements.
The standard is not whether an idea sounds useful; it is whether the result survives
the chronological, multi-seed protocol in this repository.

## Set up

Use Python 3.12 and the pinned environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m dastan.verify
python -m dastan.datasets verify
python -m dastan.reproduce
```

`verify` checks the published weights. `reproduce` retrains every model head in a
temporary directory and requires equivalent holdout quality within the documented
0.0106 single-seed noise. It also reports exact artefact and prediction differences.
Use `python -m dastan.reproduce --strict` on the release platform when changing
weights. The three-seed evidence standard below uses the tighter 0.0061 keep margin.

## Run an experiment

Do not overwrite `models/` while exploring. Write candidate artefacts under the
gitignored `experiments/` directory:

```bash
python -m dastan.train --out experiments/candidate
python -m dastan.evaluate --seeds 3 --out experiments/evaluation.json
```

Changes to the model architecture belong in `dastan/model.py`. Feature-family changes
should be expressed through `dastan/data.py` and documented with their provenance.

## Evidence required

- Score predictions within each gameweek at player-gameweek grain.
- Report both the all-player and 60-plus-minute starter cohorts.
- Use the three published seeds: 42, 7, and 2026.
- Compare the effect with the measured 0.0061 keep margin.
- Do not use a discovery or tuning window as confirmation evidence.
- Preserve deadline anchoring and encode unavailable snapshot data as `-1`, not `0`.

A useful negative result is welcome. Record what was tried, the exact window and seeds,
and why it should remain rejected.

## New data

The repository contains both the immutable derived frame and the optional public-source
builder. Read [`docs/REBUILDING_DATA.md`](docs/REBUILDING_DATA.md) before changing it.
New or updated data must document its source, join keys, capture time, deadline
acceptance rule, coverage, missing-value policy, and licence. Never admit information
captured after the FPL deadline for the gameweek being predicted.

## Pull requests

Include:

1. The hypothesis and mechanism.
2. The exact commands used.
3. Per-cohort, multi-seed results and the relevant baseline.
4. Runtime or model-size changes.
5. Any provenance or licensing implications.

Run `python -m dastan.verify`, `python -m dastan.datasets verify`, and
`python -m unittest discover -v` before opening the pull request. Changes to training
code, dependencies, features, or released weights must also pass
`python -m dastan.reproduce`. Changes to FPL/Understat identities must pass
`python -m dastan.mappings`.
