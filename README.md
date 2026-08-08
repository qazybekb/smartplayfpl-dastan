# Dastan

**An open-source expected-points model for Fantasy Premier League.**
Six seasons, 163,072 player-fixtures, 286 features, weights included.

Named after [Dastan Satpayev](https://en.wikipedia.org/wiki/Dastan_Satpayev), the first
Kazakh footballer signed by an English Premier League club.

Published by [SmartPlayFPL](https://smartplayfpl.com). MIT licensed — the code, the
weights, and the training data.

---

## Why this repo exists

Most public FPL models publish a notebook and a leaderboard number. The number is
usually not reproducible, and often not meaningful — pooled across a season, scored on
a restricted pool, or quietly leaking the deadline.

This repository publishes the opposite: **the full training data, the trained weights,
the evaluation harness, and the things that did not work.** Including a 500-trial
hyperparameter search that produced nothing, and a feature family currently shipped on
no evidence at all.

The most valuable thing here is probably not the model. It is
[METHODOLOGY.md](docs/METHODOLOGY.md) — specifically the measurement that the
seed-to-seed noise of this objective is **0.0106**, which is larger than almost every
feature effect anyone will ever report on FPL data.

---

## Quick start

```bash
git clone https://github.com/qazybekb/smartplayfpl-dastan
cd smartplayfpl-dastan
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m dastan.verify        # reload the released weights and score them
```

The dependency versions are pinned because XGBoost minor versions can fit different
trees from the same data and seed. CPU architectures can also produce numerically
different, quality-equivalent trees. The published artefacts and reproduction check
use Python 3.12 and XGBoost 3.2.0.

Predicting:

```python
import pandas as pd
from dastan import data, predictor

df = data.load()                              # 163,072 rows, all six seasons
model = predictor.Dastan()                    # the released weights
out = model.predict_frame(df, with_parts=True)

# Names are a separate lookup: the frame is keyed by fpl_code, which is stable across
# seasons. FPL's `element` id is not -- 1,130 of 1,959 players had theirs change.
players = pd.read_csv("data/players.csv")
out = out.merge(players[["season", "fpl_code", "player_name"]],
                on=["season", "fpl_code"], how="left")

out.nlargest(10, "xpts")[["player_name", "team_name", "gameweek", "xpts", "p60"]]
```

`with_parts=True` also returns `p60` (probability of 60+ minutes) and the four scoring
band probabilities — which say something `xpts` alone cannot: whether a 5.0 projection
is a steady five or a coin-flip between two and thirteen.

---

## Results

Walk-forward: the model is retrained from scratch for each block, so nothing in a test
window existed when its model was fitted. Three seeds. Metrics computed **within each
gameweek, then averaged** — never pooled across a season, which inflates rank
correlation badly.

Objective is `0.5 × Spearman + 0.5 × NDCG@10`.

### All players

| model | objective | vs Dastan |
|---|---|---|
| **Dastan** | **0.6072** | — |
| FPL's own `ep_next` | 0.5321 | **−0.0751** |
| rolling-5 form | 0.5260 | −0.0812 |
| last gameweek's points | 0.4950 | −0.1122 |
| price | 0.3571 | −0.2501 |

### Starters only (60+ minutes)

| model | objective | vs Dastan |
|---|---|---|
| **Dastan** | **0.3959** | — |
| price | 0.3584 | −0.0375 |
| FPL's own `ep_next` | 0.3139 | −0.0820 |
| rolling-5 form | 0.2820 | −0.1139 |
| last gameweek's points | 0.2404 | −0.1555 |

**The minutes head is the strongest part of the model: AUC ≈ 0.95 out of sample.**

Full tables, per-block detail, calibration and practical metrics:
[docs/ACCURACY.md](docs/ACCURACY.md).

### What we will not claim

Against a core feature set without the added families, Dastan gains **+0.0128 on all
players** and **+0.0010 on starters**. The second number is below the noise floor: among
players who actually start, the extra feature engineering here is **worth nothing**.

Everything the added features buy is in knowing *who will not play*. If your problem is
"rank the eleven I already know are starting", the value in this repo is the
architecture and the protocol, not the features.

Also worth knowing: **price alone scores 0.3584 on starters.** The market is good. Check
any model you build against it.

---

## Against OpenFPL

Dastan's feature engineering is **OpenFPL's**, used as code rather than reimplemented:

> Groos, D. & Zhang, S. *OpenFPL: An open-source forecasting method rivalling
> state-of-the-art Fantasy Premier League services.*
> [arXiv:2508.09992](https://arxiv.org/abs/2508.09992) ·
> [github.com/daniegr/OpenFPL](https://github.com/daniegr/OpenFPL) · MIT

| layer | origin |
|---|---|
| Player rolling features (115) | **OpenFPL**, exact |
| Team rolling features (60) | **OpenFPL**, exact |
| Opponent rolling features (60) | OpenFPL, extended by 2 stats |
| Availability, context, defensive-contribution, pre-deadline signals (51) | Dastan |
| Model | Dastan |

What is **not** taken is OpenFPL's estimator — roughly 250 genetically-searched RF/XGB
models aggregated by median, with MinMax feature scaling and a target scaler. Dastan
replaces it with a conditional decomposition (below). Dropping the scalers is safe:
tree splits are invariant to monotone rescaling.

### Head to head

Both models scored on **identical rows** under the regime OpenFPL published for: train
2020-21..2023-24, early stop 2024-25, test 2025-26 GW1-24, 18,173 player-fixtures. Both
train on the same four seasons here, so Dastan's usual six-season advantage is **not** a
factor in these numbers.

| cohort | Dastan | OpenFPL | delta |
|---|---|---|---|
| **all players** | **0.5602** | 0.5044 | **+0.0558** |
| starters only | 0.2589 | 0.2564 | +0.0025 |

| metric (all players) | Dastan | OpenFPL |
|---|---|---|
| Spearman | 0.7535 | 0.6961 |
| NDCG@10 | 0.3669 | 0.3128 |
| MAE | **0.950** | 1.167 |
| RMSE | **1.929** | 2.017 |

**On the full pool Dastan is clearly ahead** — +0.0558 is nine times the noise floor,
and MAE improves by 0.22 points per player-gameweek.

**On starters the two are indistinguishable.** +0.0025 is below the noise floor, and
OpenFPL is in fact marginally *better* on Spearman (0.1334 vs 0.1244) and MAE (2.319 vs
2.341) within that cohort. Dastan's advantage is in knowing who will not play, not in
ranking the players who do — the same conclusion the ablation reaches, from a completely
independent direction.

These numbers are lower than the walk-forward figures above because this regime trains
on four seasons instead of five and tests on GW1-24, which includes the hard early-season
gameweeks where little form has accumulated.

Reproduce:

```bash
python -m dastan.benchmark_openfpl     # writes docs/openfpl_benchmark.json
```

---

## The model

A single regressor trained on FPL points spends its capacity learning that most players
score nothing — about 62% of player-gameweeks here are zero minutes. So the target is
decomposed instead:

```
xPts = p60 · Σ P(band k | started) · E[points | band k]
     + (1 − p60) · E[points | did not start]
```

| head | trained on | predicts |
|---|---|---|
| `p60` | everyone | P(60+ minutes) |
| `non60` | rows under 60 minutes | expected points given a cameo or absence |
| `bucket` | rows 60+ minutes | P(scoring band), split at 1, 3, 10 points |
| `bucketreg` ×4 | each band | expected points within the band |
| `direct` | everyone | ordinary regression, blended in per position |

**Each head early-stops on the population it models**, not on the full validation set.
That sounds pedantic and is not — early-stopping the "given under 60 minutes" head on a
validation set dominated by starters picks the wrong tree count for the job it has.
Fixing this class of defect was worth more than any feature in this repo.

Positions are fitted independently. Band probabilities are calibrated per fold, per
position — never loaded from elsewhere, which would be model-selection leakage.

---

## Data

| file | rows | what |
|---|---|---|
| `data/features.parquet` | 163,072 | the training frame, six seasons, deadline-anchored |
| `data/pre_deadline_ep_next.parquet` | 137,610 | FPL's own projection, provenance-checked |
| `data/pre_deadline_signals.parquet` | 137,982 | status, chance of playing, set-piece order, age |
| `data/openfpl_predictions.csv` | 18,173 | OpenFPL's predictions, for the head-to-head |
| `data/players.csv` | 4,717 | `fpl_code` → name and position, per season |

Sources and the provenance rules: [docs/DATA.md](docs/DATA.md).

**The provenance rule, because it is the whole game.** A snapshot is used only if it
names the gameweek as `is_next`, agrees on its deadline, **and was captured before it**.
Median capture is 4.3 hours pre-deadline; zero rows come from after. Without that check
these columns are the answer key rather than a feature.

**Missing is `-1`, never `0`.** 54,430 rows carry a genuine `ep_next` of exactly zero.
Collapsing "no snapshot" into "FPL expects nothing" teaches the model that two very
different events are the same.

---

## Reproducing

```bash
python -m dastan.verify                # reload and score the published weights
python -m dastan.reproduce             # retrain and require equivalent holdout quality
python -m dastan.reproduce --strict    # require identical artefacts on the release platform
python -m dastan.train --out experiments/my-model
python -m dastan.evaluate --seeds 3    # walk-forward vs baselines (~90 min)
python -m dastan.benchmark_openfpl     # head-to-head on identical rows
```

`models/` contains 38 published files: 35 are required for inference; the remaining
three record training hyperparameters, the core feature manifest, and release metadata.

Reproduction starts from the published, deadline-anchored feature frame. It does not
rebuild that frame from the providers' raw archives. Extending the data to a new season
requires reconstructing the provenance-checked inputs described in
[`docs/DATA.md`](docs/DATA.md#4-rebuilding).

The default reproduction threshold is the documented 0.0061 objective keep margin for
both evaluation cohorts. Exact model bytes and prediction deltas are always reported;
`--strict` makes those exact comparisons mandatory.

`notebook/dastan.ipynb` walks through the whole thing with explanations.

---

## What we tested and rejected

[docs/FEATURES.md](docs/FEATURES.md) lists eleven candidate families that were built,
measured and thrown away, with numbers. Highlights:

- **Declared set-piece taker order: +0.0003.** Sounds strictly better than inferred
  penalty share. Five windows of scoring history has already learned who takes them.
- **Empty-stadium crowd regime: −0.0018.** Home advantage genuinely *inverts* behind
  closed doors (−0.05 vs +0.41). Rolling team form already encodes it. A real mechanism
  is not automatically a useful feature.
- **Kickoff time: −0.0057.** Actively harmful.
- **500 Optuna trials: −0.0015.** Claimed +0.0161 on its tuning window. Two separate
  selection biases, and only one of them is intuitive.

And one we are not proud of: **`F4_congestion` is in the shipped model on no evidence.**
It entered under a threshold 5.3× below the noise floor, and re-screening put its
removal at a coin flip. It is documented rather than quietly left looking justified.

---

## Limitations

- **Premier League fixtures only.** A player who went 90 minutes in the Europa League on
  Thursday looks fully rested. This is the most promising open lead in the repo.
- **Starters cohort is not improved** by the added features. See above.
- **Absolute values are conservative for starters** — the model under-predicts them by
  0.83 points on average. It ranks well; do not read the level as a point estimate.
- **Top-10 overlap is 0.183.** Of the ten highest scorers in a gameweek, about 1.8 were
  in our predicted ten. That is close to the realistic ceiling, and any FPL model
  claiming dramatically better is probably scoring pooled or leaking.
- **Not a squad optimiser.** This produces expected points. Turning those into
  transfers, captaincy and chip timing is a separate problem.

---

## Contributing

The most useful contributions, in order:

1. **All-competition minutes** — European and domestic cup fixtures, to make congestion
   real.
2. **Anything that improves the starters cohort.** Nothing we tried did.
3. **Re-test the families in FEATURES.md §"tested earlier"** — those were screened under
   a protocol now known to be unreliable, so they are unproven rather than disproven.

If you measure a feature effect, please report **seed count and noise floor** alongside
it. A single-seed "+0.005" on this objective is indistinguishable from having changed
nothing.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the experiment and pull-request workflow.

---

## Citation

```bibtex
@software{dastan2026,
  title  = {Dastan: an open-source expected-points model for Fantasy Premier League},
  author = {SmartPlayFPL},
  year   = {2026},
  url    = {https://github.com/qazybekb/smartplayfpl-dastan},
  note   = {Built on OpenFPL (arXiv:2508.09992)}
}
```

Please also cite OpenFPL, whose feature engineering this depends on.

## Licence

MIT, including the data and weights. OpenFPL is MIT. The FPL snapshot archive
([Randdalf/fplcache](https://github.com/Randdalf/fplcache)) is Unlicense.
Not affiliated with the Premier League or Fantasy Premier League.
