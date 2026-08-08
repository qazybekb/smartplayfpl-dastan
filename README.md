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

python -m dastan.artifacts     # verify public/production artifact hashes
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

`with_parts=True` also returns `p60`, coherent `p_any` and `expected_minutes`, and the
four scoring-band probabilities. The three minutes values come from one calibrated
curve and are constrained so `p_any >= p60`, `expected_minutes >= 60*p60`, and
`p_any >= expected_minutes/90`.

---

## Accuracy

The headline evaluation contains **24 clean gameweeks, 17,986 player-fixtures, and
17,622 player-gameweeks** from 2024-25 GW15-38. Each block is walk-forward: Dastan is
retrained without its test window, using the exact current 286-feature contract and
three seeds whose predictions are averaged. Metrics are calculated inside each
gameweek and then averaged, never pooled across a season.

### Dastan on every clean evaluation row

| metric | all players | starters only | what it means |
|---|---:|---:|---|
| objective | **0.6090** | **0.3977** | ranking score; higher is better, not a percentage |
| Spearman | 0.7461 | 0.2879 | correlation of the full predicted and actual ranks |
| NDCG@10 | 0.4720 | 0.5076 | quality of the predicted top ten |
| MAE | **0.918 pts** | 2.172 pts | absolute points error; lower is better |
| mean prediction / actual | 1.121 / 1.133 | 2.695 / 3.525 | calibration; starter projections are conservative |

Objective is `0.5 × Spearman + 0.5 × NDCG@10`. It measures selection quality, not the
probability that a forecast is "correct". MAE should not be compared across the two
cohorts because the all-player pool contains many zero-minute rows.

### Against FPL on identical rows

FPL's `ep_next` is missing for a small part of the evaluation set, so this comparison
restricts **both models to the rows where it exists**.

| cohort | Dastan | FPL `ep_next` | delta |
|---|---:|---:|---:|
| all players | **0.6097** | 0.5321 | **+0.0776** |
| starters only | **0.3963** | 0.3139 | **+0.0824** |

`ep_next` covers **17,307 player-gameweeks (98.2%)** here. Both methods are rescored
on that subset; missing FPL forecasts are never turned into zero.

### Other sanity checks, on paired rows

| cohort | Dastan | previous-five mean | last match | price (rank-only) | v12 recipe |
|---|---:|---:|---:|---:|---:|
| all players | **0.6090–0.6093** | 0.5317 | 0.4978 | 0.3571 | 0.5937 |
| starters only | **0.3977–0.3981** | 0.2760 | 0.2342 | 0.3584 | 0.3892 |

The small Dastan range reflects each baseline's eligible-row subset. Price is a
ranking baseline only: a price MAE would compare £0.1m units with FPL points and mean
nothing.

### Practical reading

- The 60-minute head has **0.9538 AUC** out of sample. That is ranking AUC, not 95.38%
  classification accuracy.
- Dastan's predicted top ten contains **1.88 of the actual top ten** on average.
- Its top-ranked player finishes in the actual top ten in **10 of 24 gameweeks (41.7%)**.

Full baseline tables, per-block detail, calibration, coverage and caveats:
[docs/ACCURACY.md](docs/ACCURACY.md).

### What we will not claim

Against the v12 recipe, Dastan gains **+0.0153 on all players** and **+0.0085 on
starters** in this rerun. Availability by itself adds **+0.0044 all / +0.0001
starters** over the rest of v13—below the predeclared 0.0061 keep margin on these
windows. If your problem is "rank players already known to start," do not infer an
availability-feature advantage from the headline result.

Also worth knowing: **price alone scores 0.3584 on starters.** The market is good;
check any model against it.

---

## Against OpenFPL

Dastan's feature engineering is **OpenFPL's**, used as code rather than reimplemented:

> Groos, D. *OpenFPL: An open-source forecasting method rivaling
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

Both models score on **identical rows** under a controlled regime: train
2020-21..2023-24, early stop 2024-25, test 2025-26 GW1-24, 18,173 player-gameweeks.
Dastan is averaged over all three published seeds. Both models train without the test
season and Dastan's usual six-season advantage is **not** a factor.

| cohort | Dastan | OpenFPL | delta |
|---|---|---|---|
| **all players** | **0.5641** | 0.5044 | **+0.0597** |
| starters only | 0.2626 | 0.2564 | +0.0062 |

| metric (all players) | Dastan | OpenFPL |
|---|---|---|
| Spearman | 0.7539 | 0.6961 |
| NDCG@10 | 0.3744 | 0.3128 |
| MAE | **0.948** | 1.167 |
| RMSE | **1.928** | 2.017 |

The paired gameweek-bootstrap interval for the full-pool decision-score delta is
**+0.0412 to +0.0798**. Dastan's MAE is lower by 0.219 points per player-gameweek.

**On 60-minute players the two are indistinguishable.** The +0.0062 interval is
−0.0093 to +0.0224. OpenFPL is marginally better on Spearman (0.1334 vs 0.1266) and
MAE (2.319 vs 2.340); Dastan is better on NDCG@10 (0.3987 vs 0.3794).

### Where the difference comes from

Following OpenFPL's return categories, RMSE on the same 18,173 rows is:

| realized return | Dastan | OpenFPL | lower error |
|---|---:|---:|---|
| 0 minutes | **0.583** | 0.964 | Dastan |
| played, ≤2 points | **1.381** | 1.518 | Dastan |
| 3–4 points | 1.442 | **1.304** | OpenFPL |
| ≥5 points | 5.756 | **5.675** | OpenFPL |

So the full-pool advantage is participation and low-return prediction. **OpenFPL is
slightly better on tickers and haulers.** That qualification matters more than the
headline win.

On the 17,888 rows with provenance-checked FPL projections, the three-way decision
scores are **Dastan 0.5648, OpenFPL 0.5048, FPL `ep_next` 0.4946**. The previous-five
baseline scores 0.4883 on its 17,927 eligible rows.

These numbers are lower than the walk-forward figures above because this regime trains
on four seasons instead of five and tests on GW1-24, which includes the hard early-season
gameweeks where little form has accumulated.

Reproduce and inspect the machine-readable evidence:

```bash
python -m dastan.benchmark_openfpl --seeds 3
```

We do not quote FPL Review or multi-gameweek horizons: synchronized FPL Review
predictions are unavailable for these rows, and this benchmark is one gameweek ahead.

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
| `data/pre_deadline_ep_next.parquet` | 142,173 | FPL's own projection, provenance-checked |
| `data/pre_deadline_signals.parquet` | 137,982 | status, chance of playing, set-piece order, age |
| `data/openfpl_predictions.csv` | 18,173 | OpenFPL's predictions, for the head-to-head |
| `data/players.csv` | 4,717 | `fpl_code` → name and position, per season |
| `data/mappings/fpl_understat_training_snapshot.csv` | 1,564 | exact historical IDs present in the release |
| `data/mappings/fpl_understat_players.csv` | 1,564 | audited historical IDs recommended for new joins |
| `data/mappings/current_fpl_understat_players.csv` | 1,681 | latest versioned operational player registry; 1,598 mapped |
| `data/mappings/current_fpl_understat_clubs.csv` | 30 | latest versioned operational club registry; 28 mapped |
| `data/mappings/fpl_understat_current.csv` | 573 | complete 2026-27 roster; 517 mapped, 56 unresolved |
| `data/season_registry.json` | — | completed-season coverage and immutable row/hash guards |
| `data/source_pins.json` | — | reviewed source commits for future data rebuilds |

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
python -m dastan.artifacts             # verify all 40 model-contract files
python -m dastan.datasets verify       # verify release data hashes and invariants
python -m dastan.mappings              # verify the published FPL-to-Understat mapping
python -m dastan.reproduce             # retrain and require equivalent holdout quality
python -m dastan.reproduce --strict    # require identical artefacts on the release platform
python -m dastan.train --out experiments/my-model
python -m dastan.evaluate --scope clean --seeds 3  # walk-forward vs paired baselines
python -m dastan.benchmark_openfpl --seeds 3       # controlled, identical-row head-to-head
```

`models/` contains a 40-file public contract: the 38-file production contract plus
the two public-only inputs needed for retraining (`hyperparameters.json` and
`core_feature_cols.json`). Thirty-six files are required for inference, including the
coherent minutes calibration. `models/artifact_manifest.json` fingerprints the full
contract and records the corresponding production commit and mapping release. Trained
heads, feature order, calibrations, blend weights, and serving metadata are byte-for-byte
copies of the 38-file production contract recorded in that manifest.

Model reproduction starts from the published, deadline-anchored feature frame. The
optional public-source reconstruction is separate so provider corrections cannot
silently overwrite the release:

```bash
python -m pip install -r requirements-data.txt
python -m dastan.datasets all
python -m dastan.datasets compare .cache/rebuilt-data
```

The raw cache and candidate outputs are gitignored. Each run records source and output
SHA-256 manifests. See [`docs/REBUILDING_DATA.md`](docs/REBUILDING_DATA.md) for source
pins, measured reconstruction coverage, and the boundary between exact release bytes
and mutable Understat data.

Mid-season and annual releases use a separate prepare → review → accept → train →
promote workflow. No season number is edited in Python, and candidate training refuses
to start if an accepted hash changes. See [`docs/RETRAINING.md`](docs/RETRAINING.md).

The default reproduction threshold is the documented 0.0106 single-seed objective
noise for both evaluation cohorts. Exact model bytes and prediction deltas are always
reported; `--strict` makes those exact comparisons mandatory. The tighter 0.0061 keep
margin applies to the three-seed experiment protocol, not this one-seed check.

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
