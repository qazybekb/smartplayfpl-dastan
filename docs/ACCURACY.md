# Accuracy

Dastan is evaluated against things an FPL manager could actually use instead:
FPL's official `ep_next`, OpenFPL, recent points, player price, and the previous
Dastan recipe. A score without those references is not an accuracy claim.

There are two controlled protocols. Do not compare a number from one protocol with a
number from the other:

| protocol | purpose | train / test | rows |
|---|---|---|---:|
| clean walk-forward | Dastan, `ep_next`, simple baselines, prior recipe | three unseen 2024-25 blocks | 17,622 player-GWs |
| OpenFPL head-to-head | Dastan, OpenFPL, `ep_next`, simple baselines | train 2020-21..2023-24; test 2025-26 GW1-24 | 18,173 player-GWs |

Both operate at **player-gameweek grain**. Metrics are computed inside each gameweek
and then averaged. Every paired comparison restricts all methods to the exact same
eligible rows. Dastan predictions are the arithmetic mean of seeds 42, 7, and 2026.

The machine-readable evidence is [`evaluation.json`](evaluation.json),
[`baseline_benchmark.json`](baseline_benchmark.json), and
[`openfpl_benchmark.json`](openfpl_benchmark.json). The retained
[`walkforward_predictions.parquet`](walkforward_predictions.parquet) can reproduce the
clean baseline report without fitting models again.

## The short answer

On the clean 2024-25 walk-forward blocks, Dastan's decision score is **0.6090** over all
players and **0.3977** over players who reached 60 minutes. The score is
`0.5 × Spearman + 0.5 × NDCG@10`; it is a ranking score, not "60.90% accurate."

Against FPL's official projection on the **same 17,307 rows**:

| cohort | Dastan | FPL `ep_next` | delta |
|---|---:|---:|---:|
| all players | **0.6097** | 0.5321 | **+0.0776** |
| 60+ minutes | **0.3963** | 0.3139 | **+0.0824** |

Under the separate OpenFPL protocol, Dastan and OpenFPL are trained without the test
season and scored on the same 18,173 rows:

| cohort | Dastan | OpenFPL | delta | paired 95% interval |
|---|---:|---:|---:|---:|
| all players | **0.5641** | 0.5044 | **+0.0597** | **+0.0412 to +0.0798** |
| 60+ minutes | 0.2626 | 0.2564 | +0.0062 | −0.0093 to +0.0224 |

The interval resamples gameweeks while holding the fitted predictions fixed. It
measures week-to-week variation, not all uncertainty in model training.

The conclusion is narrower than "Dastan wins": **Dastan is clearly better over the
full player pool, while Dastan and OpenFPL are not distinguishable among 60-minute
players on this sample.** The return-band table below shows why.

## How to read the metrics

| metric | direction | interpretation |
|---|---|---|
| decision score | higher | `0.5 × Spearman + 0.5 × NDCG@10`; model-selection score |
| Spearman | higher | quality of the full player ranking within each gameweek |
| NDCG@10 | higher | quality of the ten players ranked highest |
| MAE / RMSE | lower | absolute / squared points error |
| top-10 overlap | higher | fraction of actual top ten found in predicted top ten |
| p60 AUC | higher | ranking quality for reaching 60 minutes; not classification accuracy |
| Brier | lower | calibration and sharpness of the 60-minute probability |

Two cohorts are always shown:

| cohort | question | important caveat |
|---|---|---|
| all players | can the model rank the entire available pool? | most rows are non-players, so availability dominates |
| 60+ minutes | can it rank players who actually played substantial minutes? | harder and much noisier; selected after the match only for diagnosis |

MAE should not be compared across the cohorts. Removing non-players removes many easy
zero-point rows.

## Clean walk-forward evaluation

Each eight-gameweek block is retrained from scratch using only earlier data. Later
seasons are removed when an earlier window is evaluated. These results use the exact
current **286-feature** public contract and three-seed prediction ensemble.

### Dastan on every clean row

| metric | all players | 60+ minutes |
|---|---:|---:|
| decision score | **0.6090** | **0.3977** |
| Spearman | 0.7461 | 0.2879 |
| NDCG@10 | 0.4720 | 0.5076 |
| MAE | 0.918 | 2.172 |
| RMSE | 1.839 | 3.165 |
| mean prediction | 1.121 | 2.695 |
| mean actual | 1.133 | 3.525 |
| player-gameweeks | 17,622 | 4,849 |

Dastan is well calibrated across the full pool (1.121 predicted versus 1.133 actual)
and conservative among 60-minute players (2.695 versus 3.525). That is one reason to
use it primarily as a ranking, not as a promise of an exact score.

### Paired baselines

Every row below is its own exact-row comparison. `ep_next` has 98.2% coverage; the
recent-points baselines have 99.6%. Price covers the full frame but is **rank-only**:
price is not measured in FPL points, so MAE or RMSE for it would be meaningless.

All players:

| baseline | Dastan, same rows | baseline | delta | rows | GWs |
|---|---:|---:|---:|---:|---:|
| FPL `ep_next` | **0.6097** | 0.5321 | **+0.0776** | 17,307 | 24 |
| previous-five mean | **0.6093** | 0.5317 | **+0.0776** | 17,547 | 24 |
| last match points | **0.6093** | 0.4978 | **+0.1115** | 17,547 | 24 |
| price, rank-only | **0.6090** | 0.3571 | **+0.2519** | 17,622 | 24 |
| v12 recipe | **0.6090** | 0.5937 | **+0.0153** | 17,622 | 24 |

Players who reached 60 minutes:

| baseline | Dastan, same rows | baseline | delta | rows | GWs |
|---|---:|---:|---:|---:|---:|
| FPL `ep_next` | **0.3963** | 0.3139 | **+0.0824** | 4,835 | 24 |
| previous-five mean | **0.3981** | 0.2760 | **+0.1221** | 4,841 | 24 |
| last match points | **0.3981** | 0.2342 | **+0.1639** | 4,841 | 24 |
| price, rank-only | **0.3977** | 0.3584 | **+0.0393** | 4,849 | 24 |
| v12 recipe | **0.3977** | 0.3892 | **+0.0085** | 4,849 | 24 |

For point forecasts, the error comparison against `ep_next` is also favourable:

| cohort | Dastan MAE | `ep_next` MAE | Dastan RMSE | `ep_next` RMSE |
|---|---:|---:|---:|---:|
| all players | **0.930** | 1.070 | **1.855** | 2.090 |
| 60+ minutes | **2.174** | 2.546 | **3.168** | 3.523 |

### Independence from FPL's input

`ep_next` is one of Dastan's 286 features, so the head-to-head alone cannot show that
the gain is independent of FPL's forecast. A separate three-seed arm removes
`ar_ep_next` and retrains the same architecture. On the 17,307 FPL-covered rows:

| forecast | decision score | MAE | delta versus FPL |
|---|---:|---:|---:|
| Dastan | **0.6097** | 0.930 | **+0.0776** |
| Dastan without `ep_next` | **0.6079** | **0.928** | **+0.0758** |
| FPL `ep_next` | 0.5321 | 1.070 | - |

The full-model advantage has a paired-gameweek 95% interval of **+0.0526 to +0.1033**;
its MAE reduction is **0.121 to 0.161 points**. Across all 17,622 clean rows, adding
`ep_next` back changes Dastan's decision score by **+0.0018**, with an interval of
**-0.0065 to +0.0113**, and slightly worsens MAE. The observed lead over FPL remains
when its projection is not an input. These intervals resample the 24 test gameweeks
while holding each three-seed forecast ensemble fixed.

### Per-block decision score

All players:

| block | Dastan | on `ep_next` rows | FPL `ep_next` | no availability | v12 recipe |
|---|---:|---:|---:|---:|---:|
| 2024-25 GW15-22 | 0.6073 | 0.6081 | 0.5416 | 0.6078 | 0.5934 |
| 2024-25 GW23-30 | 0.6257 | 0.6262 | 0.5442 | 0.6220 | 0.6023 |
| 2024-25 GW31-38 | 0.5941 | 0.5949 | 0.5105 | 0.5841 | 0.5855 |

Players who reached 60 minutes:

| block | Dastan | on `ep_next` rows | FPL `ep_next` | no availability | v12 recipe |
|---|---:|---:|---:|---:|---:|
| 2024-25 GW15-22 | 0.4063 | 0.4044 | 0.3211 | 0.4086 | 0.3978 |
| 2024-25 GW23-30 | 0.4089 | 0.4079 | 0.3131 | 0.4111 | 0.3974 |
| 2024-25 GW31-38 | 0.3780 | 0.3766 | 0.3075 | 0.3732 | 0.3724 |

Availability improves the all-player average by **+0.0044** here and the 60-minute
average by **+0.0001**. Both are below the predeclared 0.0061 three-seed keep margin on
these windows. That is consistent with the mechanism—availability helps identify who
plays—but it is not evidence of a starter-ranking gain.

### Practical and minutes metrics

| result | value |
|---|---:|
| actual top-ten players found in predicted top ten | **1.88 of 10** |
| top prediction finishes in actual top ten | **10 of 24 GWs (41.7%)** |
| 60-minute AUC | **0.9538** |
| 60-minute Brier score | **0.0768** |

Actual top-ten scorers are dominated by difficult-to-predict hauls. The overlap is
useful context for what a high rank correlation does—and does not—guarantee.

## Controlled head-to-head with OpenFPL

[OpenFPL](https://github.com/daniegr/OpenFPL) is the fairest external reference because
Dastan's rolling feature engineering originates there. To avoid scoring Dastan on a
window its released weights saw, Dastan is retrained using OpenFPL's four-season
development regime: 2020-21..2023-24 training, 2024-25 early stopping, and 2025-26
GW1-24 testing. Dastan uses three seeds here; OpenFPL's stored predictions are fixed.

### Primary exact-row result

| metric, all players | Dastan | OpenFPL | delta |
|---|---:|---:|---:|
| decision score | **0.5641** | 0.5044 | **+0.0597** |
| Spearman | **0.7539** | 0.6961 | **+0.0578** |
| NDCG@10 | **0.3744** | 0.3128 | **+0.0616** |
| MAE | **0.948** | 1.167 | **−0.219** |
| RMSE | **1.928** | 2.017 | **−0.088** |

For 60-minute players, Dastan's decision-score delta is only +0.0062 and its paired
95% interval crosses zero. OpenFPL has slightly better Spearman (0.1334 versus 0.1266)
and MAE (2.319 versus 2.340); Dastan has better NDCG@10 (0.3987 versus 0.3794). Treat
the two methods as tied in this cohort.

### Three-way comparison with official FPL

The provenance-checked `ep_next` data cover 17,888 of the OpenFPL test rows. All three
models are rescored on that subset:

| cohort | Dastan | OpenFPL | FPL `ep_next` |
|---|---:|---:|---:|
| all players | **0.5648** | 0.5048 | 0.4946 |
| 60+ minutes | **0.2628** | 0.2566 | 0.2123 |

The previous-five baseline covers 17,927 rows and scores 0.4883, versus Dastan at
0.5652 and OpenFPL at 0.5072 on those same rows. Price covers all 18,173 rows and
scores 0.3382 as a rank-only baseline.

### Return bands, following OpenFPL

OpenFPL reports error separately for non-players, blanks, 3–4 point returns, and
haulers. Applying the same idea to the exact shared rows produces this RMSE table
(lower is better):

| realized return | rows | Dastan | OpenFPL | lower error |
|---|---:|---:|---:|---|
| 0 minutes (`Zeros`) | 10,902 | **0.583** | 0.964 | Dastan |
| played, ≤2 points (`Blanks`) | 4,562 | **1.381** | 1.518 | Dastan |
| 3–4 points (`Tickers`) | 1,112 | 1.442 | **1.304** | OpenFPL |
| ≥5 points (`Haulers`) | 1,597 | 5.756 | **5.675** | OpenFPL |

This is the most important qualification in the report. **Dastan's overall advantage
comes from non-players and low returns. OpenFPL is slightly better on tickers and
haulers.** The JSON report also provides this split for GKP, DEF, MID, and FWD.

## What we do not claim

- We do not compare with FPL Review. Synchronized proprietary forecasts for these
  rows are unavailable; copying numbers from OpenFPL's different 2024-25 sample would
  not be a head-to-head comparison.
- We do not claim two- or three-gameweek forecast accuracy. The controlled benchmark
  here is one gameweek ahead.
- The clean walk-forward suite covers 24 gameweeks of one season. The OpenFPL test adds
  24 gameweeks of a second season, but neither substitutes for continued live tracking.
- The no-`ep_next` arm establishes independence on this 24-gameweek sample, not every
  future season. Continued prospective tracking remains necessary.
- The 60-minute cohort is selected using realized minutes. It diagnoses ranking among
  players who played; it is not a pre-deadline selection rule.

## Reproduce and audit

```bash
python -m dastan.datasets verify
python -m dastan.artifacts
python -m dastan.benchmark_baselines --check
python -m dastan.evaluate --scope clean --seeds 3
python -m dastan.benchmark_openfpl --seeds 3
python -m unittest discover -v
```

The two training benchmarks are intentionally not part of normal CI because they fit
dozens of XGBoost heads. CI regenerates the clean baseline report from retained
predictions, then verifies the evidence schema, row counts, input hashes, and release
contracts. Regenerate and review all reports whenever data, features, model code, or
the evaluation protocol changes.
