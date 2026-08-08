# Accuracy

Every number here comes from **walk-forward evaluation**: the model is retrained from
scratch for each block, so nothing in a block's test window existed when its model was
fitted. Three seeds (42, 7, 2026), predictions averaged. Metrics are computed **inside
each gameweek and then averaged** — see [METHODOLOGY.md](METHODOLOGY.md) §2 for why that
matters more than it sounds.

Reproduce with `python -m dastan.evaluate --seeds 3`.

> **One note on precision.** These blocks were scored with 287 features. The released
> weights have 286: a column that was 95.2% exactly zero (a sparsely populated field
> whose gaps had been filled with `0` rather than left absent) was removed afterwards.
> Removing it measured **−0.0001 all / −0.0002 starters** across three windows, so the
> tables below are unaffected at the precision shown. It was dropped not for accuracy
> but to remove a train/serve skew: in production that column is fully populated with
> real values, so a served row would land where the model had barely trained.

> **Read the two cohorts separately.** They tell genuinely different stories, and the
> headline improvement lives almost entirely in one of them. That is stated plainly
> below rather than averaged away.

---

## 1. Headline

Three chronological blocks of 2024-25, none of which was ever used to select a feature,
a fold or a hyperparameter.

### All players

| model | objective | Spearman | NDCG@10 | MAE |
|---|---|---|---|---|
| **Dastan** | **0.6072** | 0.7464 | 0.4680 | **0.915** |
| core feature set only | 0.5944 | — | — | — |
| FPL's own `ep_next` | 0.5321 | — | — | 1.06 |
| rolling-5 form | 0.5260 | — | — | — |
| last gameweek's points | 0.4950 | — | — | — |
| price | 0.3571 | — | — | — |

**Dastan beats FPL's own published projection by +0.0751** — more than twelve times the
noise floor. It beats the best naive baseline by +0.0812.

### Starters only (played 60+ minutes)

| model | objective | Spearman | NDCG@10 | MAE |
|---|---|---|---|---|
| **Dastan** | **0.3959** | 0.2866 | 0.5051 | 2.172 |
| core feature set only | 0.3949 | — | — | — |
| price | 0.3584 | — | — | — |
| FPL's own `ep_next` | 0.3139 | — | — | — |
| rolling-5 form | 0.2820 | — | — | — |
| last gameweek's points | 0.2404 | — | — | — |

**Dastan beats `ep_next` by +0.0820 here too.** But note the second row.

---

## 2. The honest caveat

Against the core feature set, the added families are worth:

| cohort | gain | noise floor | verdict |
|---|---|---|---|
| all players | **+0.0128** | 0.0061 | real |
| starters only | **+0.0010** | 0.0061 | **indistinguishable from nothing** |

**Everything the extra features buy is in knowing who will not play.** Among players who
actually start, the enriched feature set is no better than the core one. The same is
true of the availability family specifically: +0.0057 on all players, +0.0004 on
starters.

This is consistent with the mechanism rather than a surprise — an injury flag tells you
*whether* a player features, not how well he does — but it is a real limit and anyone
building on this should know it. **If your use case is "rank the players I already know
will start", the feature engineering here is not where the value is.** The architecture
and the training protocol are.

Second honest note: **price alone scores 0.3584 on starters**, against the full model's
0.3959. The market is a strong estimator among players who play, and any model claiming
to be useful should be checked against it rather than only against form baselines.

---

## 3. Per-block detail

Objective, all players:

| block | Dastan | without availability | core only | FPL `ep_next` |
|---|---|---|---|---|
| 2024-25 GW15-22 | 0.6128 | 0.6063 | 0.5997 | 0.5416 |
| 2024-25 GW23-30 | 0.6241 | 0.6196 | 0.6077 | 0.5442 |
| 2024-25 GW31-38 | 0.5847 | 0.5785 | 0.5757 | 0.5105 |
| 2025-26 GW23-30 † | 0.5794 | 0.5790 | 0.5573 | 0.5284 |
| 2025-26 GW31-38 † | 0.5922 | 0.5881 | 0.5699 | 0.5246 |

Objective, starters only:

| block | Dastan | without availability | core only |
|---|---|---|---|
| 2024-25 GW15-22 | 0.4091 | 0.4011 | 0.4038 |
| 2024-25 GW23-30 | 0.4105 | 0.4164 | 0.4047 |
| 2024-25 GW31-38 | 0.3680 | 0.3689 | 0.3761 |
| 2025-26 GW23-30 † | 0.3056 | 0.3003 | 0.2829 |
| 2025-26 GW31-38 † | 0.3250 | 0.3271 | 0.3181 |

† These two blocks were used during feature discovery, so they are **expected to read
high** and are shown for completeness only. Quote the 2024-25 blocks.

Note how the starters column moves around: the ablation *wins* on two of five blocks.
That is what a difference smaller than the noise floor looks like, and it is why single
-block, single-seed comparisons are not evidence.

---

## 4. Error and calibration

Clean blocks, Dastan:

| | all players | starters |
|---|---|---|
| MAE | 0.915 | 2.172 |
| RMSE | 1.839 | 3.164 |
| mean predicted | 1.119 | 2.699 |
| mean actual | 1.133 | 3.525 |

**Well calibrated on the full pool** (1.119 vs 1.133) and **under-predicts starters by
0.83 points** (2.699 vs 3.525).

That under-prediction is structural, not a bug to be tuned away. Points are clipped at
zero and the distribution is violently right-skewed: a model minimising squared error
across a population where the median starter returns 2 will not predict the 13 that
occasionally happens. It matters for how you *use* the output — Dastan ranks well but
its absolute values are conservative for starters, so do not read a 5.0 projection as a
point estimate of a haul.

Do not compare MAE across cohorts. The all-players MAE is low mostly because most
players score nothing and the model correctly says so.

---

## 5. Practical metrics

How the model does at the decision a manager actually makes:

| metric | value |
|---|---|
| top-10 overlap | **0.183** |
| our top pick finishes in the actual top 10 | **37.5%** of gameweeks |

Top-10 overlap of 0.183 means that of the ten highest-scoring players in a gameweek,
**about 1.8 were in our predicted top ten**.

That sounds poor and is roughly the ceiling anyone should expect. A gameweek's actual
top ten is dominated by hauls — a defender scoring a header, a midfielder getting a
brace — and those events are close to irreducibly random at the individual level. Any
FPL model quoting a dramatically higher figure is either scoring pooled across
gameweeks, scoring on a restricted pool, or leaking.

---

## 6. The minutes gate

The `p60` head predicts whether a player will reach 60 minutes. It is the strongest
component of the model and is useful on its own.

| block | AUC | Brier | base rate |
|---|---|---|---|
| 2024-25 GW15-22 | 0.9525 | 0.0784 | 0.290 |
| 2024-25 GW23-30 | 0.9588 | 0.0722 | 0.272 |
| 2024-25 GW31-38 | 0.9503 | 0.0796 | 0.264 |

**AUC ≈ 0.95** out of sample. If you want one thing from this repository and do not care
about points modelling, take this head: `predictor.Dastan().predict_position(...)`
returns `p60` directly.

---

## 7. Against OpenFPL

Dastan's feature engineering is built on OpenFPL, so it is the fairest reference point.
Both are scored on identical rows under the regime OpenFPL published for — train
2020-21..2023-24, early stop 2024-25, test 2025-26 GW1-24, 18,173 player-fixtures.

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

Both train on the same four seasons here, so Dastan's usual six-season advantage is not
a factor.

**All players: +0.0558**, nine times the noise floor, with MAE better by 0.22 points.

**Starters: +0.0025**, below the noise floor — and OpenFPL is marginally better within
that cohort on Spearman (0.1334 vs 0.1244) and MAE (2.319 vs 2.341). This reproduces
§2's conclusion from a completely independent direction: the advantage is in predicting
participation, not in ranking the players who play.

These are lower than §1 because this regime trains on four seasons rather than five and
tests on GW1-24, including the hard early-season weeks.

Run `python -m dastan.benchmark_openfpl` to reproduce; results are written to
`docs/openfpl_benchmark.json`.

---

## 8. What would move these numbers

Ranked by our estimate, all open:

1. **All-competition minutes.** Congestion features here see Premier League fixtures
   only, so a player who went 90 minutes in Europe on Thursday looks rested.
2. **Better availability data.** The `all` cohort gain came entirely from knowing who
   plays. Press-conference and lineup data closer to the deadline should extend that.
3. **Anything that improves the starters cohort.** Nothing we added did. This is the
   open problem, and it is where the remaining value is.
