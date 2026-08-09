# Reproducible baselines

This report is generated from retained, out-of-sample walk-forward predictions.
Every head-to-head comparison restricts Dastan and the baseline to identical
player-gameweek rows. Ranking metrics are calculated within each gameweek and
then averaged.

```bash
python -m dastan.benchmark_baselines --seeds 3 --n-jobs 8
```

Scope: **24 gameweeks, 17,622 player-gameweeks**.

## FPL's projection

FPL `ep_next` is the strongest free external baseline available in the official
game. It is also one Dastan input, so the full-model row measures value added
over FPL; the separately retrained no-`ep_next` row tests independence.

| cohort | forecast | rows | objective | Spearman | NDCG@10 | MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| all | **Dastan** | 17,307 | 0.6097 | 0.7474 | 0.4720 | 0.930 | 1.855 |
| all | **Dastan without `ep_next`** | 17,307 | 0.6079 | 0.7475 | 0.4683 | 0.928 | 1.855 |
| all | **FPL `ep_next`** | 17,307 | 0.5321 | 0.6696 | 0.3946 | 1.070 | 2.090 |
| starters | **Dastan** | 4,835 | 0.3963 | 0.2851 | 0.5076 | 2.174 | 3.168 |
| starters | **Dastan without `ep_next`** | 4,835 | 0.3951 | 0.2856 | 0.5046 | 2.171 | 3.167 |
| starters | **FPL `ep_next`** | 4,835 | 0.3139 | 0.2071 | 0.4207 | 2.546 | 3.523 |

Paired-gameweek bootstrap (95%): objective advantage
**+0.0526 to +0.1033**; MAE reduction
**0.121 to 0.161 points**. This interval resamples the
24 test gameweeks; it does not add another layer of model-seed uncertainty.

Removing `ep_next` still leaves an objective advantage of
**+0.0758** over FPL. Adding the feature
back changes Dastan's objective by only
**+0.0018**; see the JSON report for its
paired interval and error metrics.

## Public baselines

The table below reports the all-player cohort. `Price` is a ranking proxy, not a
points forecast, so point-error metrics do not apply.

| baseline | coverage | Dastan objective | baseline objective | advantage | baseline MAE |
|---|---:|---:|---:|---:|---:|
| FPL ep_next | 17,307/17,622 (98.2%) | 0.6097 | 0.5321 | +0.0776 | 1.070 |
| Last appearance points | 17,547/17,622 (99.6%) | 0.6093 | 0.4977 | +0.1116 | 1.122 |
| Last-5 appearance mean | 17,547/17,622 (99.6%) | 0.6093 | 0.5317 | +0.0776 | 1.025 |
| Price | 17,622/17,622 (100.0%) | 0.6090 | 0.3571 | +0.2519 | n/a |

## Participation

The 60-minute head is evaluated as a probability model, separately from xPts.
AUC measures ranking of participation likelihood; Brier measures probability
calibration and sharpness. AUC is not classification accuracy.

| AUC | Brier | 60-minute base rate |
|---:|---:|---:|
| 0.9538 | 0.0768 | 27.6% |

## Error sanity check

An all-player MAE can look artificially good because most registered players do
not appear. The constant-zero forecast is included as an error-only sanity check;
it cannot rank players and therefore has no Spearman or NDCG score.

| forecast | rows | MAE | RMSE |
|---|---:|---:|---:|
| Dastan | 17,622 | 0.917 | 1.846 |
| Always zero | 17,622 | 1.145 | 2.605 |

## OpenFPL-style return buckets

These outcome-conditioned errors use the buckets from the OpenFPL paper.
They explain *where* error occurs; they do not test whether a model can identify a
future bucket before the match. The test window differs, so these numbers are not
directly comparable with OpenFPL's paper tables.

| actual outcome | rows | Dastan RMSE (MAE) | FPL `ep_next` RMSE (MAE) |
|---|---:|---:|---:|
| Zeros | 10,181 | 0.630 (0.284) | 0.740 (0.300) |
| Blanks | 4,979 | 1.405 (1.077) | 2.034 (1.443) |
| Tickers | 645 | 1.404 (1.089) | 2.290 (1.771) |
| Haulers | 1,502 | 5.462 (4.749) | 5.572 (4.770) |

## Error by position

This is the same paired `ep_next` sample, split by the player's FPL position.

| position | rows | Dastan RMSE (MAE) | FPL `ep_next` RMSE (MAE) |
|---|---:|---:|---:|
| GKP | 1,775 | 1.586 (0.722) | 1.826 (0.936) |
| DEF | 5,853 | 1.813 (0.953) | 2.008 (1.067) |
| MID | 7,760 | 1.903 (0.925) | 2.159 (1.089) |
| FWD | 1,919 | 2.061 (1.067) | 2.340 (1.138) |

## Limits of comparison

- The public benchmark is one gameweek ahead. Historical two- and three-gameweek
  pre-deadline forecasts were not captured, so no multi-horizon claim is made.
- FPL Review is not included. A fair comparison requires licensed forecasts
  captured before the same deadlines; OpenFPL's paper values use a different test
  window and cannot be pasted beside Dastan's numbers.
- The separate OpenFPL head-to-head retrains both methods under OpenFPL's published
  regime and scores 18,173 identical rows. See `openfpl_benchmark.json`.

The machine-readable source is [`baseline_benchmark.json`](baseline_benchmark.json),
and the retained predictions are
[`walkforward_predictions.parquet`](walkforward_predictions.parquet).
