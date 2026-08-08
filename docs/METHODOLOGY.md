# Methodology

How Dastan is evaluated, and why the protocol is stricter than it first appears
necessary. If you take one thing from this repository, take this document rather than
the model.

---

## 1. The number that governs everything: noise

Train the identical configuration on the identical data with a different random seed,
and the objective moves by **0.0106 on average, and 0.0178 at worst**.

That is larger than almost every real feature effect anyone will ever measure on FPL
data. It means a single-seed experiment showing "+0.005 from my new feature" is
**indistinguishable from having changed nothing at all**.

So every result here uses **three seeds (42, 7, 2026)**, and the threshold for keeping
a feature is derived from that measurement rather than chosen:

```
keep margin = 0.0106 / sqrt(3) = 0.0061
```

This is not conservatism for its own sake. Before this was measured, the working
threshold was 0.0020 — **5.3× below the noise it was screening against**. A protocol
like that does not detect signal; it manufactures it. One feature family entered an
earlier version of this model on a "+0.0058" that turned out to be entirely inside its
own measurement error.

**If you fork this repo and add features, measure your seed noise first.** It costs one
afternoon and it determines whether anything else you do means anything.

---

## 2. Scoring inside a gameweek, never pooled

Metrics are computed **within each gameweek and then averaged**, never by pooling every
player-gameweek in a season into one correlation.

Pooling flatters a model badly. Most of the variance across a pooled season is
*between* gameweeks — blank weeks against double gameweeks, easy fixture runs against
hard ones. A model can score a superb pooled Spearman while being no use at the only
question a manager faces: **given this week, who do I pick?**

The difference is not cosmetic. Pooled Spearman on this data reads around 0.73; the
per-gameweek figure for the same predictions is a different number measuring a
different thing. Anyone comparing FPL models should check which one is being quoted.

Grain is **player-gameweek**, not player-fixture. A double gameweek is two fixtures but
one decision, so its fixtures are summed before scoring. Scoring at fixture grain
double-counts exactly the players whose weeks matter most.

---

## 3. Two cohorts, always, and only one of them is optimised

Every number in this repository is reported for two populations:

| cohort | who | what it measures |
|---|---|---|
| **all** | every player in the pool | ~62% recorded zero minutes, so this rewards knowing **who will not play** |
| **starters** | played 60+ minutes | much harder, much noisier — **who you were actually choosing between** |

These behave differently and can disagree. A change that improves availability
prediction moves `all` substantially and `starters` barely at all, because it is
answering "does he feature?" rather than "how well does he do?".

`starters` is **reported but never optimised against**. That asymmetry is deliberate:
it means a model that wins only by getting better at predicting non-players is visible
in the numbers rather than hidden by them.

---

## 4. Chronological splits, and the season you forget to drop

Training data always precedes test data. That much is obvious. Two failures are not.

**Failure 1 — the later season left in.** Holding back 2024-25 GW31-38 as a test window
while leaving all of 2025-26 in the training set trains the model on a full year of
football from *after* the window it is being judged on. The split must drop every later
season, not merely later gameweeks. `evaluate.block_split` does this.

**Failure 2 — the deadline is not the kickoff.** Rolling features are normally built by
shifting one row back, ordered by kickoff. In a double gameweek both fixtures share one
deadline, so sorted by kickoff the second fixture's "previous match" is the first
fixture of the same gameweek — a result nobody had seen when the squad was locked.

Measured on this dataset: **6,958 player-gameweeks contain more than one fixture, and
40.6% of them carried different rolling values across those fixtures** before this was
fixed. The correction (`data.anchor_to_deadline`) broadcasts the earliest fixture's
history across the whole gameweek, and `data.assert_deadline_anchored` fails loudly if
a frame violates it.

---

## 5. Discovery windows and confirmation windows are different windows

Screening many candidates on the same window and then quoting that window's number for
the winner is the multiple-comparisons problem wearing a lab coat. With 8 candidates
and noise of 0.0106, something will look good.

So windows have roles, and the roles are not reused:

| window | role |
|---|---|
| 2025-26 GW23-30, GW31-38 | **discovery** — screened on repeatedly |
| 2025-26 GW15-22 | **hyperparameter tuning only** |
| 2024-25 GW31-38 | **confirmation** — has never selected anything |

A candidate that passes discovery is re-measured on a window that has selected nothing.
The availability family, the only one of eight that shipped, was positive on all three
independent windows (+0.0036, +0.0107, +0.0057).

---

## 6. Best-of-N tuning needs two separate corrections

A 500-trial Optuna search over the hyperparameters claimed **+0.0161**. It was worth
**nothing**, and the way it failed is worth more than the search.

| stage | value |
|---|---|
| best of 500 trials, single seed | 0.5919 |
| the same configuration at three seeds | 0.5775 |
| **seed-selection bias** | **+0.0090** |
| held-out windows: A / B / confirmation | +0.0058 / −0.0082 / −0.0022 |
| **honest mean across three held-out windows** | **−0.0015** |

Two distinct biases, and they need different instruments:

1. **Seed-luck (+0.0090).** Picking the maximum of 500 noisy draws selects for luck.
   Re-running the top 12 configurations at three seeds removes it — and it *reordered*
   them: the stage-1 leader finished **8th of 12**. Shipping the top trial naively would
   have cost 0.0062 against the config that actually won, itself larger than the noise
   floor.
2. **Window-luck (the rest).** 500 trials selecting on one 8-gameweek block overfit that
   block. **No amount of re-seeding on the tuning window can detect this** — only
   scoring a *different* window does.

The released model therefore uses untuned hyperparameters, because they are better.

---

## 7. A real mechanism is not automatically a useful feature

The most useful negative result in this project. Home advantage genuinely **inverts** in
empty stadiums:

| crowd regime | away | home | home advantage |
|---|---|---|---|
| empty | 3.60 | 3.55 | **−0.05** |
| limited | 3.25 | 3.66 | +0.41 |
| full | 3.33 | 3.74 | +0.41 |

Real, cleanly identified, on 6,802 starter-rows, and the `limited` and `full` regimes
agreeing exactly is a good sign the windows are drawn correctly.

Adding it as a feature measured **−0.0018**. Rolling team form computed *during* the
empty-stadium period already encodes the weaker home effect; the model does not need to
be told. Football insight and feature value are different things, and only one of them
is measurable.

---

## 8. Suspect the instrument before the model

The harness that produced this project's early results had four defects: conditional
heads early-stopping on the wrong population, metrics aggregated at fixture rather than
player-gameweek grain, features not deadline-anchored, and position-gated feature
dictionaries silently ignored.

Fixing them was worth **+0.0144** — more than every feature added afterwards combined.
Several earlier conclusions reversed sign entirely, including one family reported at
−0.0084 that actually measures **+0.0171** and is now the strongest in the model.

Before trusting a measurement, check what is doing the measuring.
