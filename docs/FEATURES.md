# Features

286 columns. Where they come from, and — more usefully — the eleven candidate families
that were tested and **rejected**, with numbers.

---

## The shipped set

| block | count | origin |
|---|---|---|
| Player rolling | 115 | **OpenFPL** — 23 stats × 5 windows |
| Team rolling | 60 | **OpenFPL** — 12 stats × 5 windows |
| Opponent rolling | 60 | OpenFPL, extended (they ship 10 stats; 2 added) |
| Availability | 9 | Dastan — `p_any` / `p60` / `e_minutes` over 3, 5, 10 |
| Flat context | 6 | Dastan — price, ownership, transfer balance, venue, ranks |
| Defensive-contribution families | 20 | Dastan — measured, see below |
| Pre-deadline `ep_next` | 1 | FPL's own projection, provenance-checked |
| Pre-deadline availability signals | 3 | FPL bootstrap — status, chance of playing, news |
| **Total** | **286** | |

Rolling windows are `[1, 3, 5, 10, 38]` — last match, short form, medium form, long
form, and a full-season prior. Every rolling column is shifted so a fixture can never
contain its own result, and then **deadline-anchored** so it cannot contain the result
of an earlier fixture in the same gameweek (see METHODOLOGY §4).

### The one that surprised us

`ar_defensive_contribution_*`, `ar_cbit_*`, `ar_recoveries_*`, `ar_tackles_*` — player
defensive workload — is the **strongest single family in the model at +0.0171**. An
earlier, broken measurement had reported it at −0.0084 and it was excluded on that
basis. Defensive contributions became a scoring category in 2025-26 and the model was
blind to them.

### Pre-deadline signals, and why provenance is the whole game

Three columns from the FPL bootstrap: `status` (mapped to an ordinal risk),
`chance_of_playing_next_round`, and whether news text is attached. Worth **+0.0064**
pooled across three independent windows — the largest feature gain in the project.

The gain sits almost entirely in the `all` cohort and is ~0 on `starters`. That is
exactly what the mechanism predicts: a flag tells you **whether** a player features,
not how he ranks among those who do. A family that helps where its mechanism says it
should is more believable than one that merely scores well.

**These are only usable because of when they were captured.** A snapshot is accepted
only if it names the gameweek as `is_next`, agrees on its deadline, **and predates it**.
Median capture is 4.3 hours before the deadline; zero rows come from after it. Without
that check these columns are the answer key, not a feature — see [DATA.md](DATA.md).

Missing values are **−1, never 0**. 54,430 rows carry a genuine `ep_next` of exactly
zero; collapsing "no snapshot" into "FPL expects nothing" teaches the model that two
very different events are the same.

---

## Rejected — measured, with numbers

Everything here was implemented, screened at three seeds against the 0.0061 noise
floor, and **did not survive**. Published because knowing what has already failed is
more valuable than another list of what worked, and because several of these sound
obviously good.

### Tested against the final model

| family | columns | all | verdict |
|---|---|---|---|
| **biographical** — age at deadline, days since joining club | 2 | +0.0021 | below noise floor |
| **set_pieces** — declared penalty / free-kick / corner order | 3 | +0.0003 | below noise floor |
| **crowd** — behind-closed-doors and limited-capacity regime | 2 | −0.0018 | rejected |
| **travel** — away-team travel distance in km | 1 | −0.0041 | rejected |
| **kickoff** — hour, day of week, lunchtime/evening flags | 5 | −0.0057 | rejected, actively harmful |
| **rest_diff** — rest days, opponent rest days, differential | 3 | −0.0064 | rejected, worst of the eight |

**`set_pieces` is the instructive one.** Declared taker order sounds strictly better
than penalty share inferred from history — it is known before the player has taken
anything. It measured +0.0003. Five windows of scoring history has already learned who
takes penalties, and only 7–10% of rows are designated takers at all. Redundancy beats
plausibility.

**`crowd` is the other one.** Home advantage really does invert in empty stadiums
(−0.05 versus +0.41). The effect is real and the feature is still worthless, because
rolling team form computed during that period already contains it. See METHODOLOGY §7.

**`biographical`** failed partly on coverage: `birth_date` is populated on only 27% of
rows.

### Tested earlier, under a protocol now known to be unreliable

These were screened before the harness defects were found, so treat them as *unproven*
rather than *disproven*. Re-testing them properly is a genuinely open contribution.

| family | reported | status |
|---|---|---|
| `history_depth` — deeper rolling windows | −0.0117 | not re-run |
| `opp_position_concession` — goals conceded to each position | −0.0058 | not re-run |
| `pen_share` — penalty share inferred from history | −0.0033 | superseded by `set_pieces` (+0.0003) |
| `dc_threshold` — defensive-contribution hit/near rates | −0.0017 | retained in the model |
| `congestion` — rest days, matches in 14 days | −0.0024 | **see below** |

### The family that is in the model on no evidence

`F4_congestion` (2 columns: `ar_rest_days`, `ar_team_matches_14d`) is **shipped, and
should probably not be**. It entered on "+0.0058" under the old 0.0020 margin — a
number inside its own measurement error. Re-screened properly, removing it measured
+0.0010 on discovery and −0.0029 on the confirmation window: a coin flip.

It is retained only because changing it costs a retrain and buys nothing measurable.
It is documented here rather than quietly left looking justified.

There is a plausible reason it does nothing: **the frame contains Premier League
fixtures only**. A player who went 90 minutes in the Europa League on Thursday looks
fully rested to `ar_rest_days`. Real congestion is all-competition congestion, and
testing that properly needs European and domestic-cup fixture data this dataset does
not have. That is the most promising open lead in this file.

---

## What is deliberately not in the feature set

**Target-gameweek outcomes.** Five columns in the raw data (`starts`,
`defensive_contribution`, `cbit`, `recoveries`, `tackles`) describe the gameweek being
predicted. They are 100% zero whenever the player recorded zero minutes and correlate
0.60–0.90 with minutes. Using them is leakage. Their *lagged rolling* versions are
features; the contemporaneous columns are not.

**Anything without pre-deadline provenance.** If it cannot be shown to have existed
before the deadline, it does not go in, regardless of how predictive it looks. Features
that look extraordinarily strong are usually leaking.

**Player embeddings or per-player fixed effects.** Not tested. The rolling windows
already carry per-player level, and identity features risk memorising rather than
generalising across the season boundary where squads change.

---

## Ideas worth testing that we have not

Ranked by our guess at expected value. All are open.

1. **All-competition minutes.** European and cup fixtures, to make congestion real.
   The single most likely win in this list.
2. **Physical load** — distance covered, sprint counts. A genuinely different axis from
   anything here; needs a licensed provider.
3. **Lineup / team-news scraping closer to the deadline.** The availability family shows
   that "will he play" is where the points are. Better availability data should beat
   better points modelling.
4. **Set-piece order interacted with fixture** — the flat version failed, but
   "designated penalty taker *against a team that concedes penalties*" is a different
   quantity.
5. **Opponent-adjusted rolling stats.** The rolling features are raw; normalising by
   opponent strength at the point of accumulation is untested here.
