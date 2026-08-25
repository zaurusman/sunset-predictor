# Afterglow scoring v2 — diagnosis and plan

Written 2026-08-25.

## Status

| Phase | State |
|---|---|
| 0 — Measuring stick | **Mostly done.** `POST /rate` + `GET /ratings/stats` collect first-party labels with their raw inputs; one-tap UI on the verdict card, usable for any date up to a year back. `scripts/evaluate.py` is the committed harness — distribution stats, guardrails, and label correlation. Webcam labels not started. **One real label exists**, so accuracy remains unmeasured. |
| 1 — Neutralise dead weight | **Done.** Horizon and precipitation are now multiplicative gates, not weighted components; weights are 0.60/0.25/0.15. |
| 2 — Light corridor | **Done.** Wired into predict, forecast and heatmap. |
| 3 — Aerosol + column moisture | **Done.** Aerosol response is monotone decreasing (Corfidi); moisture scores the water column (TCWV); archive visibility is no longer invented; measured AOD now covers historical days too. Moisture's variance share went 2.8-5.0 % → 16.3-18.2 %. |
| 3.5 — Beauty is plural | **Done.** Unplanned; added after user feedback. Colour is scored as five independent pathways combined by soft-max, not one weighted recipe. See Part 4. |
| 4 — Earth-shadow afterglow timing | **Partly.** The clear-sky pathway peaks on real solar-depression geometry. Per-layer shadow-height illumination — the part that deletes tuned constants — is still the −3°/σ2° bell. |
| 5 — Calibrate the scale | **Done, then partly reversed.** Percentile display shipped, was measured, and was replaced: the score is now the raw physics score, with the seasonal percentile shown beneath it as context. See "Why percentile display was reversed". |
| 6 — Make ML shelving explicit | **Done.** `ML_BLEND_ALPHA = 1.0`, load-time quality gate, artifact moved to `data/dead/`. |
| 7 — Better inputs (ICON, ensemble confidence) | Not started — and the next phase to do, because it is the only one left that human labels cannot adjudicate. |

**Measured effect of Phase 2** — same one-year climatology as Part 2 below,
scored with and without the corridor:

```
                     mean    sd   p10   p50   p90   Epic%   >=70%
TelAviv      before  64.2   8.6  61.1  61.4  81.1   11.0     15.9
             after   62.8   8.5  58.3  61.3  75.3    7.7     13.4
London       before  61.2  15.3  41.3  61.6  83.2   15.9     26.6
             after   56.8  12.8  41.1  59.5  73.6    4.4     14.2
SanFrancisco before  61.4  14.1  40.7  61.3  83.9   15.6     22.2
             after   56.7  12.1  40.7  58.6  72.5    4.7     13.7
```

The corridor is *selective*, not a blanket subtraction: p10 barely moves
(London 41.3 → 41.1, SF 40.7 → 40.7) while the Epic rate falls by 2–3.6×. It is
removing false positives at the top of the distribution, which is exactly the
failure mode Part 2 identified.

**Measured effect of Phases 1 + 2 together** — same climatology, original
baseline vs. now:

```
                     mean    sd   p10   p50   p90   Epic%   >=70%
TelAviv      before  64.2   8.6  61.1  61.4  81.1   11.0     15.9
             now     49.5  12.1  43.1  47.0  68.5    4.7      9.9
London       before  61.2  15.3  41.3  61.6  83.2   15.9     26.6
             now     43.3  16.1  24.1  44.2  66.1    2.2      7.1
SanFrancisco before  61.4  14.1  40.7  61.3  83.9   15.6     22.2
             now     43.3  15.3  24.1  43.7  66.2    3.3      6.3
```

What improved:
- **Tel Aviv's degeneracy is broken.** The p10–p50 gap was 0.3 points
  (61.1 vs 61.4 — more than 40 % of the year returning one number); it is now
  3.9 points, and sd rose from 8.6 to 12.1.
- **Epic is rare again**: 11–16 % → 2.2–4.7 %.
- **The bottom of the range is reachable**: London p10 fell 41.3 → 24.1, so
  "Poor" is finally a category that occurs (0–1 % → 21–25 %).
- `GO_OUTSIDE_THRESHOLD = 70` now fires on 6–10 % of evenings — a few nights a
  month, which is what "worth changing your plans for" should mean. It became a
  defensible number rather than a knob.

### What Phase 1 exposed: the category cutoffs are now wrong

Deflating the scale left the *labels* calibrated to the old inflated one. The
mass simply moved from "Good" into "Decent":

```
TelAviv       Poor: 2.7%  Decent:79.7%  Good: 5.8%  Great: 7.1%  Epic: 4.7%
London        Poor:25.2%  Decent:49.9%  Good:14.0%  Great: 8.8%  Epic: 2.2%
SanFrancisco  Poor:20.8%  Decent:59.2%  Good: 8.8%  Great: 7.9%  Epic: 3.3%
```

All five bands are in use, which is a real improvement over `Good:80%,
Poor:0%`. But 50–80 % of evenings now read "Decent", so the *ordering* got
better while the *labelling* got worse.

**Do not fix this by hand-tuning SCORE_THRESHOLDS.** Hand-tuned constants are
how the engine got into this state. The fix is Phase 5 — percentile-map the raw
score against multi-year climatology at the user's own location, so the bands
mean the same thing in Tel Aviv and London. Phase 5 is therefore no longer
optional polish; it is a prerequisite for putting Phase 1 in front of users.

### Phase 5 result: band shares are now identical across climates

> **Superseded.** This section records what percentile display achieved and is
> kept because the measurement is real. The display was later changed to an
> absolute score for the reason given in the section after it.

Verified with `scripts/evaluate.py --days 365`:

```
                 Poor   Decent   Good  Great   Epic    >=70
TelAviv         29.9%   38.1%  20.0%   9.0%   3.0%    9.0%
London          29.9%   38.1%  20.0%   9.0%   3.0%    9.0%
SanFrancisco    29.9%   38.1%  20.0%   9.0%   3.0%    9.0%
```

Against the design targets of 30 / 38 / 20 / 9 / 3. Identical in three very
different climates, which is the entire point: the same word now means the same
rarity everywhere. The "Decent covers 75 %" guardrail failure is gone.

Cold start is the one visible seam. Building a location's climatology takes
~11 s, so it runs in the background while a global reference curve stands in.
For Tel Aviv — the tightest, highest distribution of the three, and so the worst
case — the same evening reads 41.9 "Decent" on the reference curve and 21.9
"Poor" once local history lands. Both are defensible (raw 45.5 sits just under
Tel Aviv's median of 47.4, in a distribution dense enough that two points is 30
percentiles) but the shift is visible if you refresh. The API exposes
`climatology_is_local` and the UI says so explicitly rather than hiding it. The
curve is cached for 30 days and persists across restarts, so in practice a user
meets this once.

Still outstanding: `moisture` remains pinned at ~100 on dry days even after
losing the rain penalty, because what is left of it is only surface humidity.
Phase 3's column moisture is what gives that weight something real to measure.

---

### Why percentile display was reversed

Percentile display has a failure mode that only shows once the physics starts
improving: **it is self-normalising.** A fix that makes the engine more right
about a *kind* of evening lifts every evening of that kind, so the rank of any
one of them barely moves.

Measured, on a real evening the user had rated by eye at roughly 75/100. The
horizon-strip fix (near-field blocking, commit `14a00eb`) raised its raw score
from 48.9 to 57.4 — a substantial, correct improvement. Its displayed score went
**30.9 → 30.6**. The improvement was invisible because it applied to every other
clear Tel Aviv evening too.

That is fatal for a model still being tuned: the number a user rates against
cannot be one that hides the effect of tuning. It also makes the score answer
the wrong question. A user glancing at the app wants "how good will the sky be",
not "how does tonight compare with the other 364 evenings here".

So:

- **The displayed score is the raw physics score.** `SCORE_THRESHOLDS` are
  absolute again — 85 / 72 / 55 / 38 — derived from the pooled raw distribution
  across the three cities, not hand-picked, and deliberately *not* equal-frequency.
  `GO_OUTSIDE_THRESHOLD = 75.0`.
- **The percentile survives as context**, shown beneath the number, and is now
  **seasonal**: ranked against evenings within ±45 days of the same day of year
  (`SEASON_WINDOW_DAYS`, wrapping at New Year, falling back to the full curve
  below `MIN_SEASONAL_SAMPLES`). This resolves the open question Phase 5 left —
  the app can now say "better than 51 % of August evenings here" instead of
  reading "Poor" for seven straight late-August days.

Band shares are consequently no longer identical across climates, **which is the
point**:

```
                 Poor    Good    Epic   go-outside
TelAviv          4.9%   53.2%    1.6%      14.0%
London          25.5%   31.0%    2.5%      20.3%
SanFrancisco    21.6%   31.0%    6.6%      23.3%
```

Tel Aviv genuinely almost never has a bad evening and rarely a spectacular one;
London and San Francisco have both. Forcing those three climates to the same
histogram was an artefact of the display, not a property of their skies.

**Consequence for the remaining phases:** Phases 4 and 7 sharpen the raw score.
Under percentile display their effect would have been largely absorbed. It will
now show.

---

## Part 1 — What the model does today

> **As of 2026-08-25, before any of this plan landed.** Parts 1 and 2 are the
> frozen baseline the diagnosis was made against; they are deliberately not
> updated as phases complete. For the current engine, read the module docstring
> in `scoring_engine.py` and Part 4 below.

`ScoringEngine` (`backend/app/services/scoring_engine.py`) scores four
snapshots around sunset (−15m, sunset, +15m, +30m) and aggregates them.

Each snapshot gets four components, weighted-averaged:

| Component | Weight | Inputs |
|---|---|---|
| `cloud_quality` | 0.42 | low/mid/high/total cloud %, sun elevation |
| `atmosphere` | 0.28 | visibility, AOD, surface RH |
| `moisture` | 0.20 | precip now, precip 3h, pressure/cloud/visibility trends, surface RH |
| `horizon` | 0.10 | user-supplied obstruction angle |

`score_window` then takes **max** across the four points, adds a consistency
bonus (≤ +3), subtracts a volatility penalty (≤ −8), clamps to [0,100].
`MLModel.blend` mixes in an ML score if one is loaded. Categories are fixed
cutoffs (Poor <30, Decent 30, Good 50, Great 65, Epic 80).

All weather comes from Open-Meteo at **one grid cell — the observer's**.

---

## Part 2 — Diagnosis

### Evidence: one year of scores, three cities

Scored every day 2025-08-01 → 2026-07-31 through the real engine:

```
TelAviv        n=365  mean= 64.2  p10= 61.1 p50= 61.4 p90= 81.1 p99= 87.4
               Good:80%  Epic:10%  Great:6%  Decent:1%  Poor:0%
London         n=365  mean= 61.2  p10= 41.3 p50= 61.6 p90= 83.2 p99= 88.2
               Good:40%  Decent:24%  Great:18%  Epic:15%  Poor:1%
SanFrancisco   n=365  mean= 61.4  p10= 40.7 p50= 61.3 p90= 83.9 p99= 87.4
               Good:54%  Decent:19%  Epic:15%  Great:10%  Poor:0%
```

Per-component mean, standard deviation, and share of the final score's
day-to-day variance:

```
TelAviv          mean    sd   weight   variance share   days pinned at 100
  cloud_quality  38.5  13.5    0.42        64.7%              0.0%
  atmosphere     52.1   1.8    0.28         5.8%              0.0%
  moisture       97.6  12.9    0.20        29.6%             92.6%
  horizon        91.3   0.0    0.10         0.0%              0.0%

London           mean    sd   weight   variance share   days pinned at 100
  cloud_quality  32.6  24.7    0.42        72.8%              0.0%
  atmosphere     50.1   4.0    0.28         7.8%              0.0%
  moisture       95.9  13.8    0.20        19.4%             81.9%
  horizon        91.3   0.0    0.10         0.0%              0.0%
```

### D1 — 38% of the weight carries ~6% of the information

`horizon` has **sd = 0.0**. It is a per-user constant. It holds 10% of the
weight and contributes exactly nothing to distinguishing tonight from
tomorrow. It belongs in the output as a viewing caveat, not in the score.

`atmosphere` has sd 1.8–4.0 against a 28% weight. Cause: the archive API
never returns `visibility` (verified — it returns `null`), so it defaults to
15 km on every historical day; and when AOD is missing the fallback is
`vis_score * 0.75`, derived from that same constant. Two of its three inputs
collapse to one frozen number.

`moisture` is pinned at 100 on 82–93% of days. It is a rain veto wearing a
20% weight.

Net: the model is a cloud-cover model with a rain switch, wrapped in three
components that mostly don't move.

### D2 — The score distribution is degenerate and inflated

Tel Aviv's p10 and p50 are both 61.x: **more than 40% of the year returns
essentially the same number**. On a cloudless dry evening, `moisture` = 100,
`horizon` = 91.3, `atmosphere` = 52.5 — 58% of the weight is literally
constant, and `cloud_quality` barely moves. The model says "61" for half the
year in a dry climate.

Meanwhile "Epic" fires on 10–15% of days — roughly 50 evenings a year — and
"Poor" fires on 0–1%. Five categories, two of them doing 90% of the work,
and the rarest label isn't rare.

This is the real reason `GO_OUTSIDE_THRESHOLD` was just moved 45 → 70 on this
branch. That change treats the symptom. The underlying scale is uncalibrated,
so no threshold on it can mean anything stable across locations.

### D3 — Zero-radius blindness (the biggest physical gap)

Every serious predictor agrees the dominant factor is whether light can
*reach* the clouds above you — i.e. whether the corridor toward the sunset
azimuth, 100–300 km away, is clear. Afterglow samples only the observer's
grid cell, so it cannot see this at all.

- Corfidi (NOAA SPC): the most spectacular sunsets are "solid decks of middle
  or high clouds that cover the entire sky except for a narrow clear strip
  near the horizon."
- SunsetHue: casts rays from the observer toward the sunset and walks them
  outward; states the ray approach is more accurate than "the conventional
  local grid-cell approach" — which is exactly what Afterglow does.
- US10459119 gives usable numbers for how far upstream obstruction matters:
  low clouds ~80–100 mi, mid ~140–200 mi, high ~250+ mi.
- Geometry: roughly 111 km of horizontal travel per degree of solar depression.

Concretely: an overcast deck 200 km west kills tonight's sunset, and Afterglow
currently scores that evening Epic if the sky overhead happens to be pretty.

### D4 — The aerosol term is backwards

`atmosphere_score` uses `bell_curve(aod, peak=0.18, sigma=0.15)`, so AOD 0.18
scores 100 and pristine air at AOD 0.03 scores ~56. The comment justifies it
as "the pink hour effect."

Corfidi is explicit that this is wrong: tropospheric aerosols "do *not*
enhance sky colors — they subdue them," and "clean air is the main ingredient
common to brightly colored sunrises and sunsets." Haze turns vivid oranges
and reds into "pale yellows and pinks." The nuance in the literature is about
*stratospheric* (volcanic) aerosol, which is not what an AOD-550 field at
ground level is measuring.

The response should be monotone decreasing, not a bell peaked in the middle.

### D5 — Moisture is measured at the level that matters least

Afterglow uses `relative_humidity_2m` only. SunsetWx weights moisture across
the whole troposphere from the surface to 200 hPa, with **upper levels
weighted most heavily and surface moisture weighted down substantially**,
because it's low-level moisture that blocks light rather than colors it.

Available and unused (verified against the live API):
- `relative_humidity_{700,500,300}hPa` — forecast API only
- `total_column_integrated_water_vapour` — **forecast *and* archive**
- `direct_normal_irradiance` — forecast *and* archive

TCWV being on both endpoints matters: it's the one column-moisture signal that
keeps the heatmap and the live forecast on the same footing.

### D6 — Afterglow timing is a fitted curve where real geometry exists

The afterglow boost is a Gaussian at −3° with σ = 2°, applied identically to
every cloud layer. Real Earth-shadow geometry:

```
h = (R + Hs)/cos(d) − R        Hs ≈ 3 km (screening height for red light)
```

giving roughly 1 km of shadow rise per degree² of depression. Cirrus at ~10 km
stays lit to ~3.2° depression (10–15 min after sunset); a 4 km altostratus
deck goes dark around 2°; low cloud goes dark almost immediately. The current
single bell applies cirrus timing to stratus. Per-layer illumination falls
straight out of the formula and would replace three hand-tuned constants.

### D7 — The ML branch was tried, failed, and was deliberately shelved

**This is established history, not a new finding.** ML was attempted, it did not
deliver anything of value, and it was switched off for that reason. Physics-only
in production is a deliberate engineering decision that the evidence below
supports. Nothing in this plan asks to revisit that decision.

What the numbers add is *why* it failed, which matters only because it tells us
what would have to change before anyone considers it again — and because the
disabled code path is still wired up in a way that can bite.

`trained_models/model_metadata.json`:

```json
"rmse": 33.19, "mae": 28.54, "spearman_r": -0.0266, "spearman_p": 0.5536
```

Spearman −0.027 at p = 0.55 on 499 validation rows is *no signal at all* —
slightly negative, indistinguishable from noise.

It cannot be otherwise, by construction. `scripts/build_and_train.py` takes
r/sunset posts, and for each post joins its **timestamp** to archive weather at
five fixed cities (SF, Tel Aviv, London, Sydney, Cape Town) — regardless of
where the photo was actually taken. The label is percentile-rank of upvotes.
So the model is asked to predict a Norwegian sunset's upvote count from Cape
Town's cloud cover at the same instant. There is nothing to learn.

So the past attempt failed for a specific, mechanical reason — a broken join —
rather than because the problem resists learning. That distinction is worth
recording, but it is **not** an argument to try again: fixing the join requires
geolocated labels, which is the most expensive item in Phase 0. ML stays off.

The one thing that does need action is that the *disabled* path is still armed:

1. `ML_BLEND_ALPHA = 0.4` (pinned in `backend/.env`, so it overrides
   `config.py`) means `final = 0.4·physics + 0.6·ml`. The comment reads
   "1.0 = pure physics", which is correct but easy to misread as
   "0.4 = mostly physics". If a `.joblib` ever lands next to that metadata,
   noise silently takes **60%** of the score.
2. `MLModel.load()` checks only that the file exists. It never looks at the
   metadata it just loaded, so a model with negative correlation loads happily.

Today nothing loads because no `.joblib` is deployed — the shelving worked. The
risk is only that the switch is easy to flip back on by accident.

### D8 — Confidence is invented; ensemble spread is free

`compute_confidence` is a stack of hand-picked bonuses and penalties around a
base of 60. Open-Meteo's ensemble endpoint returns `cloud_cover_member01…N`
(verified live). Spread across members at the sunset hour is an actual
measurement of forecast uncertainty, which is what the number claims to be.

### D9 — Minor, but worth knowing

- The four window points span 45 minutes and Open-Meteo is hourly, so all four
  usually land in the **same bucket**. The only thing that genuinely varies
  across the window is sun elevation. The consistency bonus therefore fires
  near-deterministically, and `score_window`'s max-of-four + bonus is a
  systematic upward push on what is effectively a single sample.
- Model choice is left at Open-Meteo's `auto`. Practitioners comparing these
  services rank ICON-based cloud forecasts above GFS/NAM-based ones; the
  `models=` parameter is free to set.

---

## Part 3 — Plan

Ordered so that each phase is verifiable when it lands. Phase 0 comes first
because without it every weight change is a guess — which is how the current
constants got here.

### Phase 0 — Build the measuring stick

Nothing else in this plan can be evaluated without ground truth and a metric.

1. **Evaluation harness** — `backend/scripts/evaluate.py`:
   - Spearman rank correlation against labels. Rank, not RMSE: the product
     question is "is tonight better than last night", not "is it a 73".
   - Score distribution + category histogram per location (the Part 2 table,
     as a regression test).
   - Reliability curve: of evenings scored ≥80, what fraction were actually good.
   - Guardrail test that fails CI if any category exceeds a share bound
     (e.g. Epic > 5%) or if any location's p10–p50 span collapses below a few
     points.
2. **Ground truth**, cheapest first:
   - *Self-labelling*: one tap in the app to rate the evening 1–5, stored with
     the location and the full snapshot. ~100 evenings gives usable signal for
     one location, and it's the user's own taste, which is the actual target.
   - *Webcam scoring*: horizon-facing public webcams, sample frames from
     sunset−10m to sunset+25m, score saturation/warm-hue mass. Scales to
     hundreds of days × dozens of sites, gives geographic spread.
   - *Geotagged photos*: Flickr/Reddit posts **with EXIF lat/lon and time**, so
     the weather join is to the real location. This is the fix for D7's fatal
     flaw and the only version of the existing dataset worth keeping.

**Deliverable:** a number. Today's model has a measured Spearman ρ against
real labels — currently unknown — and every phase below reports its own.

### Phase 1 — Neutralise the dead weight (fast, low risk)

Independent of ground truth; justified by the variance table alone.

- Drop `horizon` from the weighted average. Report it as a separate viewing
  caveat in the response ("your horizon is 12° obstructed — you'll lose the
  last 8 minutes"), or apply it as a small multiplier. It cannot inform a
  day-to-day score with sd 0.
- Stop letting `moisture` spend 20% of the weight to say "it isn't raining".
  Split it: an explicit **precipitation veto** (multiplicative, near-zero score
  in active rain) plus a genuine **column-moisture** term (below).
- Reweight around what actually discriminates. Starting point for tuning:
  cloud/illumination ~0.60, column moisture ~0.20, air clarity ~0.20 — then let
  Phase 0's harness tune it rather than intuition.

### Phase 2 — The light corridor (biggest single accuracy gain)

The one change that adds a mechanism the model currently cannot see at all.

- Compute the sunset azimuth from `astral`.
- Sample ~6 points along that azimuth at 50/100/150/200/300/400 km.
  Open-Meteo takes **comma-separated coordinate lists, up to 1000 locations
  per request** (verified), so this is **one extra HTTP call**, not six —
  important given the recent rate-limit work in `8319826`.
- Corridor transmittance = product over samples of `(1 − blocking_i)`, where
  each sample's blocking uses the layer appropriate to its distance
  (per US10459119: low cloud matters out to ~150 km, mid to ~320 km, high
  beyond). Process high → low so low cloud takes precedence.
- Apply as a **multiplier on the colour term, not another weighted addend.**
  That is the physics: `colour = canvas × illumination`. No light through the
  corridor means no colour, however good the deck overhead is.
- Cache corridor samples on the existing `CACHE_COORD_DECIMALS` grid — nearby
  users share one fetch.

### Phase 3 — Fix the two mis-specified terms

- **Aerosol (D4):** replace the bell at 0.18 with a monotone decreasing
  response — pristine (≤0.05) ≈ 100, gentle decline through 0.15, sharper
  decay past 0.3, ~15 by 0.8. Cite Corfidi in the docstring so it doesn't get
  "fixed" back.
- **Moisture (D5):** replace surface-RH-only with a level-weighted column,
  upper levels weighted highest per SunsetWx. Use `relative_humidity_700hPa`
  / `500hPa` where available, `total_column_integrated_water_vapour` as the
  universal fallback so archive and forecast stay consistent. Keep only a
  small surface-RH penalty.
- **Archive visibility (D1):** stop defaulting to a constant 15 km. Derive
  clarity from TCWV + AOD, both of which the archive actually returns.
  Consider `direct_normal_irradiance` at the sunset hour as a direct measure
  of slant-path extinction — it's available on both endpoints and is close to
  a physical read on "is light getting through".

**Outcome, measured.** `scripts/evaluate.py --days 365`, three cities:

| | moisture variance share (before → after) | atmosphere sd |
|---|---|---|
| Tel Aviv | 2.8 % → 17.6 % | 13.9 |
| London | 4.0 % → 16.3 % | 11.4 |
| San Francisco | 5.0 % → 18.2 % | 7.2 |

All distribution guardrails now pass. Three things came out of the work that
were not in the plan:

- **Pressure-level humidity is unusable.** The archive returns null for every
  `relative_humidity_*hPa` level (checked across 72 consecutive hours, with and
  without `models=era5`). Scoring it on the forecast path would make live
  predictions incomparable with the climatology they are ranked against, so
  TCWV — available on both endpoints — carries the component alone.
- **Historical days were being scored with a different atmosphere term.** The
  archive path passed no air-quality data, so every historical day fell through
  to the humidity proxy while live forecasts used measured AOD. The air-quality
  endpoint serves a year of history in ONE request; it is now wired into the
  range path and both single-date archive paths.
- **Cached climatologies outlive a scoring change.** Curves persist to disk for
  30 days, so after this phase moved the median from ~47 to ~53, warm locations
  were still ranking new scores against the old distribution — silently, since
  the number stays plausible. The cache key now carries `SCALE_VERSION`, and
  `REFERENCE_QUANTILES` has a regenerate-me note for the same reason.

**Open question this surfaced — since resolved.** Calibration ranked an evening
against the whole year, so a seasonal low read as a run of "Poor" days —
late-August Tel Aviv was seven straight. That is physically right (Mediterranean
summer is hazy and cloudless; the drama is in winter fronts) but was the wrong
product answer. Ranking now happens within a ±45-day seasonal window, and the
rank is context rather than the headline number. See "Why percentile display was
reversed".

### Phase 4 — Real geometry for afterglow timing

**Partly done.** The clear-sky pathway (Part 4) already peaks on real solar
depression rather than a fitted curve. What remains is the per-layer part below,
which is the half that deletes tuned constants.

- Replace the −3°/σ2° bell with the Earth-shadow screening height
  `h = (R+Hs)/cos(d) − R`, `Hs ≈ 3 km`.
- Per-layer illumination: a layer is lit while the shadow height is below its
  top. Use `geopotential_height` at pressure levels, or representative heights
  (low 1 km, mid 4 km, high 9 km) as a first cut.
- This deletes three tuned constants, and gets "cirrus glows for 20 minutes,
  stratus for two" for free instead of by hand.

### Phase 5 — Calibrate the scale

- Replace max-of-window + bonus with an **illumination-weighted integral**
  across the window. The four points sample a curve; integrate it rather than
  taking the max and adding a fudge.
- **Percentile-map the raw score onto the displayed 0–100 using multi-year
  climatology at the user's own location**, so Epic means "top ~3% of evenings
  *here*". This is what makes the categories honest in both Tel Aviv and
  London, fixes the p10 = p50 collapse, and turns `GO_OUTSIDE_THRESHOLD` from a
  knob into a definition.

### Phase 6 — Make the ML shelving explicit in code

ML already failed in practice and was turned off. This phase makes that decision
**visible and hard to undo by accident** — it is not a step toward re-enabling it.

- Set `ML_BLEND_ALPHA = 1.0` in `backend/.env` (which currently pins `0.4`) and
  in `config.py`. A path that is off should read as off.
- Gate `MLModel.load()` on a metadata quality threshold — refuse to load unless
  `spearman_r` clears something like 0.15. A model that measured −0.027 should
  not be loadable at all.
- Move `trained_models/model_metadata.json` to `data/dead/` with a short README:
  what was tried, that it produced no signal, and the broken location join that
  explains it. Right now the only record of a failed experiment is a metrics
  blob that reads like a working model.

**Not proposed:** retraining. The prerequisite is geolocated labels, and until
Phase 0 has produced those *and* the physics work in Phases 1–5 has been
measured, there is no case for spending effort here. Revisit only if Phase 0
ground truth exists and physics has visibly plateaued below it — and treat that
as a fresh decision with its own evidence, not a step this plan schedules.

### Phase 7 — Better inputs

- Set `models=icon_seamless` explicitly rather than relying on `auto`.
- Derive confidence from **ensemble spread** — `cloud_cover_member01…N` at the
  sunset hour — instead of the hand-built heuristic. Cache aggressively; the
  ensemble endpoint is heavier than the deterministic one.

---

## Suggested order

As planned: Phase 0 → Phase 6 (the safety half) → Phase 1 → Phase 2 → Phase 3 →
Phase 4 → Phase 5 → Phase 7.

As actually executed: 0 → 6 → 1 → 2 → 5 → 3 → **3.5** → 5-revised → (4, 7
remaining). Phase 5 moved earlier because Phase 1 deflated the scale and left the
category labels meaningless in the meantime. Phase 3.5 was not in the plan at
all — it came from a user photograph of a cloudless Tel Aviv sunset that the
engine scored as a failure.

**Remaining order, and why:** Phase 7 next, because ensemble spread is the only
outstanding item that human labels cannot adjudicate — it replaces an invented
confidence number with a measured one. Phase 4 after ~15 labels exist, because it
is a tuning phase, and tuning against one label is fitting to noise. An earlier
over-tune of the clear-sky ceiling (raising it to 95, which pushed Tel Aviv's p50
to 80 and p90 to 87) was caught only by a distribution guardrail, not by
judgement.

Phase 6 is a five-minute change that makes an already-made decision permanent,
so it goes early and then stops being a topic. Phase 2 is the largest accuracy
win. Phase 5 is what the user will actually *feel*, because it's what makes
"Epic" mean something.

The through-line: **every accuracy gain in this plan comes from physics and
calibration, not from learning.** That is a deliberate consequence of the ML
attempt having already failed.

---

## Part 4 — Beauty is plural (Phase 3.5, unplanned)

### Where this came from

Not from the diagnosis. It came from a photograph: a Tel Aviv beach sunset on
2026-08-23, cloudless, the whole sky in colour bands, which the engine scored
30.9. The user's note was that summer sunsets there *are* beautiful, "they just
look different and mostly with no clouds", and then, decisively:

> don't try to optimize to a certain combination of params

That is an architectural objection, not a tuning request. The engine had one
notion of a good sky — lit cloud, ideally mid/high, ideally 40–70 % cover — and
everything else was scored as a deficient version of it. A cloudless sky was not
a different kind of evening; it was a failed one.

### What replaced it

`cloud_quality` is no longer a formula. It is the combination of five
**pathways**, each scored on its own terms, each a genuine physical route to a
coloured sky:

| Pathway | What it is | Ceiling |
|---|---|---|
| `lit_cloud` | The classic: underlit mid/high cloud | 100 |
| `twilight_gradient` | Clear-sky colour banding — Belt of Venus, anti-twilight arch; needs clean, dry air and an open horizon | 78 |
| `crepuscular` | Sun rays through broken cloud | 62 |
| `breaking_storm` | Cloud clearing fast at sunset — the rarest and most dramatic | 96 |
| `horizon_band` | A lit strip beneath an otherwise solid deck | 58 |

The ceilings are not equal, and shouldn't be: a breaking storm can be the best
sunset of the year, a band under a deck is a pleasant surprise. Combination is
**soft-max, not weighted average**:

```python
best = max(scores)
secondary = sum(all others) / 100
lift = MULTI_PATHWAY_LIFT * min(secondary, 1.5) * (1 - best / 100)
score = clamp(best + lift)
```

The best route carries the evening; other active routes lift it, and the lift
shrinks as the best score approaches its ceiling. An average would have done the
opposite — punished a sky that was spectacular in exactly one way, which is what
most spectacular skies are.

### Why this is not just five more tuned constants

The guardrail is a **win-distribution check** in `scripts/evaluate.py`: if any
single pathway wins more than 92 % of evenings, the architecture has collapsed
back into one recipe wearing five hats. Measured across the three cities:

```
twilight_gradient  41.7 – 76.2 %
lit_cloud          19.5 – 36.5 %
crepuscular         3.0 – 14.9 %
breaking_storm      0.3 –  1.1 %
horizon_band        1.1 –  7.3 %
```

Different climates are carried by different pathways, which is the claim the
architecture makes and the thing that would have been invisible under a single
formula.

The winner is also **exposed to the UI** (`dominant_pathway`), because it changes
the advice as much as the score does: a gradient evening peaks later than a
lit-cloud one, and you look at a different part of the sky.

### Known weaknesses

- **`crepuscular` is the weakest.** Cloud fraction is a poor proxy for
  *brokenness* — 50 % cover can be one solid sheet over half the sky or a
  hundred gaps. This is a data problem, not a tuning problem; it needs a
  texture or variance field the API does not expose.
- **`breaking_storm` needed rescuing once.** Five sub-unit factors multiplied
  together capped it at 27 against a ceiling of 96, so it never won an evening.
  The data was there (16 candidate evenings in 121 London days, with cloud
  trends of −69 % and −76 %); the arithmetic was wrong. Now a geometric mean
  over three evidence factors. **Multiplication is for gates; a geometric mean
  is for accumulating evidence.**
- **`horizon_band` silently assumed an open corridor.** The corridor factor
  defaulted to 1.0 when unmeasured, which for a pathway *defined* by light
  reaching under a deck is the one default guaranteed to be wrong. Now
  `Optional[float]`, scoring 0 without a measurement.

### The near-field fix that came out of the same evening

Scoring the corridor at ~340 km answers "is light reaching the cloud above me".
It does not answer "what does the strip of horizon I am looking at look like" —
that strip is 30–150 km away. Adding a near-field reading of the corridor
samples already being fetched (`NEAR_FIELD_MAX_KM = 200`) took that evening from
48.9 to 57.4 raw. It is also what exposed the self-normalising percentile
problem, because the displayed score moved 30.9 → 30.6.

## Sources

- [The Colors of Twilight and Sunset — Stephen F. Corfidi, NOAA SPC](https://www.spc.noaa.gov/publications/corfidi/sunset/)
- [About the Model — SunsetWx](https://sunsetwx.com/about-the-model/)
- [Sunsethue whitepaper](https://sunsethue.com/whitepaper) and [Predicting Sunset Quality](https://sunsethue.com/blog/predict-sunset)
- [US10459119B2 — System and method for predicting sunset vibrancy](https://patents.google.com/patent/US10459119B2/en)
- [The Science of Red Skies — Absurdly Optimized](https://www.absurdlyoptimized.com/outdoors/sunsets/)
- [Predicting Sunrise and Sunset Colors — Stephen Bay](https://stephenbayphotography.com/blog/predicting-sunrise-and-sunset-colors/)
- [Open-Meteo docs](https://open-meteo.com/en/docs) · [multi-location requests](https://openmeteo.substack.com/p/weather-data-for-multiple-locations)
