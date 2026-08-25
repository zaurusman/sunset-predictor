# Afterglow scoring v2 — diagnosis and plan

Written 2026-08-25.

## Status

| Phase | State |
|---|---|
| 0 — Measuring stick | **Mostly done.** `POST /rate` + `GET /ratings/stats` collect first-party labels with their raw inputs; one-tap UI on the verdict card. `scripts/evaluate.py` is the committed harness — distribution stats, guardrails, and label correlation. Webcam labels not started. |
| 1 — Neutralise dead weight | **Done.** Horizon and precipitation are now multiplicative gates, not weighted components; weights are 0.60/0.25/0.15. |
| 2 — Light corridor | **Done.** Wired into predict, forecast and heatmap. |
| 3 — Aerosol + column moisture | Not started |
| 4 — Earth-shadow afterglow timing | Not started |
| 5 — Calibrate the scale | **Done.** Displayed score is a percentile against local climatology; band shares are fixed by construction. |
| 6 — Make ML shelving explicit | **Done.** `ML_BLEND_ALPHA = 1.0`, load-time quality gate, artifact moved to `data/dead/`. |
| 7 — Better inputs (ICON, ensemble confidence) | Not started |

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

## Part 1 — What the model does today

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

### Phase 4 — Real geometry for afterglow timing

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

Phase 0 → Phase 6 (the safety half) → Phase 1 → Phase 2 → Phase 3 → Phase 4 →
Phase 5 → Phase 7.

Phase 6 is a five-minute change that makes an already-made decision permanent,
so it goes early and then stops being a topic. Phase 2 is the largest accuracy
win. Phase 5 is what the user will actually *feel*, because it's what makes
"Epic" mean something.

The through-line: **every accuracy gain in this plan comes from physics and
calibration, not from learning.** That is a deliberate consequence of the ML
attempt having already failed.

---

## Sources

- [The Colors of Twilight and Sunset — Stephen F. Corfidi, NOAA SPC](https://www.spc.noaa.gov/publications/corfidi/sunset/)
- [About the Model — SunsetWx](https://sunsetwx.com/about-the-model/)
- [Sunsethue whitepaper](https://sunsethue.com/whitepaper) and [Predicting Sunset Quality](https://sunsethue.com/blog/predict-sunset)
- [US10459119B2 — System and method for predicting sunset vibrancy](https://patents.google.com/patent/US10459119B2/en)
- [The Science of Red Skies — Absurdly Optimized](https://www.absurdlyoptimized.com/outdoors/sunsets/)
- [Predicting Sunrise and Sunset Colors — Stephen Bay](https://stephenbayphotography.com/blog/predicting-sunrise-and-sunset-colors/)
- [Open-Meteo docs](https://open-meteo.com/en/docs) · [multi-location requests](https://openmeteo.substack.com/p/weather-data-for-multiple-locations)
