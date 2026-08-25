# Dead experiment — Reddit-labelled ML calibration model

**Outcome: no signal. Shelved. Do not resurrect as-is.**

`model_metadata.json` in this folder is the record of the one ML calibration
model ever trained for Afterglow. Its own metrics:

```
rmse: 33.19   mae: 28.54   spearman_r: -0.0266   spearman_p: 0.5536   n_val: 499
```

A rank correlation of −0.027 at p = 0.55 is indistinguishable from noise —
very slightly worse than guessing.

## Why it could not have worked

Two defects in the label pipeline (`scripts/build_and_train.py`):

1. **Selection bias (fatal).** Labels were percentile-ranked upvotes on r/sunset
   posts. Every row was a sky somebody chose to photograph, edit and post, so
   the dataset contained no bad evenings. The best such a model can learn is
   "which nice sunset got more upvotes" — mostly composition, gear and posting
   time — not "is tonight worth going outside for". Percentile-ranking does not
   fix this; there is simply no negative class.

2. **Location mismatch.** Each post's *timestamp* was joined to archive weather
   at five fixed cities (San Francisco, Tel Aviv, London, Sydney, Cape Town)
   regardless of where the photo was actually taken. The features described a
   different sky than the label. Reddit also strips EXIF on i.redd.it uploads,
   so the real coordinates are not recoverable after the fact.

## What would have to be true to try again

- Labels that include the **dull evenings** — a fixed observer or camera that
  reports every night, not only the photogenic ones.
- Features and labels describing the **same location**.
- Physics fixed first. A residual model on top of an engine that is still
  missing its dominant mechanism (the upstream light corridor) mostly learns to
  patch that gap, badly, from few noisy labels.

`POST /rate` now collects first-party ratings that satisfy the first two. See
`docs/scoring-v2-plan.md` for the full diagnosis and the conditions under which
this question gets reopened.

## Guardrail

`MLModel.load()` now refuses any model whose metadata reports
`spearman_r < ML_MIN_SPEARMAN` (default 0.15), and `ML_BLEND_ALPHA` is 1.0
(pure physics). Restoring the old artifact will no longer silently affect
predictions.
