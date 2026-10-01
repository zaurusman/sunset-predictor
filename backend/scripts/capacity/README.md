# Capacity simulation

Estimates how many users Afterglow can handle on its free tiers, and which
limit breaks first:

- Render free: 512 MB of RAM and 0.1 CPU
- Open-Meteo free: 10k calls/day, 5k/hour, 600/minute
- Neon free: 100 CU-hours a month
- Vercel Hobby: 1M edge requests a month

## Two steps

1. **Measure** what each request really costs, using the current code
   (scripts `measure_*.py`).
2. **Simulate** 30 days, minute by minute, with those costs (`simulate.py`).

`simulate.py` is pure Python and makes no network calls. Its constants are the
measured costs, and `BEHAVIOUR` holds the assumptions about how users behave.
Re-measure whenever the cache, the weather fetching or the scoring code
changes, then update the constants at the top of `simulate.py`.

```bash
cd backend
python scripts/capacity/simulate.py                       # default sweep of daily-user levels
python scripts/capacity/simulate.py 50 400 --json
python scripts/capacity/simulate.py --set p_explore=0.02 --set PICKLE_S_PER_MB=0
```

## Re-measuring

Use the backend venv. The scripts write their output to `$CAPACITY_OUT`, which
defaults to `<tmp>/afterglow-capacity`.

| Script | Measures | Simulator constants it feeds |
|---|---|---|
| `measure_costs.py` | Open-Meteo calls (raw and weighted) and CPU for each request type, cold and warm. Makes about 100 real Open-Meteo requests | `OM`, `CPU`, `SETS` |
| `measure_footprint.py` | Cache bytes added per location by predict, forecast and heatmap | `MB`, `RAM_PER_PICKLED_MB` |
| `measure_cache_stall.py N` | A cold /predict on a cache preloaded with N copies of the cache from `measure_costs.py`. Records CPU, `cache.set` count, the longest event-loop freeze and peak RSS. Run `measure_costs.py` first | Checks the RAM and freeze model |

`compare_responses.py BASE_BACKEND NEW_BACKEND` checks that a change leaves
every user-facing response identical (scores, categories, windows, reasons).
It runs /predict, /forecast and /heatmap against live Open-Meteo for both
backends, one after the other, and diffs the JSON. Run it before merging
anything that touches fetching or caching.

`RENDER_FACTOR` is the ratio between Render and local CPU time. To set it, time
a few cached requests against the live backend, subtract the round trip to
`/health`, and divide by the local CPU time for the same request:

```bash
curl -s -o /dev/null -w "%{time_total}\n" https://sunset-predictor-b8ig.onrender.com/health
curl -s -o /dev/null -w "%{time_total}\n" -X POST -H 'Content-Type: application/json' \
  -d '{"latitude":32.08,"longitude":34.78}' https://sunset-predictor-b8ig.onrender.com/predict
```

## Results as of 2026-10-01 (before the cache fix)

| Limit | Breaks at about |
|---|---|
| Render RAM (cache keeps every location for 30 days, OOM) | 60–75 distinct locations a month, about 30–55 daily users |
| Render CPU freezes (whole cache re-pickled on every write) | from about 25 daily users |
| Open-Meteo 10k calls/day | about 350 daily users (Render's outbound IP is shared, so the real number is lower) |
| Vercel 1M edge requests/month | about 5,000 daily users |
| Neon 100 CU-hours | not reached |

What drives memory and Open-Meteo usage is the number of **distinct
0.1° cells** people look at, not the number of users. Users in the same cell
share one cached forecast.
