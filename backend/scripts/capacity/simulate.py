"""30-day, minute-resolution capacity simulation of Afterglow on its free tiers.

Per-request costs were MEASURED against the real backend (measure_costs.py,
measure_footprint.py, measure_cache_stall.py) and the live Render service.
User-behaviour numbers are ASSUMPTIONS, listed in BEHAVIOUR. When the code
changes, re-measure and update the constants (or override them on the command
line) — see README.md.

Two architectures:
  v1  main as of 2026-10-01 (1dc85d8): cache re-pickled whole on every write,
      live objects in RAM for 30 days, each reader fetching its own forecast
      on a 2 h timer (1 h for the ensemble).
  v2  PR #29: compressed, budgeted cache; weather/aerosol/ensemble shared per
      location; forecasts refreshed when their model publishes a run.

    python simulate.py                           # v2, default DAU sweep
    python simulate.py --arch v1 50 400          # the old code
    python simulate.py --json --set p_explore=0.02
"""
import bisect
import json
import math
import random
from collections import defaultdict

ARCH = "v2"

# ── Measured: Open-Meteo weighted calls ──────────────────────────────────────
OM = dict(
    weather=1.3,             # 13 vars, ≤ 14 days
    aq=1.0,
    ensemble=1.0,
    corridor=6.0,            # six coordinates, billed as six calls
    climatology=55.8,        # new cell: archive months + aerosol (65.1 - 9.3)
    heatmap_cold_clim=107.8, # cell already has climatology
    heatmap_cold=163.6,      # nothing cached
    geocode=1.0,
)
OM["refresh_v1"] = OM["weather"] + OM["aq"] + OM["ensemble"] + OM["corridor"]   # 9.3

# ── Measured: CPU, local seconds (Render ≈ RENDER_FACTOR ×) ──────────────────
CPU_V1 = dict(predict_warm=0.017, predict_refresh=0.036, climatology=0.181,
              forecast_warm=0.106, forecast_refresh=0.106, heatmap_warm=0.019,
              heatmap_cold=0.55, heatmap_cold_clim=0.25, geocode=0.02,
              alert_check=0.11, push_send=0.012)
CPU_V2 = dict(predict_warm=0.0006, predict_refresh=0.036, climatology=0.134,
              forecast_warm=0.0024, forecast_refresh=0.066, heatmap_warm=0.0204,
              heatmap_cold=0.277, heatmap_cold_clim=0.20, geocode=0.029,
              alert_check=0.11, push_send=0.012)
RENDER_FACTOR = 7.0          # live warm predict 0.12 s vs 0.017 s local (v1)
SETS = dict(refresh=5, climatology=20, heatmap_cold=25, heatmap_cold_clim=18, frozen=2)
PICKLE_S_PER_MB = 0.006      # v1: whole-cache re-pickle on EVERY cache.set

# ── Measured: memory ─────────────────────────────────────────────────────────
# Pickled MB per entry group (v1) / compressed MB as held in RAM (v2).
MB_V1 = dict(clim=0.395, heatmap=0.809, short=0.03)
MB_V2 = dict(clim=0.081, heatmap=0.165, short=0.009)
RAM_PER_MB_V1 = 7.0          # live objects ~6x pickle + pickle.dump buffers
RAM_PER_MB_V2 = 1.0          # compressed bytes are what is held
HOT_DECODED_MB = 30          # v2: 64 most recent values kept decoded
CACHE_BUDGET_MB = 120        # v2: LRU eviction beyond this
RAM_BASE_MB = 115
RAM_LIMIT_MB = 512
TTL_LONG = 30 * 1440
STALE_GRACE = 12 * 60

# ── Refresh schedules (minutes) ──────────────────────────────────────────────
TTL_V1 = dict(forecast=120, ensemble=60)
TTL_AUTO = 120               # v2: `auto`-model fetches keep the 2 h TTL
MAX_AGE_V2 = 360             # FORECAST_MAX_AGE_SECONDS
# When each model's run becomes available, Israel local time (UTC+3), read from
# Open-Meteo's meta.json on 2026-10-01. A family is refreshed when ANY covering
# model publishes: icon_seamless = ICON-EU (3 h) + ICON global (6 h); its
# ensemble = ICON-EU-EPS (6 h) + ICON-EPS (12 h); aerosol = CAMS global (12 h)
# + CAMS Europe (24 h).
RUNS = dict(
    forecast=sorted({(40 + 180 * i) % 1440 for i in range(8)} | {29, 389, 749, 1109}),
    ensemble=sorted({15, 375, 735, 1095} | {402, 1122}),
    aq=sorted({106, 826} | {875}),
)

NEON_IDLE_MIN = 5
NEON_CU = 0.25
EDGE_REQ_FIRST, EDGE_REQ_REPEAT = 13, 4
KB_FIRST, KB_REPEAT = 203, 12

# ── Behaviour assumptions ────────────────────────────────────────────────────
BEHAVIOUR = dict(
    opens_per_dau=1.4,       # daily-glance app
    mau_per_dau=2.5,
    p_forecast=0.20, p_heatmap=0.04, p_other_date=0.10, p_explore=0.06,
    explore_world=0.5,       # half of explored places are abroad
    israel_cells=150, world_places=5000, zipf_s=1.05,
    p_subscribe=0.12, bells=1.3,
    epic_days=(4, 11, 19, 26), epic_cell_share=0.6, push_tap=0.45,
    sunset_min=18 * 60 + 15,  # Israel, early October (local)
)

LIMITS = dict(  # free-tier ceilings each column is compared against
    ram_mb=RAM_LIMIT_MB, om_max_day=10_000, om_max_hour=5_000, om_max_min=600,
    neon_cu_h=100, vercel_edge_req=1_000_000, vercel_gb=100,
)


def hour_profile(sunset):
    """Share of the day's opens in each minute (local time)."""
    w = []
    for m in range(1440):
        h = m / 60
        base = 0.15 if h < 6.5 else 1.0 if h < 22 else 0.4
        base += 1.2 * math.exp(-((h - 8) / 0.8) ** 2)           # morning glance
        base += 1.0 * math.exp(-((h - 13) / 1.0) ** 2)          # lunch
        d = (m - sunset) / 60
        base += 6.0 * math.exp(-((d + 0.6) / 0.7) ** 2)         # pre-sunset rush
        w.append(base)
    s = sum(w)
    return [x / s for x in w]


def zipf_sampler(n, s, rng):
    weights = [1 / (k ** s) for k in range(1, n + 1)]
    tot = sum(weights)
    cum, acc = [], 0.0
    for x in weights:
        acc += x / tot
        cum.append(acc)
    return lambda: min(bisect.bisect_left(cum, rng.random()), n - 1)


def next_run(now, family):
    """Absolute minute when the next run of *family* becomes available."""
    day, tod = divmod(now, 1440)
    times = RUNS[family]
    i = bisect.bisect_right(times, tod)
    return day * 1440 + times[i] if i < len(times) else (day + 1) * 1440 + times[0]


def simulate(dau, days=30, seed=1, arch=None):
    arch = arch or ARCH
    v2 = arch == "v2"
    CPU = CPU_V2 if v2 else CPU_V1
    MB = MB_V2 if v2 else MB_V1
    B = BEHAVIOUR
    rng = random.Random(seed)
    il = zipf_sampler(B["israel_cells"], B["zipf_s"], rng)
    world = zipf_sampler(B["world_places"], B["zipf_s"], rng)
    mau = int(dau * B["mau_per_dau"])
    home = [("il", il()) for _ in range(mau)]
    subs = [u for u in range(mau) if rng.random() < B["p_subscribe"]]
    sub_cells = defaultdict(int)
    for u in subs:
        sub_cells[home[u]] += 1
    prof = hour_profile(B["sunset_min"])
    cum, acc = [], 0.0
    for p in prof:
        acc += p
        cum.append(acc)
    S = B["sunset_min"]

    clim_exp, heat_exp = {}, {}
    valid = {}          # short-lived key -> minute it stops being current
    resident = {}       # short-lived key -> minute it leaves RAM
    frozen_day = {}
    neon_touch = []
    om_day, om_hour, om_min = defaultdict(float), defaultdict(float), defaultdict(float)
    om_by = defaultdict(float)
    cpu_min = defaultdict(float)
    worst_stall = peak_ram = peak_cache = 0.0
    edge_req = kb = opens_total = 0
    seen_users = set()
    push_spike = {}
    cur_cache = [0.0]

    def cache_mb(now):
        mb = MB["clim"] * sum(1 for e in clim_exp.values() if e > now)
        mb += MB["heatmap"] * sum(1 for e in heat_exp.values() if e > now)
        mb += MB["short"] * sum(1 for e in resident.values() if e > now)
        return mb

    def do_sets(now, n):
        nonlocal worst_stall
        if v2:
            return  # debounced background flush of compressed bytes: negligible
        cost = PICKLE_S_PER_MB * cur_cache[0]
        worst_stall = max(worst_stall, cost * RENDER_FACTOR)
        cpu_min[now] += n * cost

    def om(now, w, cat):
        om_by[cat] += w
        om_day[now // 1440] += w
        om_hour[now // 60] += w
        om_min[now] += w

    def fetch(now, key, cost, rule, cat="forecast refresh"):
        """Pay *cost* unless *key* is still current; return True if fetched."""
        if valid.get(key, -1) > now:
            return False
        if rule == "auto" or not v2:
            ttl = TTL_AUTO if rule == "auto" else TTL_V1.get(rule, 120)
            valid[key] = now + ttl
            life = ttl
        else:
            valid[key] = min(next_run(now, rule), now + MAX_AGE_V2)
            life = MAX_AGE_V2
        resident[key] = now + life + STALE_GRACE
        om(now, cost, cat)
        return True

    def ensure_clim(now, cell):
        if clim_exp.get(cell, -1) > now:
            return
        clim_exp[cell] = now + TTL_LONG
        om(now, OM["climatology"], "climatology (new place)")
        cpu_min[now] += CPU["climatology"]
        do_sets(now, SETS["climatology"])
        neon_touch.append(now)

    def predict(now, cell, k=0, warm=True):
        """/predict for date offset k (0 = tonight)."""
        if warm:
            ensure_clim(now, cell)
        if k == 0 and now % 1440 > S + 40:            # window over → frozen
            cpu_min[now] += CPU["predict_warm"]
            if frozen_day.get(cell) != now // 1440:
                frozen_day[cell] = now // 1440
                do_sets(now, SETS["frozen"])
                neon_touch.append(now)
            return
        if not v2:
            paid = fetch(now, (cell, "pred", k), OM["refresh_v1"] - OM["ensemble"], "forecast")
            paid |= fetch(now, (cell, "ens", k), OM["ensemble"], "ensemble")
        elif k <= 5:                                  # icon_seamless: shared bundle
            paid = fetch(now, (cell, "weather"), OM["weather"], "forecast")
            paid |= fetch(now, (cell, "aq"), OM["aq"], "aq")
            paid |= fetch(now, (cell, "corr", k), OM["corridor"], "forecast", "corridor")
            paid |= fetch(now, (cell, "ens"), OM["ensemble"], "ensemble")
        else:                                         # day 6: own `auto` fetch
            paid = fetch(now, (cell, "own", k), OM["weather"] + OM["aq"], "auto")
            paid |= fetch(now, (cell, "corr", k), OM["corridor"], "auto", "corridor")
            paid |= fetch(now, (cell, "ens"), OM["ensemble"], "ensemble")
        cpu_min[now] += CPU["predict_refresh"] if paid else CPU["predict_warm"]
        if paid:
            do_sets(now, SETS["refresh"])

    def forecast(now, cell):
        if not v2:
            paid = fetch(now, (cell, "fc"), OM["refresh_v1"] - OM["ensemble"], "forecast")
            paid |= fetch(now, (cell, "fc_ens"), OM["ensemble"], "ensemble")
        else:
            paid = fetch(now, (cell, "weather"), OM["weather"], "forecast")
            paid |= fetch(now, (cell, "aq"), OM["aq"], "aq")
            paid |= fetch(now, (cell, "ens"), OM["ensemble"], "ensemble")
            paid |= fetch(now, (cell, "corr_month"), OM["corridor"], "auto", "corridor")
        cpu_min[now] += CPU["forecast_refresh"] if paid else CPU["forecast_warm"]
        if paid:
            do_sets(now, SETS["refresh"] + 1)

    def heatmap(now, cell):
        if heat_exp.get(cell, -1) > now:
            cpu_min[now] += CPU["heatmap_warm"]
            return
        had = clim_exp.get(cell, -1) > now
        heat_exp[cell] = now + TTL_LONG
        kind = "heatmap_cold_clim" if had else "heatmap_cold"
        om(now, OM[kind], "heatmap (new place)")
        cpu_min[now] += CPU[kind]
        do_sets(now, SETS[kind])
        neon_touch.append(now)

    for day in range(days):
        epic = day in B["epic_days"]
        n_opens = int(dau * B["opens_per_dau"] * (1.5 if epic else 1.0))
        minutes = defaultdict(int)
        for _ in range(n_opens):
            minutes[min(bisect.bisect_left(cum, rng.random()), 1439)] += 1
        alert_min = ((S - 240) // 60) * 60 + 7
        if epic:   # Epic push 4 h before sunset, then a burst of taps
            for _ in range(int(len(subs) * B["epic_cell_share"] * B["push_tap"])):
                minutes[min(alert_min + 2 + int(rng.expovariate(1 / 3)), 1439)] += 1
        for tod in range(1440):
            now = day * 1440 + tod
            if tod % 60 == 7:                                   # hourly alert cron
                neon_touch.append(now)
                if tod == alert_min:
                    for cell in sub_cells:
                        predict(now, cell, 0, warm=False)
                    if epic:
                        sent = int(len(subs) * B["epic_cell_share"] * B["bells"])
                        cpu_min[now] += sent * CPU["push_send"]
                        push_spike[day] = sent
            if tod % 30 == 0:
                cur_cache[0] = cache_mb(now)
                held = min(cur_cache[0], CACHE_BUDGET_MB) if v2 else cur_cache[0]
                ram = RAM_BASE_MB + held * (RAM_PER_MB_V2 if v2 else RAM_PER_MB_V1)
                ram += HOT_DECODED_MB if v2 else 0
                peak_cache = max(peak_cache, cur_cache[0])
                peak_ram = max(peak_ram, ram)
            for _ in range(minutes.get(tod, 0)):
                opens_total += 1
                u = rng.randrange(mau)
                first = u not in seen_users
                seen_users.add(u)
                edge_req += EDGE_REQ_FIRST if first else EDGE_REQ_REPEAT
                kb += KB_FIRST if first else KB_REPEAT
                cell = home[u]
                if rng.random() < B["p_explore"]:
                    om(now, 2 * OM["geocode"], "geocode")
                    cpu_min[now] += 2 * CPU["geocode"]
                    cell = ("w", world()) if rng.random() < B["explore_world"] else ("il", il())
                predict(now, cell, 0)
                if rng.random() < B["p_other_date"]:
                    predict(now, cell, rng.randint(1, 6))
                if rng.random() < B["p_forecast"]:
                    forecast(now, cell)
                if rng.random() < B["p_heatmap"]:
                    heatmap(now, cell)

    awake, last_end = 0, -1
    for t in sorted(neon_touch):
        start, end = max(t, last_end), t + NEON_IDLE_MIN + 1
        if end > start:
            awake += end - start
        last_end = max(last_end, end)
    render_busy = {m: c * RENDER_FACTOR for m, c in cpu_min.items()}
    return dict(
        arch=arch, dau=dau, mau=mau, subs=len(subs), opens_month=opens_total,
        # Day 0 builds every location's climatology at once — in production
        # that only follows losing the durable cache — so the limit check
        # uses the steady-state peak day.
        om_max_day=round(max(om_day[d] for d in range(min(2, days - 1), days))),
        om_cold_start_day=round(om_day[0]),
        om_avg_day=round(sum(om_day.values()) / days),
        om_max_hour=round(max(om_hour.values())), om_max_min=round(max(om_min.values())),
        render_peak_busy_s_per_min=round(max(render_busy.values()), 1),
        render_min_overloaded=sum(1 for v in render_busy.values() if v > 60),
        worst_freeze_s=round(worst_stall, 2),
        cells_30d=sum(1 for e in clim_exp.values() if e > days * 1440),
        cache_mb=round(peak_cache, 1), ram_mb=round(peak_ram),
        neon_cu_h=round(awake / 60 * NEON_CU, 1), neon_h_per_day=round(awake / 60 / days, 1),
        vercel_edge_req=edge_req, vercel_gb=round(kb / 1e6, 2),
        push_max=max(push_spike.values(), default=0),
        om_month_by=dict(sorted(((k, round(v)) for k, v in om_by.items()), key=lambda kv: -kv[1])),
        om_days=[round(om_day[d]) for d in range(days)],
    )


def _apply(overrides):
    g = globals()
    for item in overrides:
        name, _, raw = item.partition("=")
        try:
            val = json.loads(raw)
        except ValueError:
            val = raw
        if name in BEHAVIOUR:
            BEHAVIOUR[name] = tuple(val) if isinstance(val, list) else val
        elif name in g and name.isupper():
            g[name] = val
        else:
            raise SystemExit(f"unknown parameter {name!r}")
    LIMITS["ram_mb"] = g["RAM_LIMIT_MB"]


def _row(r):
    def flag(k):
        v = r[k]
        return f"{v}{'!' if k in LIMITS and v > LIMITS[k] else ''}"
    return (f"{r['dau']:>6} {r['cells_30d']:>6} {flag('ram_mb'):>8} "
            f"{r['worst_freeze_s']:>7} {r['render_min_overloaded']:>7} "
            f"{r['om_avg_day']:>7} {flag('om_max_day'):>8} {flag('om_max_hour'):>7} "
            f"{flag('om_max_min'):>6} {flag('neon_cu_h'):>7} "
            f"{flag('vercel_edge_req'):>10} {r['push_max']:>6}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dau", nargs="*", type=int, help="daily active users to simulate")
    ap.add_argument("--arch", choices=("v1", "v2"), default=ARCH)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a constant (e.g. RENDER_FACTOR=10) or a BEHAVIOUR key")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--json", action="store_true", help="one JSON object per DAU level")
    a = ap.parse_args()
    _apply(a.set)
    levels = a.dau or [25, 50, 100, 200, 400, 800, 1500, 3000, 6000]
    if not a.json:
        print(f"arch={a.arch}")
        print("   DAU  cells   RAM_MB  freeze  ovrMin  OM/avg   OM/max  OM/hr  OM/min  NeonCUh  VercelEdge  push")
        print("  (! = over the free-tier limit; OM/max = busiest day after the cold start;")
        print("   freeze = worst event-loop stall on Render, s;")
        print("   ovrMin = minutes in the month Render needed >60 s of CPU per minute)")
    for d in levels:
        r = simulate(d, days=a.days, arch=a.arch)
        print(json.dumps(r) if a.json else _row(r), flush=True)
