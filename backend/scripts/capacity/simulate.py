"""30-day, minute-resolution capacity simulation of Afterglow on its free tiers.

All per-request costs below were MEASURED on 2026-10-01 against the real
backend code (measure_costs.py, measure_footprint.py, measure_cache_stall.py)
and the live Render service. User-behaviour numbers are ASSUMPTIONS, listed in
BEHAVIOUR. When the code changes, re-measure and update the constants (or
override them on the command line) — see README.md.

    python simulate.py                         # default DAU sweep, table
    python simulate.py 50 400 --json           # JSON lines
    python simulate.py --set p_explore=0.02 --set PICKLE_S_PER_MB=0
"""
import math, random, sys, json
from collections import defaultdict

# ── Measured costs ────────────────────────────────────────────────────────────
OM = dict(            # Open-Meteo weighted calls
    predict_refresh=9.3,     # today/other date, cell already has climatology
    forecast_refresh=9.3,
    new_cell=65.1,           # first predict in a cell: 9.3 + climatology build
    heatmap_cold_clim=107.8, # cell has climatology
    heatmap_cold=163.6,      # nothing cached
    heatmap_refresh=3.0,     # current partial month after its TTL
    geocode=1.0,
)
CPU = dict(           # local CPU seconds (Mac); Render ≈ RENDER_FACTOR × this
    predict_warm=0.017, predict_refresh=0.036, new_cell=0.217,
    forecast_warm=0.106, forecast_refresh=0.106,
    heatmap_warm=0.019, heatmap_cold=0.55, heatmap_cold_clim=0.25,
    geocode=0.02, alert_check=0.11, push_send=0.012,
)
RENDER_FACTOR = 7.0          # live warm predict 0.12 s vs 0.017 s local
SETS = dict(predict_refresh=5, forecast_refresh=6, new_cell=20,
            heatmap_cold=25, heatmap_cold_clim=18, heatmap_refresh=2, frozen=2)
PICKLE_S_PER_MB = 0.006      # whole-cache re-pickle on EVERY cache.set (local)
MB = dict(clim=0.395, heatmap=0.809, short=0.03)  # pickled MB per entry group
RAM_PER_PICKLED_MB = 6.0     # in-RAM objects vs pickle bytes (22.2 / 3.7)
RAM_BASE_MB = 115
RAM_LIMIT_MB = 512
TTL_SHORT = 2 * 60           # minutes
SHORT_RESIDENT = 14 * 60     # 2 h TTL + 12 h stale grace stays in RAM
TTL_LONG = 30 * 1440
NEON_IDLE_MIN = 5
NEON_CU = 0.25
EDGE_REQ_FIRST, EDGE_REQ_REPEAT = 13, 4
KB_FIRST, KB_REPEAT = 203, 12

# ── Behaviour assumptions ─────────────────────────────────────────────────────
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
    cum, acc = [], 0
    for x in weights:
        acc += x / tot
        cum.append(acc)
    import bisect
    return lambda: min(bisect.bisect_left(cum, rng.random()), n - 1)


def simulate(dau, days=30, seed=1):
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
    S = B["sunset_min"]

    clim_exp, heat_exp, short_exp, frozen_day = {}, {}, {}, {}
    neon_touch = []
    om_day, om_hour, om_min = defaultdict(float), defaultdict(float), defaultdict(float)
    cpu_min = defaultdict(float)
    worst_stall, peak_ram, peak_pickle = 0.0, 0.0, 0.0
    edge_req = kb = opens_total = 0
    seen_users = set()
    push_spike = {}

    def pickle_mb(now):
        mb = MB["clim"] * sum(1 for e in clim_exp.values() if e > now)
        mb += MB["heatmap"] * sum(1 for e in heat_exp.values() if e > now)
        mb += MB["short"] * sum(1 for e in short_exp.values() if e + SHORT_RESIDENT - TTL_SHORT > now)
        return mb

    cur_pickle = [0.0]

    def do_sets(now, n):
        nonlocal worst_stall
        cost = n * PICKLE_S_PER_MB * cur_pickle[0]
        worst_stall = max(worst_stall, PICKLE_S_PER_MB * cur_pickle[0] * RENDER_FACTOR)
        cpu_min[now] += cost

    def om(now, w):
        om_day[now // 1440] += w; om_hour[now // 60] += w; om_min[now] += w

    def ensure_clim(now, cell):
        if clim_exp.get(cell, -1) > now:
            return False
        clim_exp[cell] = now + TTL_LONG
        om(now, OM["new_cell"] - OM["predict_refresh"])
        cpu_min[now] += CPU["new_cell"] - CPU["predict_refresh"]
        do_sets(now, SETS["new_cell"])
        neon_touch.append(now)
        return True

    def predict(now, cell, key="today"):
        tod = now % 1440
        ensure_clim(now, cell)
        if key == "today" and tod > S + 40:          # viewing window over → frozen
            if frozen_day.get(cell) == now // 1440:
                cpu_min[now] += CPU["predict_warm"]; return
            frozen_day[cell] = now // 1440
            cpu_min[now] += CPU["predict_warm"]; do_sets(now, SETS["frozen"])
            neon_touch.append(now); return
        k = (cell, key)
        if short_exp.get(k, -1) > now:
            cpu_min[now] += CPU["predict_warm"]; return
        short_exp[k] = now + TTL_SHORT
        om(now, OM["predict_refresh"]); cpu_min[now] += CPU["predict_refresh"]
        do_sets(now, SETS["predict_refresh"])

    for day in range(days):
        epic = day in B["epic_days"]
        mult = 1.5 if epic else 1.0
        n_opens = int(dau * B["opens_per_dau"] * mult)
        # bucket opens by minute
        minutes = defaultdict(int)
        for _ in range(n_opens):
            r, acc, m = rng.random(), 0.0, 0
            # inverse CDF via cached cumulative
            minutes[_inv(prof, r)] += 1
        # epic push spike, 4 h before sunset (alert run fires at :07 of that hour)
        alert_min = ((S - 240) // 60) * 60 + 7
        if epic:
            for cell, n in sub_cells.items():
                pass
            n_push_users = len(subs) * B["epic_cell_share"]
            taps = int(n_push_users * B["push_tap"])
            for _ in range(taps):
                m = alert_min + 2 + int(rng.expovariate(1 / 3))
                minutes[min(m, 1439)] += 1
        for tod in range(1440):
            now = day * 1440 + tod
            if tod % 60 == 7:                                   # hourly alert cron
                neon_touch.append(now)
                if tod == alert_min:
                    for cell in sub_cells:
                        cpu_min[now] += CPU["alert_check"] * 0.3
                        if short_exp.get((cell, "today"), -1) <= now:
                            short_exp[(cell, "today")] = now + TTL_SHORT
                            om(now, OM["predict_refresh"]); do_sets(now, SETS["predict_refresh"])
                    if epic:
                        sent = int(len(subs) * B["epic_cell_share"] * B["bells"])
                        cpu_min[now] += sent * CPU["push_send"]
                        push_spike[day] = sent
            if tod % 30 == 0:
                cur_pickle[0] = pickle_mb(now)
                peak_pickle = max(peak_pickle, cur_pickle[0])
                peak_ram = max(peak_ram, RAM_BASE_MB + RAM_PER_PICKLED_MB * cur_pickle[0]
                               + 1.0 * cur_pickle[0])          # + pickle.dump memo/buffers
            for _ in range(minutes.get(tod, 0)):
                opens_total += 1
                u = rng.randrange(mau)
                first = u not in seen_users
                seen_users.add(u)
                edge_req += EDGE_REQ_FIRST if first else EDGE_REQ_REPEAT
                kb += KB_FIRST if first else KB_REPEAT
                cell = home[u]
                if rng.random() < B["p_explore"]:
                    om(now, 2 * OM["geocode"]); cpu_min[now] += 2 * CPU["geocode"]
                    cell = ("w", world()) if rng.random() < B["explore_world"] else ("il", il())
                predict(now, cell)
                if rng.random() < B["p_other_date"]:
                    predict(now, cell, key=f"d{rng.randint(1, 6)}")
                if rng.random() < B["p_forecast"]:
                    k = (cell, "fc")
                    if short_exp.get(k, -1) > now:
                        cpu_min[now] += CPU["forecast_warm"]
                    else:
                        short_exp[k] = now + TTL_SHORT
                        om(now, OM["forecast_refresh"]); cpu_min[now] += CPU["forecast_refresh"]
                        do_sets(now, SETS["forecast_refresh"])
                if rng.random() < B["p_heatmap"]:
                    if heat_exp.get(cell, -1) > now:
                        cpu_min[now] += CPU["heatmap_warm"]
                    else:
                        had = clim_exp.get(cell, -1) > now
                        heat_exp[cell] = now + TTL_LONG
                        kind = "heatmap_cold_clim" if had else "heatmap_cold"
                        om(now, OM[kind]); cpu_min[now] += CPU[kind]; do_sets(now, SETS[kind])
                        neon_touch.append(now)

    # Neon awake minutes = union of [touch, touch + idle]
    awake, last_end = 0, -1
    for t in sorted(neon_touch):
        start, end = max(t, last_end), t + NEON_IDLE_MIN + 1
        if end > start:
            awake += end - start
        last_end = max(last_end, end)
    render_busy = {m: c * RENDER_FACTOR for m, c in cpu_min.items()}
    peak_busy = max(render_busy.values())
    over = sum(1 for v in render_busy.values() if v > 60)
    cells_30d = sum(1 for e in clim_exp.values() if e > days * 1440)
    return dict(
        dau=dau, mau=mau, subs=len(subs), opens_month=opens_total,
        om_max_day=round(max(om_day.values())), om_max_hour=round(max(om_hour.values())),
        om_max_min=round(max(om_min.values())),
        render_peak_busy_s_per_min=round(peak_busy, 1), render_min_overloaded=over,
        worst_freeze_s=round(worst_stall, 2),
        cells_30d=cells_30d, pickle_mb=round(peak_pickle, 1), ram_mb=round(peak_ram),
        neon_cu_h=round(awake / 60 * NEON_CU, 1), neon_h_per_day=round(awake / 60 / days, 1),
        vercel_edge_req=edge_req, vercel_gb=round(kb / 1e6, 2),
        push_max=max(push_spike.values(), default=0),
    )


_CUM = {}
def _inv(prof, r):
    key = id(prof)
    if key not in _CUM:
        acc, cum = 0, []
        for p in prof:
            acc += p; cum.append(acc)
        _CUM[key] = cum
    import bisect
    return min(bisect.bisect_left(_CUM[key], r), 1439)


LIMITS = dict(  # free-tier ceilings each column is compared against
    ram_mb=RAM_LIMIT_MB, om_max_day=10_000, om_max_hour=5_000, om_max_min=600,
    neon_cu_h=100, vercel_edge_req=1_000_000, vercel_gb=100,
)


def _apply(overrides):
    g = globals()
    for item in overrides:
        name, _, raw = item.partition("=")
        val = json.loads(raw) if raw[:1] in "[{\"" or raw.replace(".", "", 1).replace("-", "", 1).isdigit() else raw
        if name in BEHAVIOUR:
            BEHAVIOUR[name] = tuple(val) if isinstance(val, list) else val
        elif name in g and name.isupper():
            g[name] = val
        else:
            raise SystemExit(f"unknown parameter {name!r}")
    if "RAM_LIMIT_MB" in g:
        LIMITS["ram_mb"] = RAM_LIMIT_MB


def _row(r):
    flag = lambda k, v: f"{v}{'!' if k in LIMITS and v > LIMITS[k] else ''}"
    return (f"{r['dau']:>6} {r['cells_30d']:>6} {flag('ram_mb', r['ram_mb']):>8} "
            f"{r['worst_freeze_s']:>7} {r['render_min_overloaded']:>7} "
            f"{flag('om_max_day', r['om_max_day']):>8} {flag('om_max_hour', r['om_max_hour']):>7} "
            f"{flag('om_max_min', r['om_max_min']):>6} {flag('neon_cu_h', r['neon_cu_h']):>7} "
            f"{flag('vercel_edge_req', r['vercel_edge_req']):>10} {r['push_max']:>6}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dau", nargs="*", type=int, help="daily active users to simulate")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a constant (e.g. PICKLE_S_PER_MB=0) or a BEHAVIOUR key")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--json", action="store_true", help="one JSON object per DAU level")
    a = ap.parse_args()
    _apply(a.set)
    levels = a.dau or [25, 50, 100, 200, 400, 800, 1500, 3000, 6000]
    if not a.json:
        print("   DAU  cells   RAM_MB  freeze  ovrMin   OM/day  OM/hr  OM/min  NeonCUh  VercelEdge  push")
        print("  (! = over the free-tier limit; freeze = worst single event-loop stall on Render, s;")
        print("   ovrMin = minutes in the month Render needed >60 s of CPU per minute)")
    for d in levels:
        r = simulate(d, days=a.days)
        print(json.dumps(r) if a.json else _row(r), flush=True)
