"""Prove a change leaves every user-facing response identical.

Runs /predict (tonight .. day +6, default date, 3 days ago), the 7-day
/forecast and /heatmap against LIVE Open-Meteo for each backend, one after
the other, and diffs the JSON (timestamps dropped). Climatology is warmed
first so both rank against a finished curve. Paced to stay under Open-Meteo's
600 calls/min; takes ~6 minutes and ~1,500 weighted calls from this machine.

    git worktree add --detach /tmp/afterglow-base origin/main
    python scripts/capacity/compare_responses.py /tmp/afterglow-base/backend .

A model run published between the two passes can cause real differences;
re-run before concluding anything from a diff.
"""
import asyncio, json, os, subprocess, sys, time
from pathlib import Path

SNAPSHOT = r'''
import asyncio, json, os, sys
backend, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, backend)
os.environ.update(CACHE_PERSIST_PATH="", DATABASE_URL="", VAPID_PRIVATE_KEY="",
                  RATE_LIMIT_CLIENT_HOURLY_CALLS="0", OPEN_METEO_DAILY_SOFT_CAP="0")
from datetime import date, timedelta
import httpx
from asgi_lifespan import LifespanManager
from app.main import app
DROP = {"requested_at", "generated_at", "timestamp"}
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items() if k not in DROP}
    if isinstance(o, list): return [clean(v) for v in o]
    return o
LOCS = [(32.08, 34.78), (32.82, 34.99), (29.56, 34.95)]
async def main():
    res = {}
    async with LifespanManager(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=300) as c:
            today = date.today()
            for la, lo in LOCS:
                await c.post("/predict", json={"latitude": la, "longitude": lo})
                await asyncio.sleep(20)
            while any("_warm" in repr(t.get_coro()) for t in asyncio.all_tasks()):
                await asyncio.sleep(0.2)
            for la, lo in LOCS:
                r = await c.post("/forecast", json={"latitude": la, "longitude": lo, "days": 7})
                res[f"forecast {la},{lo}"] = (r.status_code, clean(r.json()))
                for k in range(7):
                    d = today + timedelta(days=k)
                    r = await c.post("/predict", json={"latitude": la, "longitude": lo, "target_date": str(d)})
                    res[f"predict {la},{lo} +{k}"] = (r.status_code, clean(r.json()))
                r = await c.post("/predict", json={"latitude": la, "longitude": lo})
                res[f"predict {la},{lo} default"] = (r.status_code, clean(r.json()))
                r = await c.post("/predict", json={"latitude": la, "longitude": lo, "target_date": str(today - timedelta(days=3))})
                res[f"predict {la},{lo} -3"] = (r.status_code, clean(r.json()))
                await asyncio.sleep(25)
            la, lo = LOCS[0]
            r = await c.get(f"/heatmap?lat={la}&lon={lo}&months=2")
            res[f"heatmap {la},{lo}"] = (r.status_code, clean(r.json()))
    json.dump(res, open(out, "w"), sort_keys=True, default=str)
asyncio.run(main())
'''


def diff(a, b, path=""):
    if type(a) != type(b):
        yield path, a, b
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            yield from diff(a.get(k), b.get(k), f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            yield path + "[len]", len(a), len(b)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from diff(x, y, f"{path}[{i}]")
    elif a != b:
        yield path, a, b


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _paths import OUT
    script = OUT / "snapshot_responses.py"
    script.write_text(SNAPSHOT)
    results = []
    for i, backend in enumerate(sys.argv[1:]):
        if i:
            time.sleep(60)  # let Open-Meteo's per-minute window reset
        out = OUT / f"responses_{i}.json"
        subprocess.run([sys.executable, str(script), str(Path(backend).resolve()), str(out)],
                       check=True, stderr=subprocess.DEVNULL)
        results.append(json.loads(out.read_text()))
    a, b = results
    bad = [k for k, v in {**a, **b}.items() if v[0] != 200]
    diffs = [(k, p, x, y) for k in sorted(a) for p, x, y in diff(a[k], b.get(k))]
    print(f"responses compared: {len(a)}   non-200: {bad or 'none'}")
    for k, p, x, y in diffs[:20]:
        print(f"DIFF {k}{p}: {str(x)[:60]} -> {str(y)[:60]}")
    print(f"differing fields: {len(diffs)}")
    sys.exit(1 if diffs or bad else 0)


if __name__ == "__main__":
    main()
