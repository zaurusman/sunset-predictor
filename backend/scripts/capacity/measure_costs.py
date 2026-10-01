"""Measure real per-request cost of the Afterglow backend.

Runs the actual FastAPI app in-process (lifespan included, no DB), wraps the
shared httpx client to count every Open-Meteo request and its weighted cost,
and records CPU seconds per request. Each scenario uses a fresh location so
it is genuinely cold.

Makes ~100 real Open-Meteo requests (~450 weighted calls). Leaves the
resulting cache at $CAPACITY_OUT/cache.pkl for measure_cache_stall.py.
"""
import asyncio, json, os, resource, sys, time
from collections import Counter
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import OUT  # noqa: E402  (also puts backend/ on sys.path)
os.environ.update(
    CACHE_PERSIST_PATH=str(OUT / "cache.pkl"),
    DATABASE_URL="", VAPID_PRIVATE_KEY="", LOG_LEVEL="WARNING",
)
if os.path.exists(os.environ["CACHE_PERSIST_PATH"]):
    os.remove(os.environ["CACHE_PERSIST_PATH"])

import httpx
from asgi_lifespan import LifespanManager  # noqa

calls = []
_orig_get = httpx.AsyncClient.get


def weight(params):
    nloc = len(str(params.get("latitude", "0")).split(","))
    nvars = sum(len(str(params[k]).split(",")) for k in ("hourly", "daily", "current") if k in params)
    if "start_date" in params:
        days = (date.fromisoformat(params["end_date"]) - date.fromisoformat(params["start_date"])).days + 1
    else:
        days = int(params.get("forecast_days", 7)) + int(params.get("past_days", 0))
    return nloc * max(1.0, nvars / 10) * max(1.0, days / 14)


async def counting_get(self, url, *a, params=None, **kw):
    if "open-meteo" in str(url):
        calls.append((str(url).split("//")[1].split("/")[0], weight(params or {})))
    return await _orig_get(self, url, *a, params=params, **kw)

httpx.AsyncClient.get = counting_get

from app.main import app  # noqa: E402


async def settle(timeout=180):
    """Wait for background tasks (climatology warm) to finish."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()
                  and "durable" not in repr(t.get_coro())]
        busy = [t for t in others if "warm" in repr(t.get_coro()) or "_build" in repr(t.get_coro())]
        if not busy:
            return
        await asyncio.sleep(0.5)


async def run(client, label, method, path, **kw):
    n0 = len(calls)
    c0, w0 = time.process_time(), time.perf_counter()
    r = await client.request(method, path, timeout=300, **kw)
    cpu_req, wall = time.process_time() - c0, time.perf_counter() - w0
    n_req = len(calls) - n0
    await settle()
    cpu_total = time.process_time() - c0
    new = calls[n0:]
    out = dict(label=label, status=r.status_code, wall_s=round(wall, 2),
               cpu_req_s=round(cpu_req, 3), cpu_incl_bg_s=round(cpu_total, 3),
               om_calls_in_request=n_req, om_calls_total=len(new),
               om_weighted=round(sum(w for _, w in new), 1),
               by_host=dict(Counter(h for h, _ in new)))
    print(json.dumps(out), flush=True)
    return out


async def main():
    # Fresh, distinct cells (0.1° grid) per scenario so nothing is shared.
    A = (32.08, 34.78)   # Tel Aviv
    B = (31.25, 34.79)   # Be'er Sheva
    C = (29.56, 34.95)   # Eilat
    D = (33.82, 35.49)   # Beirut-ish
    today = date.today()
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            res = []
            p = lambda ll, d=None: {"latitude": ll[0], "longitude": ll[1], **({"target_date": str(d)} if d else {})}
            res.append(await run(c, "predict cold (new place, incl. bg climatology)", "POST", "/predict", json=p(A, today)))
            res.append(await run(c, "predict warm (same cell)", "POST", "/predict", json=p(A, today)))
            res.append(await run(c, "predict warm, neighbour 2km away", "POST", "/predict", json=p((A[0] + 0.02, A[1] + 0.01), today)))
            res.append(await run(c, "predict tomorrow (same cell)", "POST", "/predict", json=p(A, today + timedelta(days=1))))
            res.append(await run(c, "forecast 7d (cell warm from predict)", "POST", "/forecast", json={"latitude": A[0], "longitude": A[1], "days": 7}))
            res.append(await run(c, "forecast 7d warm", "POST", "/forecast", json={"latitude": A[0], "longitude": A[1], "days": 7}))
            res.append(await run(c, "heatmap 6mo (cell has climatology)", "GET", f"/heatmap?lat={A[0]}&lon={A[1]}&months=6"))
            res.append(await run(c, "heatmap 6mo warm", "GET", f"/heatmap?lat={A[0]}&lon={A[1]}&months=6"))
            res.append(await run(c, "heatmap 6mo fully cold place", "GET", f"/heatmap?lat={B[0]}&lon={B[1]}&months=6"))
            res.append(await run(c, "forecast 7d fully cold place", "POST", "/forecast", json={"latitude": C[0], "longitude": C[1], "days": 7}))
            res.append(await run(c, "geocode search", "GET", "/geocode?name=Haifa&count=8"))
            # Alert-style check: warm_climatology=False on a cold cell
            svc = app.state.prediction_service
            from app.schemas.prediction import PredictRequest
            n0 = len(calls); c0 = time.process_time()
            await svc.predict(PredictRequest(latitude=D[0], longitude=D[1], target_date=today), warm_climatology=False)
            print(json.dumps(dict(label="alert check (cold cell, no climatology)", om_calls_total=len(calls) - n0,
                                  om_weighted=round(sum(w for _, w in calls[n0:]), 1),
                                  cpu_req_s=round(time.process_time() - c0, 3))), flush=True)
            # CPU of a warm predict, averaged
            c0 = time.process_time()
            for _ in range(20):
                await c.post("/predict", json=p(A, today))
            print(json.dumps(dict(label="avg warm predict x20", cpu_s=round((time.process_time() - c0) / 20, 4))))
            c0 = time.process_time()
            for _ in range(5):
                await c.get(f"/heatmap?lat={A[0]}&lon={A[1]}&months=6")
            print(json.dumps(dict(label="avg warm heatmap x5", cpu_s=round((time.process_time() - c0) / 5, 4))))
            c0 = time.process_time()
            for _ in range(10):
                await c.post("/forecast", json={"latitude": A[0], "longitude": A[1], "days": 7})
            print(json.dumps(dict(label="avg warm forecast x10", cpu_s=round((time.process_time() - c0) / 10, 4))))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(json.dumps(dict(label="peak RSS MB", value=round(rss, 1))))

asyncio.run(main())
