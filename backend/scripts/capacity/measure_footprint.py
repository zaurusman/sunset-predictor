"""Cache bytes added per location by each kind of request (predict, forecast, heatmap)."""
import asyncio, os, sys, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import OUT
PKL = str(OUT / "footprint.pkl")
os.environ.update(CACHE_PERSIST_PATH=PKL, DATABASE_URL="", VAPID_PRIVATE_KEY="")
if os.path.exists(PKL): os.remove(PKL)
import httpx, tracemalloc
from asgi_lifespan import LifespanManager
from app.main import app
def size():
    app.state.prediction_service._weather._cache.flush() if hasattr(app.state.prediction_service._weather._cache, "flush") else None
    return os.path.getsize(PKL)/1e3 if os.path.exists(PKL) else 0
async def settle():
    while any("warm" in repr(t.get_coro()) for t in asyncio.all_tasks()): await asyncio.sleep(0.5)
async def main():
    locs=[(30.6,34.8),(32.8,35.0),(33.0,35.5)]
    async with LifespanManager(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            s0=size()
            for la,lo in locs:
                await c.post("/predict", json={"latitude":la,"longitude":lo}); await settle()
            s1=size(); print(f"predict(+climatology) per location: {(s1-s0)/3:.0f} KB pickled")
            for la,lo in locs:
                await c.post("/forecast", json={"latitude":la,"longitude":lo,"days":7})
            s2=size(); print(f"+forecast per location: {(s2-s1)/3:.0f} KB")
            for la,lo in locs:
                await c.get(f"/heatmap?lat={la}&lon={lo}&months=6")
            s3=size(); print(f"+heatmap per location: {(s3-s2)/3:.0f} KB")
            st = app.state.prediction_service._weather._cache._store
            tracemalloc.start(); x = pickle.loads(pickle.dumps(st)); cur,_=tracemalloc.get_traced_memory()
            print(f"in-RAM size of whole cache: {cur/1e6:.1f} MB for 3 full-use locations; pickle {s3/1e3:.1f} MB")
asyncio.run(main())
