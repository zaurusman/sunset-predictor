"""Cold /predict on a cache preloaded with N copies of measure_costs.py's cache:
CPU, cache.set count, longest event-loop freeze, peak RSS.  Usage: N (default 15 ≈ 60 cells)
"""
import asyncio, os, sys, pickle, time, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import OUT
P=str(OUT / "big.pkl")
os.environ.update(CACHE_PERSIST_PATH=P, DATABASE_URL="", VAPID_PRIVATE_KEY="")
N=int(sys.argv[1]) if len(sys.argv) > 1 else 15
src = pickle.load(open(OUT / "cache.pkl", "rb"))
fmt2 = isinstance(src, dict) and src.get("format") == 2
entries = src["entries"] if fmt2 else src
# bytes(bytearray(b)): a distinct object per copy, or pickle would store it once.
big = {f"{k}#{i}": (bytes(bytearray(b)), e) for i in range(N) for k, (b, e) in entries.items()}
if not fmt2:  # the pre-fix layout held live objects, so give each copy its own
    blob = pickle.dumps(entries)
    big = {f"{k}#{i}": v for i in range(N) for k, v in pickle.loads(blob).items()}
with open(P, "wb") as f:
    pickle.dump({"format": 2, "entries": big} if fmt2 else big, f, protocol=pickle.HIGHEST_PROTOCOL)
del big
print(f"preloaded pickle {os.path.getsize(P)/1e6:.0f} MB", flush=True)
import httpx
from asgi_lifespan import LifespanManager
from app.utils.cache import TTLCache
sets=[0]; orig=TTLCache.set
def counting(self,*a,**k): sets[0]+=1; return orig(self,*a,**k)
TTLCache.set=counting
from app.main import app
async def main():
    async with LifespanManager(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            # measure event-loop freeze with a ticker
            gaps=[]; stop=False
            async def ticker():
                t=time.perf_counter()
                while not stop:
                    await asyncio.sleep(0.01); n=time.perf_counter(); gaps.append(n-t); t=n
            tk=asyncio.create_task(ticker())
            c0=time.process_time(); s0=sets[0]
            r=await c.post("/predict", json={"latitude":30.9,"longitude":35.1})
            while any("warm" in repr(t.get_coro()) for t in asyncio.all_tasks()): await asyncio.sleep(0.2)
            stop=True; await tk
            print(f"cold predict: status {r.status_code}, cache.set calls {sets[0]-s0}, CPU {time.process_time()-c0:.1f}s local, longest event-loop freeze {max(gaps):.2f}s local, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.0f} MB")
asyncio.run(main())
