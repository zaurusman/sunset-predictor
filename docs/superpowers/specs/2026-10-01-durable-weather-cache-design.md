# Durable weather cache tier (Postgres)

## Why
`TTLCache` lives in memory and mirrors to a pickle in /tmp. Render's disk is
ephemeral, so every deploy/restart wipes it and each location rebuilds its
climatology from Open-Meteo archive data (~111 weighted calls per new
location). Those bursts are a suspected cause of the intermittent 503s (see
the Open-Meteo 503 investigation, PRs #21/#23).

## Scope
Persist only entries whose effective TTL is >= 1 day: climatology curves
(30 d), archive/aq/corridor months (30 d), frozen evenings (1 d), and hist
ranges (1 d). Short forecast/snapshot/ensemble entries stay memory-only, so
Neon's compute is not kept awake by constant writes.

## Design
- `app/utils/durable_cache.py`: `PostgresCacheTier(pool)` with
  `ensure_schema()`, `load_all(grace)`, `write_many(rows)`,
  `purge_expired(grace)`. Table
  `cache_entries(key text primary key, value bytea, expires_at timestamptz)`.
  It reuses the subscription store's asyncpg pool (exposed as `store.pool`).
- `TTLCache.attach_durable(tier)`: bulk-loads unexpired rows with one query,
  keeps the later expiry when a key is also in the pickle, purges expired rows,
  and starts one background writer.
- `set()` with TTL >= `DURABLE_MIN_TTL_SECONDS` enqueues
  `(key, pickled value, expires_at)`. The writer waits a short linger,
  drains the queue, dedupes by key and calls `write_many` once per burst.
  Failures are logged as warnings. The entry stays in memory.
- Expired rows are purged at attach time and lazily after a write when the
  last purge is older than a day. Purging never wakes the database on its own.
- `close_durable()` on shutdown flushes pending writes.
- Without DATABASE_URL, nothing changes.

## Testing
- Unit tests with an in-memory fake tier cover: the TTL threshold, batching,
  failure isolation, merge on load, the grace window, and pickle round trips
  of real value types.
- A Postgres contract test is gated on TEST_DATABASE_URL.
- Measurement: count Open-Meteo calls for Tel Aviv on a cold process after a
  simulated restart, with and without the durable tier.

## Plan
1. RED/GREEN: TTLCache durable hooks against the fake tier.
2. RED/GREEN: PostgresCacheTier against real local Postgres.
3. Wire into `main.py` lifespan (connect DB first, attach, close on shutdown).
4. Full suite, restart measurement, PR.
