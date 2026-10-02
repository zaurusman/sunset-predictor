"""The browser fetches Open-Meteo when our server can't.

Open-Meteo limits calls per IP address, and Render's free plan shares its
outbound addresses with other customers, whose usage can exhaust "our" daily
limit. Each visitor's browser has its own address and its own limit.

For tonight's /predict and the 7-day /forecast: when Open-Meteo refuses the
server (its daily/hourly limit), or the server's own share for non-tonight
work is used up (call_budget), the endpoint answers 503 with `client_fetch` —
the exact Open-Meteo URLs it still needs. The browser fetches them and
repeats the request with everything it has so far in `client_data`
(URL → JSON). The server reads those instead of downloading and does all the
scoring itself, so the score is the same as if it had downloaded the data.
Missing URLs are collected where fetches are independent, so it takes about
three rounds (weather + aerosol, corridor, ensemble).

Browser-supplied data never reaches another visitor: it is not written to the
shared cache (TTLCache.set skips it), and a request using it never joins or
starts a shared in-flight fetch (WeatherService._shared_fetch).
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Coroutine, Iterator, Optional

import httpx
from fastapi.responses import JSONResponse

# Responses the browser already fetched, by canonical URL (None: not in use).
client_data: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "client_data", default=None
)
# Set by the endpoints that can hand a fetch to the browser.
client_fetch_allowed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "client_fetch_allowed", default=False
)


class ClientFetchNeeded(BaseException):
    """The browser should fetch *urls* and retry. A BaseException so the
    handlers that turn a failed fetch into a fallback (proxy aerosol, no
    corridor, no ensemble spread) can't swallow it: the browser can get the
    real data, so no fallback should change the score."""

    def __init__(self, urls: list[str]) -> None:
        super().__init__(urls)
        self.urls = list(dict.fromkeys(urls))


async def collect(*coros: Coroutine[Any, Any, Any]) -> list[Any]:
    """Await each in turn; if some need the browser, ask for all of their URLs
    in one ClientFetchNeeded instead of one round per URL."""
    results: list[Any] = []
    missing: list[str] = []
    try:
        for coro in coros:
            try:
                results.append(await coro)
            except ClientFetchNeeded as need:
                missing += need.urls
                results.append(None)
    finally:
        for coro in coros[len(results) + 1:]:   # any other failure: not started
            coro.close()
    if missing:
        raise ClientFetchNeeded(missing)
    return results


@contextmanager
def browser_may_fetch(allowed: bool, supplied: Optional[dict[str, Any]]) -> Iterator[None]:
    """For one request: may the browser fetch, and what has it fetched so far."""
    allowed_token = client_fetch_allowed.set(allowed)
    data_token = client_data.set(supplied if allowed else None)
    try:
        yield
    finally:
        client_data.reset(data_token)
        client_fetch_allowed.reset(allowed_token)


def fetch_request(need: ClientFetchNeeded) -> JSONResponse:
    """The 503 that asks the browser to fetch *need.urls* and ask again."""
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Weather data provider is temporarily rate-limited for the server.",
            "client_fetch": need.urls,
        },
    )


def canonical_url(url: str, params: dict) -> str:
    """The exact URL a request goes to — the key both sides agree on."""
    return str(httpx.URL(url, params=params))
