"""PROOF OF CONCEPT — the browser fetches Open-Meteo when our server can't.

Open-Meteo limits calls per IP address, and Render's free plan shares its
outbound addresses with other customers, whose usage can exhaust "our" daily
limit. Each visitor's browser has its own address and its own limit.

For tonight's /predict only: when Open-Meteo refuses the server, the endpoint
answers 503 with `client_fetch` — the exact Open-Meteo URL it needed. The
browser fetches it and repeats the request with the responses it has so far
in `client_data` (URL → JSON). The server then reads those instead of
downloading, and does all the scoring itself, so the score is the same as if
it had downloaded the data. A few rounds: weather, aerosol, corridor,
ensemble.

Data supplied by a browser is never written to the shared cache (TTLCache.set
skips it), so one visitor can't feed made-up weather to others.
"""
from __future__ import annotations

import contextvars
from typing import Any, Optional

import httpx

# Responses the browser already fetched, by canonical URL (None: not in use).
client_data: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "client_data", default=None
)
# Set by the endpoints that can hand a fetch to the browser (tonight's /predict).
client_fetch_allowed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "client_fetch_allowed", default=False
)


class ClientFetchNeeded(BaseException):
    """The browser should fetch *url* and retry. A BaseException so the
    handlers that turn a failed fetch into a fallback (proxy aerosol, no
    corridor, no ensemble spread) can't swallow it: the browser can get the
    real data, so no fallback should change tonight's score."""

    def __init__(self, url: str) -> None:
        super().__init__(url)
        self.url = url


def canonical_url(url: str, params: dict) -> str:
    """The exact URL a request goes to — the key both sides agree on."""
    return str(httpx.URL(url, params=params))
