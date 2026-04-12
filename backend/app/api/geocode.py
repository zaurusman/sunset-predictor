"""GET /geocode — proxy to Open-Meteo Geocoding API.

Calls are made server-side so the browser isn't blocked by CORS.
"""
from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(tags=["geocoding"])


@router.get("/geocode", summary="Search for locations by name")
async def geocode(
    name: str = Query(..., min_length=1, description="Place name to search for"),
    count: int = Query(8, ge=1, le=20),
    request: Request = None,
) -> dict:
    settings = request.app.state.settings
    url = f"{settings.OPEN_METEO_GEOCODING_URL}/search"
    params = {"name": name, "count": count, "language": "en", "format": "json"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail="Geocoding API error") from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail="Geocoding API unreachable") from exc

    return resp.json()
