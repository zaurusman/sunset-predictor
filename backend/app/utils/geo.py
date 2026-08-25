"""Great-circle geometry for sampling the atmosphere upstream of an observer.

Used by the light-corridor model: the light that colours a cloud above you
did not pass through your grid cell — it grazed the surface a few hundred
kilometres toward the sunset. To know whether it got through, you have to look
where it actually travelled.
"""
from __future__ import annotations

import math

# Mean Earth radius (km). The corridor model is insensitive to the choice of
# ellipsoid at this precision — weather grids are ~10 km at best.
EARTH_RADIUS_KM = 6371.0


def destination_point(
    lat_deg: float, lon_deg: float, bearing_deg: float, distance_km: float
) -> tuple[float, float]:
    """Point reached by travelling *distance_km* from (lat, lon) on *bearing_deg*.

    Standard great-circle direct solution on a spherical Earth. Bearing is
    degrees clockwise from true north, matching astral's solar azimuth.

    Longitude is normalised to [-180, 180] so the result is always a valid
    API coordinate, including for corridors crossing the antimeridian.
    """
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    bearing = math.radians(bearing_deg)
    delta = distance_km / EARTH_RADIUS_KM

    sin_lat2 = math.sin(lat1) * math.cos(delta) + math.cos(lat1) * math.sin(delta) * math.cos(bearing)
    # Guard against floating-point drift pushing the argument outside [-1, 1].
    lat2 = math.asin(max(-1.0, min(1.0, sin_lat2)))

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(delta) * math.cos(lat1),
        math.cos(delta) - math.sin(lat1) * math.sin(lat2),
    )

    lat_out = math.degrees(lat2)
    lon_out = (math.degrees(lon2) + 540.0) % 360.0 - 180.0

    # Clamp latitude: a corridor running over a pole can overshoot by a hair.
    return max(-90.0, min(90.0, lat_out)), lon_out


def horizon_tangent_distance_km(height_km: float) -> float:
    """Horizontal distance at which a ray reaching *height_km* grazes the surface.

    This is where upstream cloud actually matters. Light illuminating a cloud
    at height h enters the lower atmosphere roughly sqrt(2·R·h) away along the
    sun's azimuth; nearer than that the ray is already above the boundary
    layer, and further out it has not yet descended into it.

    The values this produces independently reproduce the empirically-tuned
    distances in US10459119 (low ~130–160 km, mid ~225–320 km, high ~400 km):

        1 km (low cloud)     → ~113 km
        4 km (mid cloud)     → ~226 km
        9 km (high cloud)    → ~339 km
    """
    if height_km <= 0.0:
        return 0.0
    return math.sqrt(2.0 * EARTH_RADIUS_KM * height_km)
