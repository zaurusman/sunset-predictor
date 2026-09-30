"""Print a fresh VAPID key pair in the base64url form pywebpush and browsers expect.

Usage: python scripts/generate_vapid_keys.py
Put VAPID_PUBLIC_KEY / VAPID_PRIVATE_KEY in Render's env (and backend/.env for local).
"""
from cryptography.hazmat.primitives import serialization
from py_vapid import Vapid01
from py_vapid.utils import b64urlencode

v = Vapid01()
v.generate_keys()
private = b64urlencode(v.private_key.private_numbers().private_value.to_bytes(32, "big"))
public = b64urlencode(
    v.public_key.public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
)
print(f"VAPID_PUBLIC_KEY={public}")
print(f"VAPID_PRIVATE_KEY={private}")
