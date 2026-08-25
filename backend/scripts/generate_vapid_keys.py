#!/usr/bin/env python3
"""Generate a VAPID keypair for Web Push.

Run once, then put the output in the environment (Render dashboard in
production, backend/.env locally):

    python scripts/generate_vapid_keys.py

The keypair identifies this server to browser push services. Rotating it
invalidates every existing subscription — browsers hold the public key and the
push service checks the signature against it — so everyone would have to opt in
again. Generate once and keep it.

The PRIVATE key is a credential. Never commit it.
"""
from __future__ import annotations

import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def _b64url(raw: bytes) -> str:
    """Base64url without padding — the encoding the Web Push spec expects."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def main() -> None:
    private_key = ec.generate_private_key(ec.SECP256R1())

    # The application server key browsers receive: the uncompressed public
    # point (0x04 ‖ X ‖ Y), 65 bytes.
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    # pywebpush accepts the raw 32-byte private scalar in base64url form.
    private_bytes = private_key.private_numbers().private_value.to_bytes(32, "big")

    print("# Add these to backend/.env (local) or the Render dashboard (prod).")
    print("# VAPID_SUBJECT must be a mailto: or https: URL you control.")
    print()
    print(f"VAPID_PUBLIC_KEY={_b64url(public_bytes)}")
    print(f"VAPID_PRIVATE_KEY={_b64url(private_bytes)}")
    print("VAPID_SUBJECT=mailto:you@example.com")
    print()
    print("# And a secret for the dispatch endpoint:")
    print(f"NOTIFY_DISPATCH_SECRET={_b64url(__import__('os').urandom(32))}")


if __name__ == "__main__":
    main()
