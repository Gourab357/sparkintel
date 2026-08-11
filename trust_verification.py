"""
SparkIntel — Layer 2: Authenticity / Trust Verification Module
================================================================
Simulates the "Verify Trust" stage from the pitch deck's pipeline:
SEBI / brokers / exchanges digitally sign outgoing communications;
investors verify authenticity via a QR code or lookup ID against a
verification registry — independent of whether the content *looks*
like a deepfake or phishing message. This is Layer 2 ("is this really
from who it claims to be") sitting alongside Layer 1 ("does this look
synthetic/malicious").

Dependency-light on purpose (stdlib + cryptography + qrcode + Pillow +
opencv) so the exact same module runs in a Colab cell, inside FastAPI,
or inside the Streamlit demo without change.
"""

from __future__ import annotations

import base64
import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
import qrcode
from PIL import Image
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


# ---------------------------------------------------------------------------
# 1. Issuer identity (SEBI / a specific broker / an exchange)
# ---------------------------------------------------------------------------

class Issuer:
    """A signing identity: SEBI, a specific broker, or an exchange.

    In production each issuer's private key lives in a KMS/HSM, never in
    application code — this generates one in-memory so the notebook is
    self-contained.
    """

    def __init__(self, name: str, key_size: int = 2048):
        self.name = name
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
        self.public_key = self._private_key.public_key()

    def public_pem(self) -> str:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

    def sign(self, message: str) -> str:
        """Base64 RSA-PSS/SHA-256 signature over `message`."""
        sig = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()


def verify_signature(message: str, signature_b64: str, public_key) -> bool:
    try:
        public_key.verify(
            base64.b64decode(signature_b64),
            message.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 2. Verification registry — stand-in for the Postgres table in production
# ---------------------------------------------------------------------------

@dataclass
class VerificationRecord:
    record_id: str
    issuer: str
    message_hash: str
    signature: str
    channel: str
    issued_at: float
    revoked: bool = False


class VerificationRegistry:
    """In-memory stand-in for a `verified_communications` Postgres table."""

    def __init__(self):
        self._store: dict[str, VerificationRecord] = {}

    def register(self, issuer: Issuer, message: str, channel: str = "email") -> VerificationRecord:
        record_id = uuid.uuid4().hex[:12]
        message_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()
        signature = issuer.sign(message)
        rec = VerificationRecord(
            record_id=record_id, issuer=issuer.name, message_hash=message_hash,
            signature=signature, channel=channel, issued_at=time.time(),
        )
        self._store[record_id] = rec
        return rec

    def lookup(self, record_id: str) -> Optional[VerificationRecord]:
        return self._store.get(record_id)

    def revoke(self, record_id: str):
        if record_id in self._store:
            self._store[record_id].revoked = True

    def verify_message(self, record_id: str, message: str, public_keys: dict) -> dict:
        """The investor-facing check: does *this* message match what the
        issuer actually registered under this ID, with a valid signature?
        """
        rec = self.lookup(record_id)
        if rec is None:
            return {"status": "UNKNOWN_ID", "authentic": False,
                     "detail": "No such verification ID — likely not from a registered issuer."}
        if rec.revoked:
            return {"status": "REVOKED", "authentic": False,
                     "detail": "Issuer revoked this communication after it was sent."}
        message_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()
        if message_hash != rec.message_hash:
            return {"status": "CONTENT_MISMATCH", "authentic": False,
                     "detail": "Text does not match what the issuer registered — "
                                "possible tampering, or a real ID copy-pasted onto fake content."}
        pub = public_keys.get(rec.issuer)
        if pub is None or not verify_signature(message, rec.signature, pub):
            return {"status": "BAD_SIGNATURE", "authentic": False, "detail": "Signature does not verify."}
        return {"status": "VERIFIED", "authentic": True, "issuer": rec.issuer,
                 "issued_at": rec.issued_at, "detail": f"Genuinely issued by {rec.issuer}."}


# ---------------------------------------------------------------------------
# 3. QR code — what actually ships in the email / circular footer
# ---------------------------------------------------------------------------

def make_verification_qr(record: VerificationRecord, base_url: str = "https://verify.sparkintel.app/v") -> Image.Image:
    payload = f"{base_url}/{record.record_id}"
    qr = qrcode.QRCode(border=2, box_size=8)
    qr.add_data(payload)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def decode_qr(image: Image.Image) -> Optional[str]:
    """Decode a QR back to its payload using OpenCV (no extra system libs
    needed — unlike pyzbar, which requires libzbar0)."""
    import cv2
    arr = np.array(image.convert("RGB"))[:, :, ::-1].copy()  # RGB -> BGR
    data, _points, _ = cv2.QRCodeDetector().detectAndDecode(arr)
    return data or None


# ---------------------------------------------------------------------------
# 4. Lightweight invisible watermark (LSB) — for images / video thumbnails
# ---------------------------------------------------------------------------

def embed_watermark(image: Image.Image, payload: str) -> Image.Image:
    """Hide `payload` (e.g. a record_id — must be null-byte-free, which any
    hex/uuid string is) in the LSBs of the red channel."""
    img = image.convert("RGB").copy()
    w, h = img.size
    needed_bits = (len(payload) + 2) * 8  # +2 bytes = null terminator
    if needed_bits > w * h:
        raise ValueError(
            f"Image too small to hold this watermark: need {needed_bits} px, have {w * h}."
        )
    arr = np.array(img)
    bits = "".join(f"{b:08b}" for b in payload.encode("utf-8")) + "0" * 16
    flat = arr.reshape(-1, 3)
    for i, bit in enumerate(bits):
        flat[i, 0] = (flat[i, 0] & 0xFE) | int(bit)
    return Image.fromarray(flat.reshape(arr.shape))


def extract_watermark(image: Image.Image, max_chars: int = 64) -> str:
    arr = np.array(image.convert("RGB")).reshape(-1, 3)
    n_bits = min(len(arr), (max_chars + 2) * 8)
    bits = [str(arr[i, 0] & 1) for i in range(n_bits)]
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = int("".join(bits[i:i + 8]), 2)
        if byte == 0:
            break
        chars.append(chr(byte))
    return "".join(chars)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
