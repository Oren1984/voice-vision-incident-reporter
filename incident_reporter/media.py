"""Upload validation and safe media storage.

Every check runs server-side regardless of what the browser did:
- size limit enforced while reading (the upload is never held beyond the limit),
- content sniffed from magic bytes (the client's filename and MIME type are not trusted),
- audio fully decoded (corrupt or truncated files fail), duration bounds checked,
- images: pixel-count limit before decoding (decompression bombs), full decode, minimum size,
  EXIF orientation applied, then re-encoded **without metadata** (removes GPS and device data).
Files are stored under server-chosen names in a per-incident directory; user-supplied names are
never used as paths.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Limits

INCIDENT_ID_RE = re.compile(r"^inc_[0-9a-f]{12}$")
AUDIO_TYPES = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg", "mp3": "audio/mpeg", "webm": "audio/webm", "m4a": "audio/mp4"}
IMAGE_TYPES = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
ASR_RATE = 16_000


class MediaError(ValueError):
    """A user-correctable problem with an uploaded file. The message is safe to show."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def read_limited(stream, limit: int, what: str) -> bytes:
    buf = bytearray()
    while True:
        chunk = stream.read(1 << 16)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > limit:
            raise MediaError(f"{what}_too_large", f"The {what} file is larger than {limit // (1024 * 1024)} MB.")
    if not buf:
        raise MediaError(f"{what}_empty", f"The {what} file is empty.")
    return bytes(buf)


def sniff_audio(data: bytes) -> str | None:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if data[4:8] == b"ftyp":
        return "m4a"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    return None


@dataclass
class AudioResult:
    data: bytes
    ext: str
    sha256: str
    samples: np.ndarray  # float32 mono 16 kHz, kept in memory only for processing
    meta: dict = field(default_factory=dict)


def decode_audio(data: bytes) -> np.ndarray:
    import av

    with av.open(io.BytesIO(data), mode="r") as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise MediaError("audio_no_stream", "The file does not contain an audio track.")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=ASR_RATE)
        parts = []
        for frame in container.decode(streams[0]):
            for f in resampler.resample(frame):
                parts.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            parts.append(f.to_ndarray().reshape(-1))
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


def audio_quality(x: np.ndarray) -> dict:
    """Level, silence share and clipping — signals for 'unclear speech', not a speech detector."""
    if x.size == 0:
        return {"rms_dbfs": None, "silent_share": 1.0, "clipped_share": 0.0}
    win = int(0.03 * ASR_RATE)
    n = max(1, x.size // win)
    frames = x[: n * win].reshape(n, win) if x.size >= win else x.reshape(1, -1)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    overall = 20 * np.log10(float(np.sqrt((x**2).mean())) + 1e-12)
    return {
        "rms_dbfs": round(float(overall), 1),
        "silent_share": round(float((db < -45).mean()), 3),
        "clipped_share": round(float((np.abs(x) > 0.99).mean()), 4),
    }


def validate_audio(data: bytes, limits: Limits) -> AudioResult:
    if len(data) > limits.audio_max_bytes:
        raise MediaError("audio_too_large", f"The audio file is larger than {limits.audio_max_bytes // (1024 * 1024)} MB.")
    ext = sniff_audio(data)
    if ext is None:
        raise MediaError("audio_type", "Unsupported audio format. Use WAV, MP3, M4A, OGG, WebM or FLAC.")
    try:
        samples = decode_audio(data)
    except MediaError:
        raise
    except Exception as e:  # PyAV raises many error types for damaged input
        raise MediaError("audio_corrupt", "The audio file could not be decoded (damaged or incomplete).") from e
    seconds = samples.size / ASR_RATE
    if seconds < limits.audio_min_seconds:
        raise MediaError("audio_too_short", f"The recording is shorter than {limits.audio_min_seconds:g} s.")
    if seconds > limits.audio_max_seconds:
        raise MediaError("audio_too_long", f"The recording is longer than {limits.audio_max_seconds:g} s.")
    meta = {"format": ext, "bytes": len(data), "seconds": round(seconds, 2), **audio_quality(samples)}
    return AudioResult(data, ext, hashlib.sha256(data).hexdigest(), samples, meta)


@dataclass
class ImageResult:
    jpeg: bytes  # sanitised re-encode, no metadata
    sha256_original: str
    sha256_stored: str
    image: Image.Image
    meta: dict = field(default_factory=dict)


def sniff_image(data: bytes) -> bool:
    return data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n" or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")


def validate_image(data: bytes, limits: Limits) -> ImageResult:
    if len(data) > limits.image_max_bytes:
        raise MediaError("image_too_large", f"The image file is larger than {limits.image_max_bytes // (1024 * 1024)} MB.")
    if not sniff_image(data):
        raise MediaError("image_type", "Unsupported image format. Use JPEG, PNG or WebP.")
    Image.MAX_IMAGE_PIXELS = limits.image_max_pixels
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt, (w, h) = probe.format, probe.size
            if fmt not in IMAGE_TYPES:
                raise MediaError("image_type", "Unsupported image format. Use JPEG, PNG or WebP.")
            if w * h > limits.image_max_pixels:
                raise MediaError("image_too_many_pixels", "The image has too many pixels.")
            probe.verify()
        with Image.open(io.BytesIO(data)) as im:
            had_exif = bool(im.info.get("exif")) or bool(im.getexif())
            im.load()  # full decode: truncated files fail here
            im = ImageOps.exif_transpose(im)
            rgb = im.convert("RGB")
    except MediaError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, RuntimeError, Image.DecompressionBombError, ValueError) as e:
        raise MediaError("image_corrupt", "The image could not be decoded (damaged, incomplete or too large).") from e
    if min(rgb.size) < limits.image_min_side:
        raise MediaError("image_too_small", f"The image is smaller than {limits.image_min_side} px on one side.")
    out = io.BytesIO()
    rgb.save(out, format="JPEG", quality=90)  # no exif/icc passed -> metadata dropped
    jpeg = out.getvalue()
    meta = {"format": fmt, "bytes": len(data), "width": rgb.size[0], "height": rgb.size[1], "metadata_removed": had_exif}
    return ImageResult(jpeg, hashlib.sha256(data).hexdigest(), hashlib.sha256(jpeg).hexdigest(), rgb, meta)


class MediaStore:
    """media/<incident_id>/{audio.<ext>, image.jpg}. Paths are built only from validated ids."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, incident_id: str) -> Path:
        if not INCIDENT_ID_RE.match(incident_id):
            raise ValueError("invalid incident id")
        d = (self.root / incident_id).resolve()
        if d.parent != self.root:
            raise ValueError("path escapes media root")
        return d

    def save(self, incident_id: str, name: str, data: bytes) -> str:
        if name not in {f"audio.{e}" for e in AUDIO_TYPES} | {"image.jpg"}:
            raise ValueError("unexpected media name")
        d = self._dir(incident_id)
        d.mkdir(exist_ok=True)
        path = d / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return name

    def path(self, incident_id: str, name: str) -> Path:
        if name not in {f"audio.{e}" for e in AUDIO_TYPES} | {"image.jpg"}:
            raise ValueError("unexpected media name")
        p = (self._dir(incident_id) / name).resolve()
        if p.parent.parent != self.root:
            raise ValueError("path escapes media root")
        return p
