"""Upload validation: type, size, duration, corruption, metadata stripping and path safety."""

import io

import pytest
from PIL import Image

from incident_reporter.config import Limits
from incident_reporter.media import MediaError, MediaStore, read_limited, validate_audio, validate_image

from .conftest import make_image, make_wav

L = Limits()


def code(fn, *a):
    with pytest.raises(MediaError) as e:
        fn(*a)
    return e.value.code


def test_valid_wav_is_decoded_with_duration_and_quality():
    r = validate_audio(make_wav(3.0), L)
    assert r.ext == "wav" and abs(r.meta["seconds"] - 3.0) < 0.05
    assert r.samples.dtype.name == "float32" and r.meta["rms_dbfs"] < 0


def test_audio_duration_limits():
    assert code(validate_audio, make_wav(0.4), L) == "audio_too_short"
    assert code(validate_audio, make_wav(3.0), Limits(audio_max_seconds=2.0)) == "audio_too_long"


def test_audio_type_is_sniffed_not_trusted():
    assert code(validate_audio, b"hello, this is a text file pretending to be audio", L) == "audio_type"
    assert code(validate_audio, make_image(), L) == "audio_type"


def test_corrupt_audio_is_rejected():
    wav = make_wav(3.0)
    broken = wav[:12] + b"\x00" * 40 + b"garbage" * 50  # valid RIFF/WAVE magic, destroyed body
    assert code(validate_audio, broken, L) in {"audio_corrupt", "audio_too_short", "audio_no_stream"}


def test_audio_size_limit_while_reading():
    with pytest.raises(MediaError) as e:
        read_limited(io.BytesIO(b"x" * 2000), 1000, "audio")
    assert e.value.code == "audio_too_large"
    assert code(read_limited, io.BytesIO(b""), 10, "audio") == "audio_empty"


def test_valid_image_is_reencoded_without_metadata():
    raw = make_image(exif_gps=True)
    assert Image.open(io.BytesIO(raw)).getexif().get(0x8825) is not None  # the upload really carried GPS
    r = validate_image(raw, L)
    assert r.meta["metadata_removed"] is True
    stored = Image.open(io.BytesIO(r.jpeg))
    assert not stored.getexif() and "exif" not in stored.info
    assert r.sha256_original != r.sha256_stored


def test_png_and_webp_accepted():
    assert validate_image(make_image(fmt="PNG"), L).meta["format"] == "PNG"
    assert validate_image(make_image(fmt="WEBP"), L).meta["format"] == "WEBP"


def test_bad_images_rejected():
    assert code(validate_image, b"GIF89a....", L) == "image_type"
    assert code(validate_image, make_image()[:600], L) == "image_corrupt"  # truncated JPEG
    assert code(validate_image, make_image(size=(40, 40)), L) == "image_too_small"
    assert code(validate_image, make_image(size=(3000, 3000)), Limits(image_max_pixels=1_000_000)) in {"image_too_many_pixels", "image_corrupt"}
    assert code(validate_image, make_image(), Limits(image_max_bytes=100)) == "image_too_large"


@pytest.mark.parametrize("bad_id", ["../etc", "inc_../../x", "inc_123", "INC_0123456789ab", "inc_0123456789ab/..", ""])
def test_media_store_rejects_unsafe_ids(tmp_path, bad_id):
    store = MediaStore(tmp_path)
    with pytest.raises(ValueError):
        store.save(bad_id, "image.jpg", b"x")


@pytest.mark.parametrize("bad_name", ["../image.jpg", "image.png", "audio.exe", "x/../../audio.wav"])
def test_media_store_rejects_unexpected_names(tmp_path, bad_name):
    store = MediaStore(tmp_path)
    with pytest.raises(ValueError):
        store.save("inc_0123456789ab", bad_name, b"x")


def test_media_store_writes_inside_root(tmp_path):
    store = MediaStore(tmp_path)
    store.save("inc_0123456789ab", "audio.wav", b"data")
    p = store.path("inc_0123456789ab", "audio.wav")
    assert p.read_bytes() == b"data" and tmp_path.resolve() in p.parents
