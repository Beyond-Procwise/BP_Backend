"""What is accepted as a logo. Decoded, not believed."""
import io

import pytest
from PIL import Image

from src.services import brand_logos as bl


def image(kind="PNG", size=(120, 40), colour=(20, 60, 120)):
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, format=kind)
    return out.getvalue()


@pytest.mark.parametrize("kind, mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_a_png_or_a_jpeg_is_a_logo(kind, mime):
    assert bl.inspect(image(kind)) == (mime, 120, 40)


@pytest.mark.parametrize("data, says", [
    (b"", "empty"),
    (b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>", "not a PNG or JPEG"),
    (b"GIF89a" + b"\x00" * 40, "not a PNG or JPEG"),
    (b"%PDF-1.4 not an image", "not a PNG or JPEG"),
    (b"\x89PNG\r\n\x1a\n" + b"junk" * 20, "not a PNG or JPEG"),
])
def test_anything_else_is_refused_whatever_it_calls_itself(data, says):
    with pytest.raises(ValueError, match=says):
        bl.inspect(data)


def test_a_gif_that_really_is_a_gif_is_still_not_a_logo():
    with pytest.raises(ValueError, match="PNG or JPEG"):
        bl.inspect(image("GIF"))


def test_a_truncated_image_is_refused():
    with pytest.raises(ValueError):
        bl.inspect(image()[:60])


def test_size_limits():
    with pytest.raises(ValueError, match="at most"):
        bl.inspect(b"x" * (bl.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="pixels"):
        bl.inspect(image(size=(bl.MAX_SIDE + 1, 2)))


def test_the_default_logo_cannot_be_removed_out_from_under_the_letterhead():
    with pytest.raises(bl.StateConflict, match="choose another"):
        bl.remove(None, 7, default_logo_id=7)
