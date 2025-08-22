"""
Image support for tests
"""

import base64
import io
import requests

from PIL import Image

from .logging_config import setup_logging

logger = setup_logging("images")


def display_base64_image(base64_string: str):
    """Display a base64 encoded image or URL using PIL.

    If input looks like an HTTP(S) URL, fetch and display it. Otherwise treat as base64.
    """
    try:
        if base64_string.startswith("http://") or base64_string.startswith("https://"):
            resp = requests.get(base64_string, timeout=10)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
            image.show()
            return

        if base64_string.startswith("data:image"):
            base64_data = base64_string.split(",")[1]
        else:
            base64_data = base64_string

        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes))
        image.show()
    except Exception as e:
        logger.error("Exception occured while displaying image: %s", e)
