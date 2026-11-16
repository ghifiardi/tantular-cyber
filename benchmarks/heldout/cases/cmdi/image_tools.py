"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import subprocess


def make_thumbnail(source_path: str, extra_options: str) -> None:
    """Create a 200px thumbnail next to the source image."""
    command = "convert " + source_path + " -resize 200x200 " + extra_options + " thumb.png"
    subprocess.call(command, shell=True)
