"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import tarfile

PLUGIN_DIR = "/opt/app/plugins"


def install_plugin(archive_path: str) -> None:
    with tarfile.open(archive_path) as bundle:
        bundle.extractall(PLUGIN_DIR)


def install_plugin_filtered(archive_path: str) -> None:
    with tarfile.open(archive_path) as bundle:
        bundle.extractall(PLUGIN_DIR, filter="data")
