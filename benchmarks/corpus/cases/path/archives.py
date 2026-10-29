"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import tarfile


def unpack(archive_path: str, dest: str) -> None:
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest)


def unpack_filtered(archive_path: str, dest: str) -> None:
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest, filter="data")
