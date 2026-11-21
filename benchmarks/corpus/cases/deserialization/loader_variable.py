"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import yaml

LOADER = yaml.SafeLoader


def read_settings(text):
    return yaml.load(text, Loader=LOADER)
