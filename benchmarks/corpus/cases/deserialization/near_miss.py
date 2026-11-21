"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import json

import yaml


def settings_positional(text):
    return yaml.load(text, yaml.SafeLoader)


def settings_keyword(text):
    return yaml.load(text, Loader=yaml.CSafeLoader)


def settings_json(text):
    return json.loads(text)
