import json

import yaml
from yaml import SafeLoader


def parse_config(text):
    return yaml.safe_load(text)


def parse_explicit(text):
    return yaml.load(text, Loader=yaml.SafeLoader)


def parse_c(text):
    return yaml.load(text, Loader=yaml.CSafeLoader)


def parse_bare(text):
    return yaml.load(text, Loader=SafeLoader)


def parse_json(text):
    return json.loads(text)
