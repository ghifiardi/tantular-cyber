"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

from jinja2 import Environment, FileSystemLoader, select_autoescape


def make_env() -> Environment:
    return Environment(loader=FileSystemLoader("templates"))


def make_env_escaped() -> Environment:
    return Environment(loader=FileSystemLoader("templates"), autoescape=select_autoescape())
