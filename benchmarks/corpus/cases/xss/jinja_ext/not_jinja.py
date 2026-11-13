"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

LAYOUT = "screens/home.html.j2"


def select_autoescape(extensions: list[str]) -> list[str]:
    return [e.lower() for e in extensions]


class Environment:
    def __init__(self, autoescape: list[str]) -> None:
        self.autoescape = autoescape


def make_registry() -> Environment:
    return Environment(autoescape=select_autoescape(["html"]))
