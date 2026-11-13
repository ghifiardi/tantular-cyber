"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import jinja2

DEFAULT_SETTINGS = {
    "template_dir": "layouts",
    "templates": {
        "alert": "alert_digest.html.j2",
        "weekly": "weekly_summary.html.j2",
    },
}


class ReportService:
    def __init__(self, settings: dict | None = None) -> None:
        self.settings = settings or DEFAULT_SETTINGS
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.settings["template_dir"]),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

    def render(self, kind: str, **values: str) -> str:
        name = self.settings["templates"].get(kind, "alert_digest.html.j2")
        return self.env.get_template(name).render(**values)
