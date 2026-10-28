"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""


class JobRunner:
    def execute(self, job_name: str) -> None:
        print("queued", job_name)


def schedule(runner: JobRunner, job_id: str) -> None:
    runner.execute("nightly-report-" + job_id)
