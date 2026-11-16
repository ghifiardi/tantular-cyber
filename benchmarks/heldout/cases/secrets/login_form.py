"""Held-out evaluation case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

PASSWORD_PROMPT = "Masukkan kata sandi Anda"
PASSWORD_HELP = "Minimal 12 karakter, gunakan huruf dan angka"


def render_labels() -> dict[str, str]:
    return {"prompt": PASSWORD_PROMPT, "help": PASSWORD_HELP}
