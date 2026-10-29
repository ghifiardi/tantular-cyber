from __future__ import annotations

import os
from pathlib import Path

import pytest

from tantular_cyber.security import (
    PathSafetyError,
    collect_files,
    redact,
    resolve_output,
    resolve_repo,
    sanitize_text,
    scrub_paths,
)


def test_resolve_repo_rejects_missing_and_files(tmp_path: Path) -> None:
    with pytest.raises(PathSafetyError, match="does not exist"):
        resolve_repo(tmp_path / "nope")
    f = tmp_path / "file.py"
    f.write_text("x = 1\n")
    with pytest.raises(PathSafetyError, match="not a directory"):
        resolve_repo(f)


def test_resolve_repo_canonicalizes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert resolve_repo(str(repo / ".." / "repo")) == repo.resolve()


def test_output_inside_repo_rejected_by_default(tmp_path: Path) -> None:
    repo = resolve_repo(tmp_path)
    with pytest.raises(PathSafetyError, match="inside the analyzed repository"):
        resolve_output(tmp_path / "out", repo)
    with pytest.raises(PathSafetyError):
        resolve_output(tmp_path, repo)
    assert resolve_output(tmp_path / "out", repo, allow_inside_repo=True) == repo / "out"


def test_output_outside_repo_ok_and_must_be_dir(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = resolve_repo(repo_dir)
    assert resolve_output(tmp_path / "out", repo) == (tmp_path / "out").resolve()
    (tmp_path / "afile").write_text("")
    with pytest.raises(PathSafetyError, match="not a directory"):
        resolve_output(tmp_path / "afile", repo)


def test_collect_files_applies_skip_policy(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "config").write_text("[core]\n")
    (repo / "node_modules" / "pkg").mkdir(parents=True)
    (repo / "node_modules" / "pkg" / "index.js").write_text("x")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('hi')\n")
    (repo / ".env").write_text("SECRET=1\n")
    (repo / "server.pem").write_text("-----BEGIN PRIVATE KEY-----\n")
    (repo / "image.png").write_bytes(b"\x89PNG\x00\x00binary")
    (repo / "big.py").write_text("x = 1\n" * 1000)
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 'outside'\n")
    os.symlink(outside, repo / "escape.py")
    os.symlink(repo / "src" / "app.py", repo / "internal_link.py")
    os.symlink(tmp_path, repo / "escape_dir")

    inv = collect_files(repo.resolve(), max_file_bytes=1000)
    reasons = {s.path: s.reason for s in inv.skipped}

    assert inv.files == ["src/app.py"]
    assert reasons[".git"] == "excluded_directory"
    assert reasons["node_modules"] == "excluded_directory"
    assert reasons[".env"] == "possible_secret_file"
    assert reasons["server.pem"] == "possible_secret_file"
    assert reasons["image.png"] == "binary"
    assert reasons["big.py"] == "oversized"
    assert reasons["escape.py"] == "symlink_escapes_scope"
    assert reasons["escape_dir"] == "symlink_escapes_scope"
    assert reasons["internal_link.py"] == "symlink"


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("key = 'AKIAABCDEFGHIJKLMNOP'", "AKIAABCDEFGHIJKLMNOP"),
        (
            "token: ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        ),
        ('password = "hunter2hunter2"', "hunter2hunter2"),
        ('{"api_key": "abcd1234efgh"}', "abcd1234efgh"),
        ("DB_URL=postgres://admin:s3cr3tpw@db:5432/app", "s3cr3tpw"),
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz"),
        ("sk-ant-api03-abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz"),
        ("xoxb-1234567890-abcdefghij", "xoxb-1234567890-abcdefghij"),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----", "MIIE"),
    ],
)
def test_redact_masks_credentials(text: str, secret: str) -> None:
    redacted, count = redact(text)
    assert secret not in redacted
    assert count >= 1
    assert "[REDACTED" in redacted


def test_redact_leaves_sql_concatenation_intact() -> None:
    line = """query = "SELECT * FROM users WHERE username = '" + username + "'" """
    assert redact(line) == (line, 0)
    assert redact("author = 'ghifiardi'") == ("author = 'ghifiardi'", 0)


def test_sanitize_text_strips_control_and_bidi_chars_and_bounds_length() -> None:
    assert sanitize_text("a\x1b[31mb\u202ec\x00d") == "a[31mbcd"
    assert len(sanitize_text("x" * 50, max_len=10)) == 10


def test_scrub_paths_hides_absolute_locations(tmp_path: Path) -> None:
    text = f"error in {tmp_path}/app.py and {Path.home()}/x"
    scrubbed = scrub_paths(text, tmp_path)
    assert str(tmp_path) not in scrubbed
    assert "<repo>/app.py" in scrubbed


def test_source_tree_contains_no_invisible_or_bidi_characters() -> None:
    """Guards against Trojan-Source style characters slipping into the code base."""
    root = Path(__file__).parents[1]
    bad = {*range(0x200B, 0x2010), *range(0x202A, 0x202F), *range(0x2060, 0x206A), 0xFEFF}
    # Deliberate Trojan-Source fixture; scanned by the benchmark, never edited by tools.
    allowlisted = {"benchmarks/corpus/cases/unicode/bidi_comment.py"}
    paths = [
        *root.glob("src/**/*.py"),
        *root.glob("src/**/*.yml"),
        *root.glob("tests/**/*.py"),
        *root.glob("tests/**/*.json"),
        *root.glob("*.md"),
        *root.glob("docs/**/*.md"),
        *root.glob("benchmarks/**/*.py"),
        *root.glob("benchmarks/**/*.json"),
        *root.glob("benchmarks/**/*.html"),
    ]
    for path in paths:
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        hits = [hex(ord(ch)) for ch in text if ord(ch) in bad]
        if rel in allowlisted:
            assert hits, f"{rel} should still contain its deliberate characters"
        else:
            assert not hits, f"{rel} contains {hits}"
