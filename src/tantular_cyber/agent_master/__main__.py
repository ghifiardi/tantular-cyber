"""Interactive terminal for the Master; no changes to the scanner CLI."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import logging
import os
import re
import sys
import termios
from pathlib import Path

from .engine import Master
from .model import LocalModel
from .schemas import Limits
from .store import Store
from .tools import Registry

PASTE_START = b"\x1b[200~"
PASTE_END = b"\x1b[201~"
PASTE_IDLE_SECONDS = 0.08
MAX_INPUT_BYTES = 200_000  # upper bound for Limits.max_output_bytes
LAUNCH_COMMAND = re.compile(
    r"^(?:\S+@[^\n]+?[%$]\s+)?(?:\w+=\S+\s+)*"
    r"(?:(?:\S*/)?python[\d.]*\s+-m\s+tantular_cyber\.agent_master\b|"
    r"tantular-agent-master(?:$|\s+--))"
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tantular-agent-master")
    p.add_argument(
        "--model",
        default=os.environ.get("TANTULAR_MASTER_MODEL"),
        help="installed Tantular tag from ollama list (required; never downloaded)",
    )
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument(
        "--root",
        type=Path,
        action="append",
        required=True,
        help="authorized read-only workspace; repeat for multiple roots",
    )
    p.add_argument(
        "--state-dir", type=Path, default=Path.home() / ".local/state/tantular-agent-master"
    )
    p.add_argument("--language", choices=["auto", "id", "en"], default="auto")
    p.add_argument("--once", help="one request, then exit (bounded smoke workflows)")
    for field, info in Limits.model_fields.items():
        p.add_argument(
            "--" + field.replace("_", "-"), type=type(info.default), default=info.default
        )
    return p


async def read_ready(fd: int) -> bytes:
    """Read POSIX stdin without leaving a thread behind on cancellation."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bytes] = loop.create_future()

    def ready() -> None:
        if not future.done():
            try:
                future.set_result(os.read(fd, 4096))
            except OSError as exc:
                future.set_exception(exc)

    loop.add_reader(fd, ready)
    try:
        value = await future
    finally:
        loop.remove_reader(fd)
    if not value:
        raise EOFError
    return value


async def terminal_input(prompt: str) -> str:
    fd = sys.stdin.fileno()
    print(prompt, end="", flush=True)

    # Terminals with bracketed paste mark the entire paste. Group queued lines on older
    # terminals too, so a copied transcript isn't dispatched a line at a time.
    buffer = bytearray()
    trailing = b""
    in_paste = False
    exceeded = False
    chunk = await read_ready(fd)
    while True:
        if not exceeded:
            exceeded = len(buffer) + len(chunk) > MAX_INPUT_BYTES
            if not exceeded:
                buffer.extend(chunk)
        markers = trailing + chunk
        start, end = markers.rfind(PASTE_START), markers.rfind(PASTE_END)
        if start >= 0 or end >= 0:
            in_paste = start > end
        trailing = markers[-(len(PASTE_START) - 1) :]
        try:
            async with asyncio.timeout(None if in_paste else PASTE_IDLE_SECONDS):
                chunk = await read_ready(fd)
        except (TimeoutError, EOFError):
            break

    if exceeded or in_paste:
        raise ValueError("incomplete or oversized terminal input")
    return (
        buffer.replace(PASTE_START, b"").replace(PASTE_END, b"").decode("utf-8", errors="replace")
    )


def is_terminal_transcript(text: str) -> bool:
    for line in text.splitlines():
        line = line.lstrip()
        if line.startswith(("Master ready / Master siap.", "you> ")) or LAUNCH_COMMAND.match(line):
            return True
    return False


async def chat(master: Master) -> None:
    print("Master ready / Master siap. /status, /resume RUN, /cancel, /quit. Ctrl+C stops.")
    pending: asyncio.Task | None = None
    bracketed = sys.stdin.isatty() and sys.stdout.isatty()
    terminal_settings = None
    try:
        if bracketed:
            fd = sys.stdin.fileno()
            terminal_settings = termios.tcgetattr(fd)
            echo_settings = terminal_settings.copy()
            # Preserve ordinary echo and editing. Caret notation makes the terminal's
            # paste delimiters visible as ^[[200~ and ^[[201~ instead of control bytes.
            echo_settings[3] &= ~getattr(termios, "ECHOCTL", 0)
            termios.tcsetattr(fd, termios.TCSANOW, echo_settings)
            print("\x1b[?2004h", end="", flush=True)
        while True:
            try:
                text = await terminal_input("you> ")
            except EOFError:
                break
            except ValueError:
                print(
                    "Input too large or incomplete. / Masukan terlalu panjang atau tidak lengkap."
                )
                continue
            text = text.strip()
            if not text:
                continue
            if is_terminal_transcript(text):
                print(
                    "Terminal output ignored. Enter your request at you>. "
                    "/ Keluaran terminal diabaikan. Ketik permintaan di you>."
                )
                continue
            if text == "/quit":
                break
            if text == "/status":
                for run in master.store.runs():
                    print(json.dumps(run))
                continue
            if text.startswith("/status "):
                try:
                    run_id = text.split(maxsplit=1)[1]
                    row = master.store.run(run_id)
                    print(json.dumps({k: row[k] for k in ("id", "status", "calls", "steps")}))
                    for task in master.store.tasks(run_id):
                        print(f"{task['id']}: {task['status']}")
                    if row["report"]:
                        print(row["report"])
                except ValueError:
                    print("Unknown run / Run tidak ditemukan.")
                continue
            if text == "/cancel":
                if pending and not pending.done():
                    pending.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await pending
                else:
                    print("No active request / Tidak ada permintaan aktif.")
                continue
            if pending and not pending.done():
                print("Request active; use /status or /cancel. / Permintaan masih berjalan.")
                continue
            if pending and not pending.cancelled() and pending.exception():
                print("Previous request failed / Permintaan sebelumnya gagal.")
            if text.startswith("/resume "):
                run_id = text.split(maxsplit=1)[1]
                try:
                    master.store.run(run_id)
                except ValueError:
                    print("Unknown run / Run tidak ditemukan.")
                    continue
                pending = asyncio.create_task(master.resume(run_id))
            elif text.startswith("/"):
                print("Unknown command / Perintah tidak dikenal.")
            else:
                pending = asyncio.create_task(master.request(text))
    finally:
        if terminal_settings is not None:
            termios.tcsetattr(fd, termios.TCSANOW, terminal_settings)
            print("\x1b[?2004l", end="", flush=True)
        if pending and not pending.done():
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending
        if pending and not pending.cancelled():
            # Retrieve exceptions without printing untrusted exception text or tracebacks.
            pending.exception()


async def run(args: argparse.Namespace) -> int:
    if not args.model:
        print("Choose --model from `ollama list`; no tag is assumed.", file=sys.stderr)
        return 2
    state = args.state_dir.expanduser()
    if state.is_symlink():
        raise ValueError("symlink state directory refused")
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    state.chmod(0o700)
    # One orchestrator owns a store. A second process must not recover live tasks.
    lock_fd = os.open(state / "master.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    store = None
    model = None
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("This state store already has an active Master.", file=sys.stderr)
            return 2
        limits = Limits(**{key: getattr(args, key) for key in Limits.model_fields})
        tools = Registry(args.root, state / "artifacts", limits)
        store = Store(state / "tasks.sqlite3")
        store.recover()
        model = LocalModel(
            args.model,
            args.ollama_url,
            timeout=limits.request_timeout,
            num_predict=limits.num_predict,
            max_bytes=limits.max_output_bytes,
            num_ctx=limits.num_ctx,
        )
        async with asyncio.timeout(min(limits.request_timeout, limits.max_duration)):
            await model.discover()
        print(f"Local model: {args.model}; weights: {model.identity.get('digest', 'unknown')}")
        limits.num_ctx = model.num_ctx  # persist the actual inference context for resume
        print(f"Context: {model.num_ctx} tokens ({model.context_source}).")
        if model.loaded_context and model.loaded_context != model.num_ctx:
            print(
                "Configured context differs from the loaded model; Ollama may queue a reload. "
                "Use --num-ctx 0 to reuse it."
            )
        print(f"Private state: {state}; drafts: {tools.output}")
        master = Master(store, model, tools, limits, language=args.language)
        if args.once:
            run_id = await master.request(args.once)
            return 0 if store.run(run_id)["status"] == "completed" else 3
        await chat(master)
        return 0
    finally:
        if model:
            await model.close()
        if store:
            store.close()
        os.close(lock_fd)


def main() -> int:
    # Libraries may log metadata; never enable HTTP payload or prompt debug logging.
    logging.basicConfig(level=logging.ERROR)
    try:
        return asyncio.run(run(parser().parse_args()))
    except KeyboardInterrupt:
        print("\nStopped / Dihentikan. Task state preserved.")
        return 130
    except Exception as exc:  # CLI boundary intentionally omits exception payloads/source.
        print("Agent Master failed: " + type(exc).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
