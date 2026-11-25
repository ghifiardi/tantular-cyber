"""``tantular-cyber runtime ...`` — offline Shirudo telemetry import and opt-in analysis.

Exit codes (aligned with ``scan``):
  0  complete
  2  invalid input (unsafe/unreadable path, oversized file, bad arguments, invalid policy)
  3  failed — nothing importable; reports still written with the errors
  4  degraded — reports written, coverage or the advisory analysis is partial
  5  ``runtime eval``: a deterministic import check failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..config import OllamaSettings
from ..reports import _atomic_write
from .models import RuntimeEvidenceReport

EXIT_OK, EXIT_INPUT, EXIT_FAILED, EXIT_DEGRADED, EXIT_GATES = 0, 2, 3, 4, 5
_STATUS_EXIT = {"complete": EXIT_OK, "degraded": EXIT_DEGRADED, "failed": EXIT_FAILED}
MAX_EVIDENCE_BYTES = 128 * 1024 * 1024


def _ollama_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model", default=None, help="local Ollama model (default: tantular-cyber:latest)"
    )
    p.add_argument("--ollama-url", default=None)
    p.add_argument("--ollama-timeout", type=float, default=None)
    p.add_argument("--allow-remote-ollama", action="store_true")


def add_runtime_parser(sub: Any) -> None:
    rt = sub.add_parser("runtime", help="offline Shirudo kernel-obs telemetry import / analysis")
    rsub = rt.add_subparsers(dest="runtime_command", required=True)

    imp = rsub.add_parser("import", help="import a kernel-obs JSON/NDJSON export (offline)")
    imp.add_argument("--input", type=Path, required=True, help="decoder JSON or NDJSON file")
    imp.add_argument("--output", type=Path, required=True, help="directory for runtime-evidence.*")
    imp.add_argument(
        "--path-mode",
        choices=["masked_user", "full", "basename", "hash_only"],
        default="masked_user",
        help="how host paths are rendered (default masks /home/<user>, /Users/<user>)",
    )

    an = rsub.add_parser("analyze", help="opt-in advisory interpretation by a local model")
    an.add_argument("--evidence", type=Path, required=True, help="runtime-evidence.json")
    an.add_argument("--output", type=Path, required=True)
    an.add_argument("--retrieval", action="store_true", help="add local reference chunks")
    an.add_argument("--corpus", type=Path, default=None, help="prepared corpus (default bundled)")
    an.add_argument("--top-k", type=int, default=4)
    an.add_argument("--max-chars", type=int, default=3600)
    an.add_argument("--replay-file", type=Path, default=None, help=argparse.SUPPRESS)
    _ollama_args(an)

    cp = rsub.add_parser("corpus", help="prepare / inspect the local reference corpus")
    cp.add_argument("action", choices=["prepare", "info"])
    cp.add_argument("--checkout", type=Path, help="local upstream checkout (prepare)")
    cp.add_argument("--revision", default=None, help="expected 40-hex commit (prepare)")
    cp.add_argument("--output", type=Path, help="corpus JSON to write (prepare; new file)")
    cp.add_argument("--corpus", type=Path, default=None, help="corpus to inspect (info)")

    ev = rsub.add_parser("eval", help="offline runtime evaluation (synthetic suite)")
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--suite", type=Path, default=None)
    ev.add_argument("--splits", default="dev,heldout,transfer")
    ev.add_argument("--llm", choices=["off", "replay", "ollama"], default="off")
    ev.add_argument("--replay-file", type=Path, default=None)
    ev.add_argument("--arms", default="retrieval_off,retrieval_on")
    ev.add_argument("--runs", type=int, default=1)
    ev.add_argument("--corpus", type=Path, default=None)
    ev.add_argument("--top-k", type=int, default=4)
    ev.add_argument("--max-chars", type=int, default=3600)
    ev.add_argument("--seed", type=int, default=7)
    ev.add_argument(
        "--record-replies",
        type=Path,
        default=None,
        help="write raw model replies (case_id/arm, runs=1 only) for later --llm replay; "
        "never overwrites",
    )
    _ollama_args(ev)

    rv = rsub.add_parser("review", help="human-review templates / summaries for runtime evals")
    rv.add_argument("action", choices=["template", "summarize"])
    rv.add_argument("--queue", type=Path, required=True, help="runtime-review-queue.json")
    rv.add_argument("--reviewer", default=None)
    rv.add_argument("--reviews", type=Path, nargs="*", default=[])
    rv.add_argument("--output", type=Path, required=True)

    lab = rsub.add_parser("lab", help="opt-in disposable Shirudo lab VM adapter")
    lab.add_argument("action", choices=["manifest", "plan", "collect"])
    lab.add_argument("--name", default=None, help="lab target name (lab-/test-/vm-...)")
    lab.add_argument("--host", default=None, help="ssh host of the lab VM")
    lab.add_argument("--user", default="root")
    lab.add_argument("--output", type=Path, default=None)
    lab.add_argument("--no-peek", action="store_true", help="drain maps (lab only)")
    lab.add_argument("--i-confirm-disposable-lab-vm", action="store_true")
    lab.add_argument("--execute", action="store_true", help="actually run the plan over ssh")

    pol = rsub.add_parser("policy", help="draft-only app-sandbox policy proposals")
    pol.add_argument("action", choices=["validate", "propose"])
    pol.add_argument("--policy", type=Path, help="draft sandbox.toml (validate)")
    pol.add_argument("--declared", default="", help="comma-separated Beefile policy ids")
    pol.add_argument("--evidence", type=Path, help="runtime-evidence.json (propose)")
    pol.add_argument("--policy-id", default=None)
    pol.add_argument("--executable-path", default=None)
    pol.add_argument("--output", type=Path, default=None)

    sc = rsub.add_parser("schema", help="print a runtime JSON schema")
    sc.add_argument("which", choices=["evidence", "analysis"])


def _load_evidence(path: Path) -> tuple[RuntimeEvidenceReport, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("evidence must be a regular file")
    raw = path.read_bytes()
    if len(raw) > MAX_EVIDENCE_BYTES:
        raise ValueError("evidence file too large")
    return RuntimeEvidenceReport.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


def _settings(args: argparse.Namespace) -> OllamaSettings:
    return OllamaSettings.from_env(
        base_url=args.ollama_url,
        model=args.model,
        timeout_s=args.ollama_timeout,
        allow_remote=True if args.allow_remote_ollama else None,
    )


def _model_identity(client: Any) -> dict[str, Any]:
    from ..benchmark.provenance import model_state

    state = model_state(client.base_url, client.model)
    return {
        "model": client.model,
        "ollama_version": state.get("ollama_version"),
        "digest": state.get("model_digest"),
        "details": state.get("model_details", {}),
        "checkpoint_identity": (
            "unverified: the Ollama tag and digest identify the served manifest, not the base "
            "weights' lineage"
        ),
    }


def cmd_import(args: argparse.Namespace) -> int:
    from .importer import RuntimeInputError, import_file
    from .reports import write_evidence

    out = args.output.expanduser().resolve()
    if out.exists() and not out.is_dir():
        print("error: output exists and is not a directory", file=sys.stderr)
        return EXIT_INPUT
    if args.input.expanduser().resolve().parent == out and args.input.name.startswith(
        "runtime-evidence"
    ):
        print("error: output would overwrite the input", file=sys.stderr)
        return EXIT_INPUT
    try:
        report = import_file(args.input, path_mode=args.path_mode)
    except RuntimeInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    written = write_evidence(report, out)
    print(f"status: {report.status}")
    for reason in [*report.errors, *report.status_reasons]:
        print(f"  - {reason}")
    for name, cov in sorted(report.coverage.maps.items()):
        print(f"{name}: counted={cov.counted} records={cov.records} events={cov.event_count_total}")
    print(f"loss known: {report.coverage.loss_known}")
    for kind, path in written.items():
        print(f"{kind}: {path}")
    if report.status != "complete":
        print(
            f"WARNING: runtime import {report.status.upper()} — see status_reasons", file=sys.stderr
        )
    return _STATUS_EXIT[report.status]


def cmd_analyze(args: argparse.Namespace) -> int:
    from ..ollama_client import OllamaClient, OllamaError
    from .analysis import RetrievalRecord, analyze_evidence
    from .evaluation import ReplayClient
    from .reports_analysis import write_analysis
    from .retrieval import (
        CorpusError,
        LexicalIndex,
        RetrievalParams,
        chunk_for_prompt,
        load_corpus,
        query_for,
    )

    try:
        evidence, sha = _load_evidence(args.evidence)
    except (OSError, ValueError, ValidationError):
        print("error: cannot read a valid runtime-evidence.json", file=sys.stderr)
        return EXIT_INPUT
    chunks: list[dict[str, str]] = []
    retrieval = RetrievalRecord(enabled=args.retrieval)
    if args.retrieval:
        params = RetrievalParams(top_k=max(1, args.top_k), max_chars=max(200, args.max_chars))
        try:
            corpus = load_corpus(args.corpus)
        except (OSError, ValueError, CorpusError) as exc:
            retrieval.unavailable_reason = f"corpus unavailable ({type(exc).__name__})"
        else:
            query = query_for(evidence)
            chunks = [chunk_for_prompt(c) for c in LexicalIndex(corpus).search(query, params)]
            retrieval = RetrievalRecord(
                enabled=True,
                corpus_id=f"{corpus['project']}@{corpus['revision']}",
                corpus_sha256=corpus["corpus_sha256"],
                parameters=params.as_dict(),
                query_terms=query[:40],
                chunk_ids=[c["chunk_id"] for c in chunks],
            )
    client: Any = None
    identity: dict[str, Any] = {}
    reason = None
    if args.replay_file:
        replies = json.loads(args.replay_file.read_text(encoding="utf-8"))
        client = ReplayClient(replies)
        client.key = "analyze"
    else:
        try:
            client = OllamaClient(_settings(args))
            client.health_check()
            identity = _model_identity(client)
        except OllamaError as exc:
            client, reason = None, f"{type(exc).__name__}: {exc}"
    report = analyze_evidence(
        evidence,
        sha,
        client,
        chunks=chunks,
        retrieval=retrieval,
        model_identity=identity,
        unavailable_reason=reason,
    )
    written = write_analysis(report, args.output.expanduser().resolve())
    print(f"analysis status: {report.status} (evidence status: {evidence.status}, unverified)")
    for r in report.status_reasons:
        print(f"  - {r}")
    for kind, path in written.items():
        print(f"{kind}: {path}")
    worst = max(_STATUS_EXIT[report.status], _STATUS_EXIT[evidence.status])
    return EXIT_DEGRADED if worst == EXIT_FAILED else worst


def cmd_corpus(args: argparse.Namespace) -> int:
    from .retrieval import CorpusError, load_corpus, prepare_corpus

    if args.action == "info":
        try:
            corpus = load_corpus(args.corpus)
        except (OSError, ValueError, CorpusError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_INPUT
        print(
            json.dumps(
                {
                    k: corpus[k]
                    for k in (
                        "project",
                        "repository",
                        "revision",
                        "revision_source",
                        "corpus_sha256",
                        "license_notice",
                    )
                },
                indent=2,
            )
        )
        print(f"documents: {len(corpus['documents'])} chunks: {len(corpus['chunks'])}")
        return EXIT_OK
    if not args.checkout or not args.output:
        print("error: corpus prepare needs --checkout and --output", file=sys.stderr)
        return EXIT_INPUT
    try:
        corpus = prepare_corpus(args.checkout, revision=args.revision)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(corpus, indent=1, ensure_ascii=True, sort_keys=True) + "\n")
    except (OSError, UnicodeDecodeError, CorpusError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    print(
        f"corpus {corpus['revision']} ({corpus['revision_source']}): "
        f"{len(corpus['chunks'])} chunks, sha256 {corpus['corpus_sha256']}"
    )
    return EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    from ..ollama_client import OllamaClient, OllamaError
    from .evaluation import ReplayClient, render_eval_markdown, review_queue, run_runtime_eval
    from .retrieval import CorpusError, RetrievalParams, load_corpus

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    if not set(arms) <= {"retrieval_off", "retrieval_on"} or not arms:
        print("error: --arms must be retrieval_off and/or retrieval_on", file=sys.stderr)
        return EXIT_INPUT
    try:
        corpus = load_corpus(args.corpus)
    except (OSError, ValueError, CorpusError) as exc:
        print(f"error: corpus: {exc}", file=sys.stderr)
        return EXIT_INPUT
    if args.record_replies is not None:
        if args.llm != "ollama" or args.runs != 1:
            print("error: --record-replies requires --llm ollama and --runs 1", file=sys.stderr)
            return EXIT_INPUT
        if args.record_replies.exists() or args.record_replies.is_symlink():
            print("error: --record-replies target exists; refusing to overwrite", file=sys.stderr)
            return EXIT_INPUT
    client: Any = None
    identity: dict[str, Any] = {}
    options: dict[str, Any] = {}
    if args.llm == "replay":
        if not args.replay_file:
            print("error: --llm replay requires --replay-file", file=sys.stderr)
            return EXIT_INPUT
        client = ReplayClient(json.loads(args.replay_file.read_text(encoding="utf-8")))
        identity = {"provider": "replay", "note": "recorded/hand-written replies; no model ran"}
    elif args.llm == "ollama":
        settings = _settings(args)
        settings = OllamaSettings(
            **{
                **settings.__dict__,
                "seed": args.seed,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
            }
        )
        try:
            client = OllamaClient(settings)
            client.health_check()
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_INPUT
        identity = _model_identity(client)
        options = {
            k: getattr(settings, k)
            for k in ("temperature", "num_ctx", "seed", "top_p", "top_k", "repeat_penalty")
        }
    replies: dict[str, str] = {}
    result = run_runtime_eval(
        suite_root=args.suite,
        splits=splits,
        llm_mode=args.llm,
        client=client,
        arms=arms,
        runs=max(1, args.runs),
        corpus=corpus,
        params=RetrievalParams(top_k=max(1, args.top_k), max_chars=max(200, args.max_chars)),
        model_identity=identity,
        options=options,
        on_progress=lambda m: print(m, file=sys.stderr) if args.llm == "ollama" else None,
        replies_out=replies if args.record_replies is not None else None,
    )
    out = args.output
    _atomic_write(out / "runtime-eval.json", result.model_dump_json(indent=2) + "\n")
    _atomic_write(out / "runtime-eval.md", render_eval_markdown(result))
    if result.quality_items:
        _atomic_write(
            out / "runtime-review-queue.json", json.dumps(review_queue(result), indent=2) + "\n"
        )
    if args.record_replies is not None:
        replies["_origin"] = (
            f"raw model output; model={identity.get('model')} digest={identity.get('digest')}; "
            f"prompt_sha256={result.protocol['prompt_sha256']}; synthetic suite only"
        )
        _atomic_write(args.record_replies, json.dumps(replies, indent=2, sort_keys=True) + "\n")
    print(f"runtime import checks: {'PASS' if result.import_passed else 'FAIL'}")
    for arm in result.arms:
        rates = ", ".join(f"{k}={v['rate']}" for k, v in arm.checks.items())
        print(f"{arm.arm}: n={arm.n} assessable={arm.assessable} abstained={arm.abstained} {rates}")
    print(f"json: {out / 'runtime-eval.json'}")
    return EXIT_OK if result.import_passed else EXIT_GATES


def cmd_review(args: argparse.Namespace) -> int:
    from .evaluation import RuntimeReviewFile, review_template, summarize_runtime_reviews

    try:
        raw = args.queue.read_bytes()
        queue = json.loads(raw)
    except (OSError, ValueError):
        print("error: cannot read queue", file=sys.stderr)
        return EXIT_INPUT
    sha = hashlib.sha256(raw).hexdigest()
    if args.action == "template":
        if not args.reviewer:
            print("error: --reviewer pseudonym required", file=sys.stderr)
            return EXIT_INPUT
        try:
            form = review_template(queue, args.reviewer, sha)
        except ValidationError:
            print("error: reviewer must be a pseudonym (letters, digits, _ or -)", file=sys.stderr)
            return EXIT_INPUT
        _atomic_write(args.output, json.dumps(form, indent=2) + "\n")
        print(f"template: {args.output} ({len(queue)} items)")
        return EXIT_OK
    try:
        reviews = [RuntimeReviewFile.model_validate_json(p.read_bytes()) for p in args.reviews]
    except (OSError, ValueError, ValidationError):
        print("error: invalid review file", file=sys.stderr)
        return EXIT_INPUT
    summary = summarize_runtime_reviews(sha, reviews)
    _atomic_write(args.output, json.dumps(summary, indent=2) + "\n")
    print(f"summary: {args.output}")
    return EXIT_OK


def cmd_lab(args: argparse.Namespace) -> int:
    from . import lab

    if args.action == "manifest":
        print(json.dumps(lab.compatibility_manifest(), indent=2))
        return EXIT_OK
    target = lab.LabTarget(
        name=args.name or "",
        ssh_host=args.host or "",
        ssh_user=args.user,
        disposable_confirmed=args.i_confirm_disposable_lab_vm,
    )
    try:
        steps = lab.collection_plan(peek=not args.no_peek)
        if args.action == "plan" or not args.execute:
            target.validate()
            for s in steps:
                print(" ".join(lab.ssh_argv(target, s.remote)) + f"   > {s.output}")
            if args.action == "collect":
                print("dry run: add --execute to run these read-only commands on the lab VM")
            return EXIT_OK
        if args.output is None:
            print("error: collect --execute needs --output", file=sys.stderr)
            return EXIT_INPUT
        run = lab.collect(target, args.output, peek=not args.no_peek)
    except lab.LabRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_INPUT
    for s in run.steps:
        print(f"{s['id']}: exit={s['exit']} bytes={s['bytes']}")
    if run.export_files:
        print(
            f"next: tantular-cyber runtime import --input {args.output / run.export_files[0]} "
            f"--output {args.output / 'evidence'}"
        )
    return EXIT_OK if all(s["exit"] == 0 for s in run.steps) else EXIT_DEGRADED


def cmd_policy(args: argparse.Namespace) -> int:
    from .sandbox_policy import explain_policy, propose_from_evidence, validate_policy

    if args.action == "validate":
        if not args.policy or not args.policy.is_file():
            print("error: --policy must be a draft sandbox.toml file", file=sys.stderr)
            return EXIT_INPUT
        declared = {d.strip() for d in args.declared.split(",") if d.strip()}
        check = validate_policy(args.policy.read_text(encoding="utf-8"), declared)
        text = explain_policy(check)
        if args.output:
            _atomic_write(args.output, text)
        print(text)
        return EXIT_OK if check.valid else EXIT_INPUT
    if not (args.evidence and args.policy_id and args.executable_path):
        print("error: propose needs --evidence, --policy-id and --executable-path", file=sys.stderr)
        return EXIT_INPUT
    try:
        evidence, _ = _load_evidence(args.evidence)
    except (OSError, ValueError, ValidationError):
        print("error: cannot read a valid runtime-evidence.json", file=sys.stderr)
        return EXIT_INPUT
    draft = propose_from_evidence(
        evidence, policy_id=args.policy_id, executable_path=args.executable_path
    )
    if args.output:
        _atomic_write(args.output, draft)
    print(draft)
    return EXIT_OK


def cmd_runtime(args: argparse.Namespace) -> int:
    c = args.runtime_command
    if c == "import":
        return cmd_import(args)
    if c == "analyze":
        return cmd_analyze(args)
    if c == "corpus":
        return cmd_corpus(args)
    if c == "eval":
        return cmd_eval(args)
    if c == "review":
        return cmd_review(args)
    if c == "lab":
        return cmd_lab(args)
    if c == "policy":
        return cmd_policy(args)
    from .analysis import RuntimeAnalysisReport

    model = RuntimeEvidenceReport if args.which == "evidence" else RuntimeAnalysisReport
    print(json.dumps(model.model_json_schema(), indent=2))
    return EXIT_OK
