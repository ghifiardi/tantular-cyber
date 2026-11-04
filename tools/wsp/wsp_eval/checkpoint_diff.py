"""Read-only tensor-level diff of two ``.safetensors`` files, standard library only.

Used to verify, before attributing any result to an edit, which tensors a published or
candidate checkpoint actually changes relative to its base: names, shapes and dtypes,
per-tensor SHA-256, byte-level changed-element counts and, for float dtypes, the maximum
absolute and L2 magnitude of the change. Files are opened read-only and streamed in chunks;
no framework is imported and nothing is executed (safetensors has no code payload, unlike
pickle-based formats, which this module does not read).

Performance: magnitudes are computed in pure Python only for chunks whose bytes differ, so a
sparse edit of a sub-1B model is practical; a dense change across a large model is slow.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, BinaryIO

from .hashing import sha256_file

MAX_HEADER_BYTES = 100 * 1024 * 1024
DTYPE_SIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_FLOAT_FMT = {"F16": "e", "F32": "f", "F64": "d"}


class SafetensorsError(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # json.loads keeps the last duplicate key silently; a header listing a tensor twice could
    # hide one entry from the diff, so it is rejected.
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise SafetensorsError(f"duplicate header key {key!r}")
        out[key] = value
    return out


def read_header(path: Path) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Return (data_start, tensors, metadata) after validating offsets against the file."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise SafetensorsError(f"{path.name}: truncated header length")
        (n,) = struct.unpack("<Q", raw_len)
        if n > MAX_HEADER_BYTES or 8 + n > size:
            raise SafetensorsError(f"{path.name}: implausible header length {n}")
        try:
            header = json.loads(fh.read(n).decode("utf-8"), object_pairs_hook=_no_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetensorsError(f"{path.name}: header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"{path.name}: header must be an object")
    metadata = header.pop("__metadata__", {}) or {}
    data_start, data_len = 8 + n, size - 8 - n
    for name, t in header.items():
        try:
            dtype, shape, (begin, end) = t["dtype"], t["shape"], t["data_offsets"]
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetensorsError(f"{path.name}: malformed entry for {name!r}") from exc
        if not (
            isinstance(shape, list)
            and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)
            and all(isinstance(o, int) and not isinstance(o, bool) for o in (begin, end))
        ):
            raise SafetensorsError(f"{path.name}: malformed shape or offsets for {name!r}")
        if dtype not in DTYPE_SIZE:
            raise SafetensorsError(f"{path.name}: unsupported dtype {dtype!r} for {name!r}")
        expected = DTYPE_SIZE[dtype] * math.prod(shape)
        if not (0 <= begin <= end <= data_len) or end - begin != expected:
            raise SafetensorsError(f"{path.name}: bad data_offsets for {name!r}")
    return data_start, header, metadata


def _floats(dtype: str, data: bytes) -> list[float]:
    if dtype == "BF16":
        return [
            struct.unpack("<f", struct.pack("<I", h << 16))[0]
            for (h,) in struct.iter_unpack("<H", data)
        ]
    fmt = "<" + _FLOAT_FMT[dtype]
    return [v for (v,) in struct.iter_unpack(fmt, data)]


def _compare_tensor(
    fa: BinaryIO, fb: BinaryIO, start_a: int, start_b: int, length: int, dtype: str, chunk: int
) -> dict[str, Any]:
    size = DTYPE_SIZE[dtype]
    chunk = max(size, chunk - chunk % size)  # whole elements, never zero
    ha, hb = hashlib.sha256(), hashlib.sha256()
    changed = 0
    max_abs = 0.0
    sq = 0.0
    non_finite = 0
    fa.seek(start_a)
    fb.seek(start_b)
    remaining = length
    while remaining:
        k = min(chunk, remaining)
        a, b = fa.read(k), fb.read(k)
        if len(a) != k or len(b) != k:
            raise SafetensorsError("unexpected end of tensor data")
        ha.update(a)
        hb.update(b)
        remaining -= k
        if a == b:
            continue
        idx = [i for i in range(0, k, size) if a[i : i + size] != b[i : i + size]]
        changed += len(idx)
        if dtype in _FLOAT_FMT or dtype == "BF16":
            va = _floats(dtype, b"".join(a[i : i + size] for i in idx))
            vb = _floats(dtype, b"".join(b[i : i + size] for i in idx))
            for x, y in zip(va, vb, strict=True):
                d = y - x
                if not math.isfinite(d):
                    non_finite += 1
                    continue
                max_abs = max(max_abs, abs(d))
                sq += d * d
    out: dict[str, Any] = {
        "sha256_base": ha.hexdigest(),
        "sha256_candidate": hb.hexdigest(),
        "changed_elements": changed,
    }
    if dtype in _FLOAT_FMT or dtype == "BF16":
        out.update(max_abs_diff=max_abs, l2_diff=math.sqrt(sq), non_finite_diffs=non_finite)
    return out


def diff_checkpoints(base: Path, candidate: Path, chunk_bytes: int = 1 << 20) -> dict[str, Any]:
    start_a, ta, meta_a = read_header(base)
    start_b, tb, meta_b = read_header(candidate)
    only_base = sorted(set(ta) - set(tb))
    only_candidate = sorted(set(tb) - set(ta))
    mismatched, changed, unchanged = [], [], 0
    total_changed = 0
    with base.open("rb") as fa, candidate.open("rb") as fb:
        for name in sorted(set(ta) & set(tb)):
            a, b = ta[name], tb[name]
            if a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
                mismatched.append(
                    {
                        "name": name,
                        "base": {"dtype": a["dtype"], "shape": a["shape"]},
                        "candidate": {"dtype": b["dtype"], "shape": b["shape"]},
                    }
                )
                continue
            length = a["data_offsets"][1] - a["data_offsets"][0]
            res = _compare_tensor(
                fa,
                fb,
                start_a + a["data_offsets"][0],
                start_b + b["data_offsets"][0],
                length,
                a["dtype"],
                chunk_bytes,
            )
            if res["changed_elements"]:
                total = math.prod(a["shape"])
                changed.append(
                    {
                        "name": name,
                        "dtype": a["dtype"],
                        "shape": a["shape"],
                        "elements": total,
                        **res,
                    }
                )
                total_changed += res["changed_elements"]
            else:
                unchanged += 1
    return {
        "base": {"file": base.name, "sha256": sha256_file(base), "tensors": len(ta)},
        "candidate": {"file": candidate.name, "sha256": sha256_file(candidate), "tensors": len(tb)},
        "metadata_equal": meta_a == meta_b,
        "only_in_base": only_base,
        "only_in_candidate": only_candidate,
        "shape_or_dtype_mismatch": mismatched,
        "changed_tensors": changed,
        "unchanged_tensors": unchanged,
        "changed_elements_total": total_changed,
        "identical": not (only_base or only_candidate or mismatched or changed),
        "comparison": "byte-level per element; magnitudes for F16/BF16/F32/F64 only",
    }


_MANIFEST_DTYPES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}


def verify_edit(diff: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Check a :func:`diff_checkpoints` report against a published edit manifest.

    ``manifest`` has the authors' shape (exp8_math/runs/<run>/edit/manifest.json): ``tensors``
    with name, shape, dtype, changed_entries, base_sha256 and edited_sha256 of the raw bytes.
    ``verified`` requires that exactly the listed tensors changed, with matching shapes, dtypes,
    changed-entry counts and both hashes, and that no tensor was added, removed or reshaped.
    Header metadata differences are reported, not failed: they do not change tensor values.
    """
    problems: list[str] = []
    expected = {t["name"]: t for t in manifest.get("tensors", [])}
    if not expected:
        problems.append("manifest lists no tensors")
    changed = {t["name"]: t for t in diff["changed_tensors"]}
    for key in ("only_in_base", "only_in_candidate", "shape_or_dtype_mismatch"):
        if diff[key]:
            problems.append(f"{key}: {diff[key]}")
    unexpected = sorted(set(changed) - set(expected))
    unchanged = sorted(set(expected) - set(changed))
    if unexpected:
        problems.append(f"tensors changed that the manifest does not list: {unexpected}")
    if unchanged:
        problems.append(f"manifest tensors that did not change: {unchanged}")
    for name in sorted(set(changed) & set(expected)):
        got, want = changed[name], expected[name]
        if list(got["shape"]) != list(want["shape"]):
            problems.append(f"{name}: shape {got['shape']} != {want['shape']}")
        if got["dtype"] != _MANIFEST_DTYPES.get(want.get("dtype"), want.get("dtype")):
            problems.append(f"{name}: dtype {got['dtype']} != {want.get('dtype')}")
        if "changed_entries" in want and got["changed_elements"] != want["changed_entries"]:
            problems.append(
                f"{name}: {got['changed_elements']} changed entries != {want['changed_entries']}"
            )
        if got["sha256_base"] != want.get("base_sha256"):
            problems.append(f"{name}: base tensor hash differs from the manifest")
        if got["sha256_candidate"] != want.get("edited_sha256"):
            problems.append(f"{name}: edited tensor hash differs from the manifest")
    return {
        "status": "verified" if not problems else "failed",
        "problems": problems,
        "tensors_checked": sorted(set(changed) & set(expected)),
        "metadata_equal": diff["metadata_equal"],
        "base_file_sha256": diff["base"]["sha256"],
        "candidate_file_sha256": diff["candidate"]["sha256"],
    }
