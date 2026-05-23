"""Streaming reader for Pushshift / arctic_shift .zst dumps.

Reddit monthly dumps are ndjson compressed with zstandard. Two gotchas:

1. Reddit dumps were compressed with a window size larger than zstd's default
   (max_window_size=2**31). The decompressor must be constructed with
   max_window_size at least that high or it raises ZstdError mid-stream.

2. Lines can be very long (some submissions with embedded JSON metadata are
   100KB+). The reader needs a generous read buffer.

We yield decoded dicts. Malformed lines (extremely rare in practice but
non-zero in older months) are counted and skipped, not raised — interrupting
a multi-month run on one bad record would be costly.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import orjson
import zstandard as zstd

# Reddit dumps use long-range mode; 2**31 is the documented upper bound that
# arctic_shift recommends. Match that here.
ZSTD_MAX_WINDOW = 2**31
READ_CHUNK = 2**24  # 16 MiB


class DumpReadStats:
    __slots__ = ("records", "bad_lines", "bytes_read")

    def __init__(self) -> None:
        self.records = 0
        self.bad_lines = 0
        self.bytes_read = 0


def _iter_text(text_iter, stats: DumpReadStats) -> Iterator[dict]:
    for line in text_iter:
        line = line.strip()
        if not line:
            continue
        try:
            rec = orjson.loads(line)
        except orjson.JSONDecodeError:
            stats.bad_lines += 1
            continue
        stats.records += 1
        yield rec


def iter_dump(path: Path, stats: DumpReadStats | None = None) -> Iterator[dict]:
    """Yield each ndjson record from a .zst dump or a plain .ndjson/.jsonl file.

    Streaming: never holds more than ~READ_CHUNK bytes of decompressed text
    plus one record line in memory. Safe for 100GB+ files.
    """
    stats = stats if stats is not None else DumpReadStats()
    p = str(path)
    if p.endswith(".zst") or p.endswith(".zst_blocks"):
        dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW)
        with open(path, "rb") as fh:
            reader = dctx.stream_reader(fh, read_size=READ_CHUNK)
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace", newline="\n")
            yield from _iter_text(text, stats)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield from _iter_text(fh, stats)


_RS_GLOBS = ("RS_*.zst", "RS_*.zst_blocks", "RS_*.ndjson", "RS_*.jsonl")
_RC_GLOBS = ("RC_*.zst", "RC_*.zst_blocks", "RC_*.ndjson", "RC_*.jsonl")


def _glob_any(d: Path, patterns: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(d.glob(pat))
    return sorted(out)


def discover_months(dumps_dir: Path) -> list[str]:
    """Return sorted list of batch-dir names that contain both RS and RC files.

    A "batch" is any subdir of dumps_dir; the name is opaque (could be YYYY-MM
    for monthly dumps, or sub-<name> for API-fetched subreddit pulls).
    """
    months = []
    for d in sorted(dumps_dir.iterdir() if dumps_dir.exists() else []):
        if not d.is_dir():
            continue
        if _glob_any(d, _RS_GLOBS) and _glob_any(d, _RC_GLOBS):
            months.append(d.name)
    return months


def submission_path(dumps_dir: Path, batch: str) -> Path:
    cands = _glob_any(dumps_dir / batch, _RS_GLOBS)
    if not cands:
        raise FileNotFoundError(f"No RS_* file in {dumps_dir/batch}")
    return cands[0]


def comment_path(dumps_dir: Path, batch: str) -> Path:
    cands = _glob_any(dumps_dir / batch, _RC_GLOBS)
    if not cands:
        raise FileNotFoundError(f"No RC_* file in {dumps_dir/batch}")
    return cands[0]
