"""Fetch all submissions and comments for a single subreddit via the
arctic-shift HTTP API (https://arctic-shift.photon-reddit.com/api).

Writes to:
  <out_dir>/sub-<subreddit>/RS_<subreddit>.ndjson
  <out_dir>/sub-<subreddit>/RC_<subreddit>.ndjson
  <out_dir>/sub-<subreddit>/fetch_state.json   (resume cursors + counters)

Both files are uncompressed newline-delimited JSON, flushed and fsync'd after
each API page so you can `tail`/`jq` them while the fetcher runs.

Pagination strategy:
  sort=asc&after=<last_created_utc>  — page forward through history. Reddit
  IDs share timestamps within a second; we de-dup against the last batch's
  seen IDs to avoid emitting duplicates at second boundaries.

Rate limiting: configurable; default 5 req/s, well under the published
2000/40s ceiling.

Usage:
  python fetch_subreddit.py --sub AskHistorians --out-dir ../data/dumps
  python fetch_subreddit.py --sub AskHistorians --out-dir ../data/dumps \\
      --max-records 5000    # smoke test
  python fetch_subreddit.py --sub AskHistorians --out-dir ../data/dumps \\
      --start 2020-01-01    # only fetch from this date forward
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import orjson
import urllib.error
import urllib.parse
import urllib.request


API_BASE = "https://arctic-shift.photon-reddit.com/api"
PAGE_LIMIT = 100  # service-enforced max
USER_AGENT = "personalization-research/0.1 (academic; arctic-shift API)"


class FetchState:
    """Resumable cursors + counters, persisted to disk after each flush."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {
            "subreddit": None,
            "rs": {"last_created_utc": 0, "last_ids": [], "n_written": 0, "done": False},
            "rc": {"last_created_utc": 0, "last_ids": [], "n_written": 0, "done": False},
            "started_at": None,
            "updated_at": None,
        }
        if path.exists():
            self.data.update(orjson.loads(path.read_bytes()))

    def save(self) -> None:
        self.data["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_bytes(orjson.dumps(self.data, option=orjson.OPT_INDENT_2))
        tmp.replace(self.path)


def _http_get_json(url: str, retries: int = 15) -> dict:
    """GET with aggressive retries for the arctic-shift API.

    Observed failure modes in practice:
      - 422 Unprocessable Entity (intermittent, same URL works on retry)
      - 429 rate-limited
      - 522 Cloudflare origin timeout (the upstream service is unresponsive;
        these come in spells of seconds-to-minutes)
      - URLError / TimeoutError (network blip)

    All are retryable. Backoff is exponential with a 10-minute cap, so a long
    Cloudflare outage doesn't kill a multi-hour subreddit pull.
    """
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                reset = e.headers.get("x-ratelimit-reset")
                wait = float(reset) if reset and reset.replace(".", "", 1).isdigit() else 30
                wait = min(max(wait, 5), 120)
                print(f"  [429] sleeping {wait:.0f}s (attempt {attempt+1}/{retries})", file=sys.stderr)
                time.sleep(wait)
                last_err = e
                continue
            if e.code >= 422:
                # Exponential backoff capped at 10 min. Sequence:
                # 2, 4, 8, 16, 32, 64, 128, 256, 512, 600, 600, 600, 600, 600, 600
                wait = min(2 ** (attempt + 1), 600)
                print(f"  [{e.code}] retry in {wait}s (attempt {attempt+1}/{retries})", file=sys.stderr)
                time.sleep(wait)
                last_err = e
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            wait = min(2 ** (attempt + 1), 600)
            print(f"  [{type(e).__name__}] retry in {wait}s (attempt {attempt+1}/{retries}): {e}",
                  file=sys.stderr)
            time.sleep(wait)
            last_err = e
    raise RuntimeError(f"giving up after {retries} retries: {last_err}")


def _fetch_kind(
    kind: str,                # "posts" or "comments"
    sub: str,
    out_fh,
    state_section: dict,
    state: FetchState,
    rate_seconds: float,
    max_records: int | None,
    start_after: int | None,
) -> None:
    endpoint = f"{API_BASE}/{kind}/search"
    seen_ids = set(state_section.get("last_ids", []))
    cursor = max(int(state_section.get("last_created_utc", 0) or 0), start_after or 0)
    n_written = state_section.get("n_written", 0)

    next_flush_at = n_written + 1  # flush after first page

    while True:
        params = {
            "subreddit": sub,
            "sort": "asc",
            "limit": PAGE_LIMIT,
        }
        if cursor:
            params["after"] = cursor
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"

        t0 = time.time()
        body = _http_get_json(url)
        data = body.get("data") or []

        if not data:
            state_section["done"] = True
            state.save()
            break

        new_records = 0
        last_ts = cursor
        last_ids_in_batch: list[str] = []
        for rec in data:
            rid = rec.get("id")
            ts = int(rec.get("created_utc") or 0)
            if rid in seen_ids and ts == cursor:
                continue  # dedupe across second-boundary pages
            out_fh.write(orjson.dumps(rec))
            out_fh.write(b"\n")
            new_records += 1
            if ts > last_ts:
                last_ts = ts
                last_ids_in_batch = [rid]
            elif ts == last_ts:
                last_ids_in_batch.append(rid)

        if new_records == 0 and last_ts == cursor:
            # API returned a page entirely composed of records at the cursor
            # second that we'd already seen. Nudge cursor forward by 1s.
            cursor = cursor + 1
        else:
            cursor = last_ts
            seen_ids = set(last_ids_in_batch)

        n_written += new_records
        state_section["last_created_utc"] = cursor
        state_section["last_ids"] = list(seen_ids)
        state_section["n_written"] = n_written

        # Flush after EVERY page so user can inspect mid-run.
        out_fh.flush()
        try:
            import os
            os.fsync(out_fh.fileno())
        except OSError:
            pass
        state.save()

        if n_written >= next_flush_at:
            print(
                f"  [{kind}] cursor={dt.datetime.fromtimestamp(cursor, tz=dt.timezone.utc).date()} "
                f"n={n_written:,} (+{new_records})",
                flush=True,
            )
            next_flush_at = n_written + 5000  # log every ~5k records

        if max_records and n_written >= max_records:
            print(f"  [{kind}] hit --max-records cap at {n_written}", flush=True)
            return

        # Conservative pacing: rate-limit between requests.
        elapsed = time.time() - t0
        sleep_for = max(0.0, rate_seconds - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)


def fetch_subreddit(
    sub: str,
    out_dir: Path,
    rate_seconds: float = 0.2,
    max_records: int | None = None,
    start_after: int | None = None,
    skip_submissions: bool = False,
    skip_comments: bool = False,
) -> dict:
    batch_dir = out_dir / f"sub-{sub}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    rs_path = batch_dir / f"RS_{sub}.ndjson"
    rc_path = batch_dir / f"RC_{sub}.ndjson"
    state_path = batch_dir / "fetch_state.json"

    state = FetchState(state_path)
    if state.data["subreddit"] is None:
        state.data["subreddit"] = sub
        state.data["started_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        state.save()

    if not skip_submissions and not state.data["rs"]["done"]:
        print(f"[{sub}] fetching submissions → {rs_path}", flush=True)
        with open(rs_path, "ab") as fh:
            _fetch_kind("posts", sub, fh, state.data["rs"], state,
                        rate_seconds, max_records, start_after)
    else:
        print(f"[{sub}] submissions already complete or skipped", flush=True)

    if not skip_comments and not state.data["rc"]["done"]:
        print(f"[{sub}] fetching comments → {rc_path}", flush=True)
        with open(rc_path, "ab") as fh:
            _fetch_kind("comments", sub, fh, state.data["rc"], state,
                        rate_seconds, max_records, start_after)
    else:
        print(f"[{sub}] comments already complete or skipped", flush=True)

    return state.data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--rate-seconds", type=float, default=0.2,
                    help="min seconds between requests (default 0.2 = 5 req/s)")
    ap.add_argument("--max-records", type=int, default=None,
                    help="stop after N records per kind (smoke testing)")
    ap.add_argument("--start", default=None,
                    help="only fetch records created on/after YYYY-MM-DD")
    ap.add_argument("--skip-submissions", action="store_true")
    ap.add_argument("--skip-comments", action="store_true")
    args = ap.parse_args()

    start_after = None
    if args.start:
        start_after = int(dt.datetime.fromisoformat(args.start)
                          .replace(tzinfo=dt.timezone.utc).timestamp())

    s = fetch_subreddit(
        sub=args.sub,
        out_dir=args.out_dir,
        rate_seconds=args.rate_seconds,
        max_records=args.max_records,
        start_after=start_after,
        skip_submissions=args.skip_submissions,
        skip_comments=args.skip_comments,
    )
    print(json.dumps({
        "sub": args.sub,
        "rs_n": s["rs"]["n_written"],
        "rc_n": s["rc"]["n_written"],
    }))


if __name__ == "__main__":
    main()
