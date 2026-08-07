"""One-shot: re-merge + push the backfilled subs to HF so the judge_model
field is reflected on the Hub. Reuses judge_qa_pairs' own upload path.

Only the subs whose legacy records were backfilled need this — pass them as
args, default = the two that had <none> records.
"""
import sys

import judge_qa_pairs as J  # module-level load_dotenv sets HF_TOKEN

DEFAULT_SUBS = ["AskAcademia", "AskBaking"]


def main():
    subs = sys.argv[1:] or DEFAULT_SUBS
    for sub in subs:
        print(f"[push] merging + uploading sub-{sub} ...", flush=True)
        J._upload_one_sub(sub, cumulative_total=0)
    print("[push] done")


if __name__ == "__main__":
    main()
