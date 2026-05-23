"""
Load PrefEval dataset. Each sample IS a user with their own preference.
No fake profiles -- we use the dataset directly.
"""

from datasets import load_dataset


def load_prefeval():
    ds = load_dataset("siyanzhao/prefeval_implicit_choice", split="train")
    return ds


def check_alignment(response: str, aligned_option: str,
                    all_options: list[str]) -> bool:
    """
    Check if the response aligns with this sample's preferred option.
    Score each option by word overlap with the response, see which wins.
    """
    response_lower = response.lower()

    def score(option: str) -> int:
        words = set(option.lower().split())
        return sum(1 for w in words if w in response_lower and len(w) > 3)

    scores = {opt: score(opt) for opt in all_options}
    best = max(scores, key=scores.get)

    # Also check direct keyword hits from the aligned option
    aligned_words = {w for w in aligned_option.lower().split() if len(w) > 4}
    direct_hits = sum(1 for w in aligned_words if w in response_lower)

    return (best == aligned_option) or (direct_hits >= 2)
