"""
Preference-based evaluation.

Instead of "did you get the right answer?", we ask:
"Does the response align with this user's preferences?"

For each query we generate multiple responses and check which ones
align with the user's preferred option from the PrefEval dataset.

Aligned responses become positives for the steering vector update.
Misaligned responses become negatives.
"""

from dataclasses import dataclass
import torch
from config import Config
from models import generate_response
from tasks import check_alignment
from steering import SteeringVector


@dataclass
class ResponseEval:
    query: str
    preference: str
    aligned_option: str
    all_options: list[str]
    response: str
    hidden_state: torch.Tensor | None
    is_aligned: bool
    aligned_score: int
    best_score: int


def evaluate_responses(samples: list[dict], models: dict, config: Config,
                       steering_vec: SteeringVector | None = None
                       ) -> list[ResponseEval]:
    """
    For each sample, generate multiple responses and check alignment.
    """
    results = []

    for sample in samples:
        query = sample["implicit_query"]
        preference = sample["preference"]
        aligned_op = sample["aligned_op"]
        all_options = sample["options"]

        for _ in range(config.num_responses_per_query):
            hook = None
            if steering_vec is not None and steering_vec.num_updates > 0:
                hook = steering_vec.make_hook()

            response, h = generate_response(query, models, config,
                                            steering_hook=hook)

            alignment = check_alignment(response, aligned_op, all_options)

            results.append(ResponseEval(
                query=query,
                preference=preference,
                aligned_option=aligned_op,
                all_options=all_options,
                response=response,
                hidden_state=h,
                is_aligned=alignment["is_aligned"],
                aligned_score=alignment["aligned_score"],
                best_score=alignment["best_score"],
            ))

    return results


def partition_by_alignment(evals: list[ResponseEval]
                           ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Split into positive (aligned) and negative (misaligned) hidden states.
    """
    positives = []
    negatives = []

    for ev in evals:
        if ev.hidden_state is None:
            continue
        h = ev.hidden_state.flatten()
        if ev.is_aligned:
            positives.append(h)
        else:
            negatives.append(h)

    return positives, negatives
