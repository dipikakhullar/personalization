"""
SteerX-style causal token identification.

For each token in a response, we compute the causal effect:
    delta_t = log p(token_t | user_history, context) - log p(token_t | context_only)

Tokens where delta_t >= threshold are "preference-driven anchor tokens".
These are the tokens whose presence was *caused* by the user's preferences,
not just generic boilerplate.

Reference: SteerX (arxiv.org/abs/2510.22256), Section 4.2-4.3
"""

import torch
import torch.nn.functional as F
from config import Config


def compute_token_causal_effects(
    model, tokenizer, query: str, response: str,
    user_history: list[dict] | None = None, config: Config = None,
) -> dict:
    """
    Compute per-token causal effects for a response.

    Factual: p(token_t | history + query)  -- model sees user preferences
    Counterfactual: p(token_t | query only) -- model has no user info

    delta_t = log p_factual - log p_counterfactual
    High delta_t = token was caused by the user's preferences
    """

    # Build factual prompt (with user history as context)
    if user_history:
        history_text = "\n".join(
            f"Q: {h['implicit_query']}\nPreferred: {h['aligned_op']}"
            for h in user_history[:3]
        )
        factual_messages = [
            {"role": "system", "content": (
                "You are a helpful assistant. Here is what you know about "
                f"this user's preferences:\n{history_text}\n\n"
                "Answer based on their preferences."
            )},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]
    else:
        factual_messages = [
            {"role": "system", "content": "Answer the user's question."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]

    # Build counterfactual prompt (no user history)
    counterfactual_messages = [
        {"role": "system", "content": "Answer the user's question."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ]

    # Tokenize both
    factual_text = tokenizer.apply_chat_template(
        factual_messages, tokenize=False
    )
    counter_text = tokenizer.apply_chat_template(
        counterfactual_messages, tokenize=False
    )

    factual_ids = tokenizer(factual_text, return_tensors="pt").to(model.device)
    counter_ids = tokenizer(counter_text, return_tensors="pt").to(model.device)

    # Get response token positions: tokenize just the response to find it
    response_token_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    resp_len = len(response_token_ids)

    with torch.no_grad():
        # Factual logits
        factual_out = model(**factual_ids)
        factual_logits = factual_out.logits[0]  # [seq_len, vocab]

        # Counterfactual logits
        counter_out = model(**counter_ids)
        counter_logits = counter_out.logits[0]

    # Get log probs for the response tokens at the end of each sequence
    # The response tokens are the last resp_len tokens
    factual_log_probs = F.log_softmax(factual_logits, dim=-1)
    counter_log_probs = F.log_softmax(counter_logits, dim=-1)

    # For each response token position, get the log prob of the actual token
    factual_seq_len = factual_ids["input_ids"].shape[1]
    counter_seq_len = counter_ids["input_ids"].shape[1]

    deltas = []
    tokens = []

    for i in range(resp_len):
        token_id = response_token_ids[i]

        # Position in factual sequence (response starts at end - resp_len)
        f_pos = factual_seq_len - resp_len - 1 + i
        c_pos = counter_seq_len - resp_len - 1 + i

        if f_pos < 0 or f_pos >= factual_seq_len or c_pos < 0 or c_pos >= counter_seq_len:
            continue

        f_logp = factual_log_probs[f_pos, token_id].item()
        c_logp = counter_log_probs[c_pos, token_id].item()

        delta = f_logp - c_logp
        deltas.append(delta)
        tokens.append(tokenizer.decode([token_id]))

    return {
        "tokens": tokens,
        "deltas": deltas,
        "response_token_ids": response_token_ids,
    }


def identify_anchor_tokens(
    causal_result: dict, threshold: float = 0.5
) -> dict:
    """
    Identify preference-driven anchor tokens using the causal threshold.
    Returns the anchor tokens and their positions.
    """
    tokens = causal_result["tokens"]
    deltas = causal_result["deltas"]

    anchor_tokens = []
    anchor_positions = []
    boilerplate_tokens = []

    for i, (tok, d) in enumerate(zip(tokens, deltas)):
        if d >= threshold:
            anchor_tokens.append(tok)
            anchor_positions.append(i)
        else:
            boilerplate_tokens.append(tok)

    return {
        "anchor_tokens": anchor_tokens,
        "anchor_positions": anchor_positions,
        "boilerplate_tokens": boilerplate_tokens,
        "n_anchor": len(anchor_tokens),
        "n_total": len(tokens),
        "pct_anchor": 100 * len(anchor_tokens) / max(len(tokens), 1),
    }


def get_anchor_hidden_states(
    model, tokenizer, response: str, anchor_positions: list[int],
    layer: int, config: Config,
) -> torch.Tensor | None:
    """
    Get hidden states at the steering layer for ONLY the anchor token positions.
    This gives us a preference-focused representation instead of mean-pooling
    over the entire (noisy) response.
    """
    if not anchor_positions:
        return None

    inputs = tokenizer(response, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    h = out.hidden_states[layer][0]  # [seq_len, hidden_dim]

    # Only take hidden states at anchor positions
    valid_positions = [p for p in anchor_positions if p < h.shape[0]]
    if not valid_positions:
        return None

    anchor_h = h[valid_positions]  # [n_anchor, hidden_dim]
    return anchor_h.mean(dim=0).detach().cpu().float()  # [hidden_dim]


def smooth_anchor_tokens(
    model, tokenizer, anchor_tokens: list[str], preference: str,
    config: Config,
) -> str:
    """
    SteerX smoothing step: transform discrete anchor tokens into a
    coherent preference description using the LLM.
    """
    token_str = " ".join(anchor_tokens)

    messages = [
        {"role": "system", "content": (
            "Given a set of preference-related keywords extracted from a "
            "user's responses, write a single concise sentence describing "
            "the user's preference style. Be specific and concrete."
        )},
        {"role": "user", "content": (
            f"Keywords: {token_str}\n"
            f"Context: {preference[:200]}\n"
            "Preference description:"
        )},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=100,
            temperature=0.3, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
