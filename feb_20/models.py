"""
Model loading and response generation.
Each sample's preference is passed as system context so the model
knows what the user wants.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config


def load_model(config: Config):
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return {"model": model, "tokenizer": tokenizer}


def generate_response(sample: dict, models: dict, config: Config,
                      steering_hook=None) -> tuple[str, torch.Tensor]:
    """
    Generate a response to a sample's implicit_query.
    The sample's preference is baked into the system prompt so the
    model has the user context.
    Returns (response_text, hidden_state).
    """
    tok = models["tokenizer"]
    model = models["model"]

    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. The user has the following preference:\n"
            f"{sample['preference']}\n\n"
            "Answer their question with a specific recommendation that "
            "respects their preference."
        )},
        {"role": "user", "content": sample["implicit_query"]},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    target_layer = model.model.layers[config.steer_layer]
    steer_handle = None
    if steering_hook is not None:
        steer_handle = target_layer.register_forward_hook(steering_hook)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
        )

    if steer_handle is not None:
        steer_handle.remove()

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tok.decode(new_tokens, skip_special_tokens=True)

    with torch.no_grad():
        out = model(output_ids, output_hidden_states=True)
    h = out.hidden_states[config.steer_layer]
    h = h.mean(dim=1).squeeze(0).detach().cpu().float()

    return response, h
