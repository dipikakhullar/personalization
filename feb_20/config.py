from dataclasses import dataclass


@dataclass
class Config:
    # ── Model ───────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # ── Steering ────────────────────────────────────────────────
    steer_layer: int = 24
    steer_alpha: float = 1.5
    ema_beta: float = 0.15

    # ── Generation ──────────────────────────────────────────────
    max_tokens: int = 256
    temperature: float = 0.7

    # ── Experiment ──────────────────────────────────────────────
    # We iterate through all 1000 dataset samples in batches.
    # After each batch, we update the steering vector.
    batch_size: int = 10  # samples per update
    num_responses: int = 2  # responses to generate per sample
