"""
Per-user steering vectors.

Each user gets a single vector u ∈ R^d (same dimension as the model's
hidden states at the target layer). This vector encodes what kind of
traces are helpful for that user.

Update rule (difference of means):
    u_update = normalize( mean(h_positive) - mean(h_negative) )

Smoothed with EMA:
    u = (1 - beta) * u_old + beta * u_update
    u = normalize(u)

Applied at inference as a forward hook that adds α * u to the hidden
states at the target layer, nudging the model toward "helpful trace"
territory for that user.
"""

import torch
import torch.nn.functional as F
from config import Config


class SteeringVector:
    def __init__(self, user_id: str, hidden_dim: int, config: Config):
        self.user_id = user_id
        self.config = config
        self.hidden_dim = hidden_dim

        # The steering vector — starts as zeros (no steering)
        self.u = torch.zeros(hidden_dim)

        # Track stats for logging
        self.num_updates = 0
        self.history: list[dict] = []

    def update(self, positive_hiddens: list[torch.Tensor],
               negative_hiddens: list[torch.Tensor]):
        """
        Update the steering vector from a batch of positive/negative
        hidden states.

        positive_hiddens: list of [hidden_dim] tensors from helpful traces
        negative_hiddens: list of [hidden_dim] tensors from unhelpful traces

        If either list is empty, skip the update — we need contrast.
        """
        if len(positive_hiddens) == 0 or len(negative_hiddens) == 0:
            self.history.append({
                "round": self.num_updates,
                "skipped": True,
                "reason": f"pos={len(positive_hiddens)}, neg={len(negative_hiddens)}",
            })
            return

        # Stack into tensors and compute means
        pos_mean = torch.stack(positive_hiddens).mean(dim=0)
        neg_mean = torch.stack(negative_hiddens).mean(dim=0)

        # Difference of means: the direction from "unhelpful" to "helpful"
        u_update = pos_mean - neg_mean

        # Normalize to unit vector
        u_update = F.normalize(u_update, dim=0)

        # EMA: blend with previous vector for stability
        if self.num_updates == 0:
            # First update — just use the raw direction
            self.u = u_update
        else:
            beta = self.config.ema_beta
            self.u = (1 - beta) * self.u + beta * u_update
            self.u = F.normalize(self.u, dim=0)

        self.num_updates += 1

        # Log
        cos_sim = F.cosine_similarity(
            self.u.unsqueeze(0), u_update.unsqueeze(0)
        ).item()
        self.history.append({
            "round": self.num_updates,
            "skipped": False,
            "n_pos": len(positive_hiddens),
            "n_neg": len(negative_hiddens),
            "cos_sim_with_update": cos_sim,
            "u_norm": self.u.norm().item(),
        })

        print(f"  [{self.user_id}] Updated steering vector: "
              f"{len(positive_hiddens)} pos, {len(negative_hiddens)} neg, "
              f"cos_sim={cos_sim:.3f}")

    def make_hook(self):
        """
        Returns a forward hook function that adds the steering direction
        to hidden states at the target layer during generation.

        The hook modifies: h = h + alpha * (u · h) * u
        This projects h further along the u direction — amplifying the
        "helpful trace" component without changing orthogonal dimensions.
        """
        alpha = self.config.steer_alpha
        u = self.u.clone()

        def hook(module, input, output):
            # Qwen layers return the hidden state tensor directly,
            # not wrapped in a tuple.
            if isinstance(output, torch.Tensor):
                h = output
            else:
                h = output[0]

            device = h.device
            u_dev = u.to(device=device, dtype=h.dtype)

            # h is [batch, seq_len, hidden_dim]
            proj = torch.einsum("...d,d->...", h, u_dev).unsqueeze(-1)
            h_steered = h + alpha * proj * u_dev

            if isinstance(output, torch.Tensor):
                return h_steered
            else:
                return (h_steered,) + tuple(output[i] for i in range(1, len(output)))

        return hook

    def save(self, path: str):
        torch.save({
            "user_id": self.user_id,
            "u": self.u,
            "num_updates": self.num_updates,
            "history": self.history,
        }, path)

    @classmethod
    def load(cls, path: str, config: Config) -> "SteeringVector":
        data = torch.load(path, weights_only=False)
        sv = cls(data["user_id"], data["u"].shape[0], config)
        sv.u = data["u"]
        sv.num_updates = data["num_updates"]
        sv.history = data["history"]
        return sv
