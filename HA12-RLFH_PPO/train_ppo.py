# -*- coding: utf-8 -*-
"""
HA12 PPO + RLHF Fine-tuning for SFLM
Student Name: Liu Shanru
Student ID: 21190664
Student Email: sliufo@connect.ust.hk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader

from src.model import SFLM
from src.utils import TokenizerABC, LangABC


# ======================================================
# Simulated Human Feedback
# ======================================================
def reward_fn(sample_str: str) -> float:
    """
    Simulated human feedback:
    Returns a reward of +1.0 if the string does not contain "ac",
    otherwise returns a penalty of -1.0.
    """
    return -1.0 if "ac" in sample_str else 1.0


# ======================================================
# Critic Network Definition
# ======================================================
class ValueNet(nn.Module):
    """
    Simple critic network for estimating value given an embedding.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the value network.
        Args:
            x (torch.Tensor): Tensor of shape (N, L, D)
        Returns:
            torch.Tensor: Value estimates of shape (N, L)
        """
        return self.net(x).squeeze(-1)


# ======================================================
# Command-line Configuration
# ======================================================
def get_config() -> Namespace:
    """
    Parse command-line arguments for PPO fine-tuning.
    """
    parser = ArgumentParser(
        usage="Run 'python3 train_ppo.py' for fine-tuning using PPO RLHF.",
        description="PPO-based RLHF fine-tuning for SFLM (no 'ac' allowed).",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset-size", type=int, default=50000,
                        help="Dummy dataset size (unused, kept for compatibility).")
    parser.add_argument("--str-len", type=int, default=20,
                        help="Maximum string length (excluding BOS/EOS).")
    parser.add_argument("--embed-dim", type=int, default=32,
                        help="Embedding dimension of the transformer.")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Number of PPO fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for PPO updates.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.0,
                        help="Weight decay.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for rewards.")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="PPO clipping epsilon.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature.")
    parser.add_argument("--save-path", type=str, default="./model_rlhf.sav",
                        help="Path to save the fine-tuned model checkpoint.")
    parser.add_argument("--pretrained-path", type=str, default="./model.sav",
                        help="Path to the pretrained supervised model.")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="Disable CUDA and use CPU only.")

    config = parser.parse_args()
    vars(config)["device"] = (
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
    )
    return config


# ======================================================
# PPO Training Loop
# ======================================================
def train(config: Namespace, model: SFLM, critic: ValueNet,
          model_opt: opt.Optimizer, critic_opt: opt.Optimizer,
          tokenizer: TokenizerABC, epoch: int) -> None:
    """
    Perform a single PPO training epoch using simulated human feedback.
    """
    model.train()
    critic.train()

    total_loss, total_reward = 0, 0

    for _ in range(config.batch_size):
        # --------------------------------------------------
        # Generate a sample sequence from the model
        # --------------------------------------------------
        with torch.no_grad():
            start = tokenizer.encode("")[:-1].to(config.device).view(1, -1)
            gen_idx = model.generate(start, steps=config.str_len, temperature=config.temperature)[0]
            gen_str = tokenizer.decode(gen_idx, truncate=True, remove_special_token=True)

        # --------------------------------------------------
        # Compute reward from simulated human feedback
        # --------------------------------------------------
        R = reward_fn(gen_str)
        total_reward += R

        # --------------------------------------------------
        # PPO update step
        # --------------------------------------------------
        input_idx = gen_idx[:-1].unsqueeze(0)
        logits = model(input_idx)
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_logprob = torch.gather(
            log_probs, 2, gen_idx[1:].unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)

        values = critic(model.tok_embedding(input_idx))
        old_logprob = chosen_logprob.detach()
        advantage = R - values.mean().item()

        ratio = torch.exp(chosen_logprob - old_logprob)
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * advantage
        actor_loss = -torch.min(unclipped, clipped).mean()
        critic_loss = F.mse_loss(values.mean(), torch.tensor(R, device=config.device))
        loss = actor_loss + 0.5 * critic_loss

        model_opt.zero_grad()
        critic_opt.zero_grad()
        loss.backward()
        model_opt.step()
        critic_opt.step()

        total_loss += loss.item()

    print(f"Epoch {epoch:04d} | Loss: {total_loss / config.batch_size:.4f} | Avg Reward: {total_reward / config.batch_size:.2f}")


# ======================================================
# Evaluation
# ======================================================
def eval(config: Namespace, model: SFLM, tokenizer: TokenizerABC, n_samples: int = 10):
    """
    Evaluate the model by generating several samples and displaying
    whether each contains the forbidden pattern 'ac'.
    """
    model.eval()
    samples = []
    for _ in trange(n_samples, desc="Evaluation", ncols=100):
        cond_idx = tokenizer.encode("")[:-1].to(config.device)
        gen_idx = model.generate(cond_idx.view(1, -1), steps=config.str_len, temperature=0.0).view(-1)
        gen_str = tokenizer.decode(gen_idx, truncate=True, remove_special_token=True)
        samples.append(gen_str)

    print("\nSamples:")
    for s in samples[:10]:
        print(f"\t{s}  | contains 'ac': {'ac' in s}")


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    config = get_config()
    tokenizer = TokenizerABC()

    # --------------------------------------------------
    # Initialize model and load pretrained supervised weights
    # --------------------------------------------------
    print("Loading pretrained supervised model as PPO baseline...")
    checkpoint = torch.load(config.pretrained_path, map_location=config.device)

    model_config = checkpoint.get("model_config", dict(
        vocab_size=tokenizer.num_tokens,
        emb_dim=config.embed_dim,
        block_size=config.str_len + 1,
    ))

    model = SFLM(**model_config).to(config.device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded pretrained baseline from {config.pretrained_path}")

    # Initialize critic network
    critic = ValueNet(model_config["emb_dim"]).to(config.device)

    # Optimizers for actor (model) and critic
    model_opt = opt.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    critic_opt = opt.AdamW(critic.parameters(), lr=config.lr)

    # --------------------------------------------------
    # PPO Fine-tuning
    # --------------------------------------------------
    print("==== Start PPO fine-tuning with simulated human feedback ====")
    for epoch in range(1, config.epochs + 1):
        train(config, model, critic, model_opt, critic_opt, tokenizer, epoch)
        if epoch % 100 == 0:
            eval(config, model, tokenizer, n_samples=10)
            print("")

    # --------------------------------------------------
    # Save fine-tuned model
    # --------------------------------------------------
    torch.save(
        dict(
            model_config=model_config,
            state_dict=model.state_dict(),
            critic_state=critic.state_dict(),
        ),
        config.save_path,
    )

    print(f"PPO fine-tuned model saved at {config.save_path}")