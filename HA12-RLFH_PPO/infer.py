# -*- coding: utf-8 -*-
"""
HA12 PPO + RLHF Fine-tuning for SFLM
Student Name: Liu Shanru
Student ID: 21190664
Student Email: sliufo@connect.ust.hk
"""
import torch
import re
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from src.model import SFLM
from src.utils import TokenizerABC


# ======================================================
# Command-line configuration
# ======================================================
def get_config() -> Namespace:
    """
    Parse command-line arguments for inference.

    Returns:
        Namespace: Configuration object including model path, temperature,
                   retry count, and device settings.
    """
    parser = ArgumentParser(
        usage="Run 'python3 infer.py' to interactively generate strings.",
        description="Interactive inference for the SFLM model with silent 'ac' post-filtering.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./model_rlhf.sav",
        help="Path to the trained model checkpoint file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 for deterministic greedy decoding).",
    )
    parser.add_argument(
        "--max-retry",
        type=int,
        default=5,
        help="Maximum number of silent retries if the generated string contains 'ac'.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disable CUDA and force CPU inference.",
    )

    config = parser.parse_args()
    vars(config)["device"] = (
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
    )
    return config


# ======================================================
# Inference loop: interactive generation with silent retry
# ======================================================
def RSPL(config: Namespace, model: SFLM, tokenizer: TokenizerABC) -> None:
    """
    Interactive Read-Sample-Print Loop (RSPL).

    Silently re-generates outputs until one without the substring 'ac' is found.
    No retry messages are shown to the user.

    Args:
        config (Namespace): Configuration object from get_config().
        model (SFLM): Trained sequence model.
        tokenizer (TokenizerABC): Tokenizer for encoding and decoding.
    """
    max_length = model.block_size - 1
    print(
        f"\nEntering Read-Generate-Print Loop...\n"
        f"Type a prompt consisting only of 'a', 'b', or 'c'.\n"
        f"Maximum sequence length: {max_length}\n"
        f"Type 'quit' to exit.\n"
        f"(Silent post-filtering is active; model will internally retry until 'ac' is avoided)\n"
    )

    while True:
        prompt = input("Your prompt: ").strip()
        if prompt.lower() == "quit":
            break

        # -------------------------------------------------
        # Input validation
        # -------------------------------------------------
        if re.sub("[abc]", "", prompt) != "":
            print("Invalid input. Only characters 'a', 'b', and 'c' are allowed.\n")
            continue
        if len(prompt) > max_length:
            print(f"Input too long. (Maximum length = {max_length})\n")
            continue

        # -------------------------------------------------
        # Generate sequence with silent retries
        # -------------------------------------------------
        cond_idx = tokenizer.encode(prompt)[:-1].to(config.device)
        final_str = None
        start_time = time.time()

        for _ in range(config.max_retry):
            gen_idx = model.generate(
                cond_idx.view(1, -1),
                steps=model.block_size - cond_idx.size(-1) + 1,
                temperature=config.temperature,
            )
            gen_str = tokenizer.decode(
                gen_idx.view(-1),
                truncate=True,
                remove_special_token=True,
            )
            if "ac" not in gen_str:
                final_str = gen_str
                break
            # Silent retry: no output messages displayed to the user

        # If all retries still contain 'ac', accept the final result
        if final_str is None:
            final_str = gen_str

        elapsed = time.time() - start_time

        # -------------------------------------------------
        # Compute metrics and display results
        # -------------------------------------------------
        na, nb, nc = final_str.count("a"), final_str.count("b"), final_str.count("c")
        err = abs(na + nb - nc)
        contains_ac = "No" if "ac" not in final_str else "Yes"

        print(
            f"SFLM: {final_str + ' ':<{max_length + 1}} "
            f"| |#a + #b - #c| = {err:>2}\n "
        )

    print("Exiting Read-Generate-Print Loop...\n")


# ======================================================
# Main entry point
# ======================================================
if __name__ == "__main__":
    config = get_config()
    tokenizer = TokenizerABC()

    print(f"Loading model from {config.save_path} ...")
    checkpoint = torch.load(config.save_path, map_location=config.device)

    model_config = checkpoint["model_config"]
    model = SFLM(**model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(config.device)

    print(
        f"Model loaded successfully.\n"
        f"Vocabulary size: {model_config['vocab_size']}\n"
        f"Embedding dimension: {model_config['emb_dim']}\n"
        f"Block size: {model_config['block_size']}\n"
        f"Total parameters: {sum(p.numel() for p in model.parameters())}\n"
    )

    RSPL(config, model, tokenizer)
