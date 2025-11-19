# -*- coding: utf-8 -*-
import torch
import re

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from src.model import SFLM
from src.utils import TokenizerABC


def get_config() -> Namespace:
    parser = ArgumentParser(
        usage="Simply run 'python3 infer.py' would load model from default path.",
        description="Inference script to use a saved language model.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./model.sav",
        help="The path of the model save file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature for sampling from the model. For greedy strategy, just set 0.0.",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable cuda."
    )
    config = parser.parse_args()
    vars(config)["device"] = (
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
    )
    return config


def RSPL(config: Namespace, model: SFLM, tokenizer: TokenizerABC) -> None:
    max_length = model.block_size - 1
    print(
        f"Entering Read-Generate-Print-Loop...\n"
        f"Type any prompt consists of 'a's, 'b's, or 'c's.\n"
        f"The maximum length should be {max_length}.\n"
        f"Type quit to exit.\n"
    )
    while True:
        prompt = input("Your prompt:")
        if prompt == "quit":
            break

        if re.sub("a|b|c", "", prompt) != "":
            print(
                "The input contains invalid character,\n"
                "please input string only contains 'a's, 'b's, or 'c's!\n"
            )
            continue
        if len(prompt) > max_length:
            print("The input exceeds the length limit!\n")
            continue
        cond_idx = tokenizer.encode(prompt)[:-1]  # Remove the tailing EOS.
        cond_idx = cond_idx.to(config.device)

        gen_idx = model.generate(
            cond_idx.view(1, -1),
            steps=model.block_size - cond_idx.size(-1) + 1,
            temperature=config.temperature,
        )
        gen_str = tokenizer.decode(gen_idx.view(-1), truncate=True, remove_special_token=True)
        na, nb, nc = gen_str.count("a"), gen_str.count("b"), gen_str.count("c")
        err = abs(na + nb - nc)
        print(
            f"SFLM:{gen_str + ' ':<{max_length+1}} "
            f"|#a+#b-#c|:{err:>2}\n"
        )
    print(
        f"Quitting Read-Generate-Print-Loop...\n"
    )


if __name__ == "__main__":
    config = get_config()
    tokenizer = TokenizerABC()
    save = torch.load(config.save_path, map_location=config.device, weights_only=True)

    model_config = save["model_config"]
    model = SFLM(**model_config)
    model.load_state_dict(save["state_dict"])
    print(
        f"Model loaded\n"
        f"vocabulary size: {model_config['vocab_size']}\n"
        f"embedding dimension: {model_config['emb_dim']}\n"
        f"block size: {model_config['block_size']}\n"
        f"#parameters: {sum([para.numel() for para in model.parameters()])}\n"
    )
    model.to(config.device)

    RSPL(config, model, tokenizer)
