# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.optim as opt

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from tqdm.auto import tqdm, trange
from torch.optim import Optimizer
from src.model import SFLM
from src.utils import TokenizerABC, LangABC
from torch.utils.data import DataLoader


def get_config() -> Namespace:
    parser = ArgumentParser(
        usage="Simply run 'python3 train.py' should be enough.",
        description="Training script for a very tiny language model.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1_000_000,
        help="The dummy size of the dataset (as samples are generated nondeterministically).",
    )
    parser.add_argument(
        "--str-len",
        type=int,
        default=20,
        help="The length of strings sampled from the dataset.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=16,
        help="The dimension of embeddings applied in the model.",
    )

    parser.add_argument(
        "--epochs", type=int, default=20, help="The # of 'epochs' of training."
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable cuda."
    )
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate.")
    parser.add_argument("--wd", type=float, default=0.0, help="The weight decay.")
    parser.add_argument("--batch-size", type=int, default=64, help="The batch size.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="./model.sav",
        help="The path of the model save file.",
    )

    config = parser.parse_args()
    vars(config)["device"] = (
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
    )
    return config


def train(
    config: Namespace, model: SFLM, opt: Optimizer, dataloader: DataLoader, epoch: int
) -> None:
    loader = tqdm(
        dataloader,
        desc=f"Epoch {epoch:0>4}/{config.epochs:0>4} Training",
        total=len(dataloader),
        ncols=100
    )
    for idx in loader:
        idx = idx.to(config.device)
        # Apply Teacher Forcing.
        input_idx = idx[:, :-1]
        predict_target = idx[:, 1:]

        logit = model(input_idx)
        loss = F.cross_entropy(logit.flatten(0, 1), predict_target.flatten(0, 1))

        opt.zero_grad()
        loss.backward()
        opt.step()
        loader.set_postfix_str(f"loss: {loss.item():7.4f}")


def eval(
    config: Namespace,
    model: SFLM,
    tokenizer: TokenizerABC,
    dataset: LangABC,
    n_samples: int,
) -> None:
    conditions, samples = [], []
    for _ in trange(n_samples, desc=f"Epoch {epoch:0>4}/{config.epochs:0>4} Eval", ncols=100):
        cond_idx = tokenizer.encode("")[:-1] #Just a BOS
        cond_idx = cond_idx.to(config.device)

        gen_idx = model.generate(
            cond_idx.view(1, -1), config.str_len + 1
        ).view(-1)
        gen_str = tokenizer.decode(gen_idx, truncate=True, remove_special_token=True)

        samples.append(gen_str)

    def err_fn(s: str) -> int:
        return abs(s.count("a") + s.count("b") - s.count("c"))

    n_print = 10
    print(f"Epoch: {epoch:0>4}")
    width_str = 25
    width_num = 15
    print(
        f"\t "
        f"{'Sample:':<{width_str}} "
        f"{'|#a + #b - #c|':<{width_num}} "
    )
    print(
        f"\t "
        f"{'':-<{width_str}} "
        f"{'':-<{width_num}} "
    )
    for sample in samples[:n_print]:
        print(
            f"\t "
            f"{sample:<{width_str}} "
            f"{err_fn(sample):<{width_num}} "
        )
    print(
        f"\t "
        f"{'...':<{width_str}} "
        f"{'...':<{width_num}}"
    )
    mean_err = sum(map(err_fn, samples)) / n_samples
    print(f"Mean |#a + #b - #c|: {mean_err:7.4f}")


if __name__ == "__main__":
    config = get_config()
    tokenizer = TokenizerABC()
    dataset = LangABC(
        config.dataset_size,
        config.str_len,
        transform=lambda s: tokenizer.encode(s, pad_to_length=config.str_len + 2),
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size)

    model_config = dict(
        vocab_size=tokenizer.num_tokens,
        emb_dim=config.embed_dim,
        block_size=config.str_len + 1,
    )
    model = SFLM(**model_config)
    model.to(config.device)

    optimizer = opt.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)

    for epoch in range(1, config.epochs + 1):
        train(config, model, optimizer, dataloader, epoch)
        eval(config, model, tokenizer, dataset, 1000)
        print()
    torch.save(
        dict(model_config=model_config, state_dict=model.state_dict()), config.save_path
    )
    print(f"Model saved as {config.save_path}")
