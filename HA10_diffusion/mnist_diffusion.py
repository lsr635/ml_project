import argparse
import json
import math
import os
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================
# Network definitions
# =============================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / (half - 1))
        angles = x[:, None] * freqs[None]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.res1 = ResidualBlock(in_ch, out_ch, time_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.res1(x, t_emb)
        h = self.res2(h, t_emb)
        return F.avg_pool2d(h, 2), h


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, time_dim: int) -> None:
        super().__init__()
        self.res1 = ResidualBlock(in_ch + skip_ch, out_ch, time_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        return self.res2(x, t_emb)


class SimpleUNet(nn.Module):
    def __init__(self, img_ch: int = 1, base_ch: int = 64, time_dim: int = 256) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim // 2),
            nn.Linear(time_dim // 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.entry = nn.Conv2d(img_ch, base_ch, 3, padding=1)
        self.down1 = DownBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, time_dim)
        self.mid1 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim)
        self.up1 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 4, time_dim)
        self.up2 = UpBlock(base_ch * 2, base_ch, base_ch * 2, time_dim)
        self.exit = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, img_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = self.entry(x)
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)
        x = self.up1(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)
        return self.exit(x)


# =============================
# Diffusion process
# =============================

def get_beta_schedule(kind: str, timesteps: int) -> torch.Tensor:
    if kind == "linear":
        return torch.linspace(1e-4, 2e-2, timesteps)
    if kind == "cosine":
        s = 0.008
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        f = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        betas = 1 - f[1:] / f[:-1]
        return torch.clamp(betas, 1e-4, 0.999)
    if kind == "sigmoid":
        steps = torch.linspace(-6, 6, timesteps)
        betas = torch.sigmoid(steps) * (0.02 - 1e-4) + 1e-4
        return betas
    raise ValueError(f"unknown schedule {kind}")


class Diffusion:
    def __init__(self, beta_schedule: torch.Tensor, device: torch.device) -> None:
        self.device = device
        self.betas = beta_schedule.to(device)
        self.num_steps = self.betas.shape[0]
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.posterior_variance[0] = 1e-20
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)
        )

    def gather(self, a: torch.Tensor, t: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
        return a.gather(0, t).view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self.gather(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self.gather(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, model: nn.Module, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted = model(x_noisy, t.float())
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def p_sample_ddpm(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pred_noise = model(x, t.float())
        sqrt_alpha = self.gather(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus = self.gather(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        x0 = torch.clamp((x - sqrt_one_minus * pred_noise) / torch.clamp(sqrt_alpha, 1e-12, 1.0), -1.0, 1.0)
        coef1 = self.gather(self.posterior_mean_coef1, t, x.shape)
        coef2 = self.gather(self.posterior_mean_coef2, t, x.shape)
        mean = coef1 * x0 + coef2 * x
        var = self.gather(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        mask = (t > 0).float().view(-1, 1, 1, 1)
        return mean + mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, ...], sampler: str, latent_scale: float = 1.0) -> torch.Tensor:
        x = torch.randn(shape, device=self.device) * latent_scale
        timesteps = torch.arange(self.num_steps - 1, -1, -1, device=self.device)
        for step in timesteps:
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample_ddpm(model, x, t)
        return x


# =============================
# Utils
# =============================

def create_dataloaders(batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    fid_tf = transforms.ToTensor()
    train_ds = datasets.MNIST("datasets/MNIST", train=True, download=True, transform=train_tf)
    fid_ds = datasets.MNIST("datasets/MNIST", train=False, download=True, transform=fid_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    fid_loader = DataLoader(fid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, fid_loader


def to_zero_one(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1) / 2, 0, 1)


@dataclass
class Config:
    schedule: str
    timesteps: int
    sampler: str
    epochs: int
    batch_size: int
    lr: float
    seed: int
    fid_samples: int
    latent_far_scale: float
    out_dir: str


# =============================
# Train and evaluate (using pytorch-fid)
# =============================

def compute_fid_with_pytorch_fid(fake_imgs: torch.Tensor, real_loader: DataLoader) -> float:
    """Compute FID by temporarily saving samples and real MNIST images."""
    from torchvision.utils import save_image

    with tempfile.TemporaryDirectory() as tmp:
        path_real = Path(tmp) / "real"
        path_fake = Path(tmp) / "fake"
        path_real.mkdir()
        path_fake.mkdir()

        # 保存真实数据样本（前1000张）
        count_real = 0
        for imgs, _ in real_loader:
            for img in imgs:
                save_image(img, path_real / f"img_{count_real:05d}.png", normalize=True)
                count_real += 1
                if count_real >= 1000:
                    break
            if count_real >= 1000:
                break

        # 保存生成图像
        for i, img in enumerate(fake_imgs):
            save_image(img, path_fake / f"gen_{i:05d}.png")

        # 调用 pytorch-fid
        cmd = [
            "python", "-m", "pytorch_fid",
            str(path_real),
            str(path_fake),
            "--device", "cuda" if torch.cuda.is_available() else "cpu"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        txt = result.stdout.strip()
        print(txt)
        fid_value = float(txt.split()[-1]) if "FID:" in txt else 9999.0
        return fid_value


def train_one(cfg: Config, device: torch.device, train_loader: DataLoader, fid_loader: DataLoader) -> dict:
    set_seed(cfg.seed)
    model = SimpleUNet().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    betas = get_beta_schedule(cfg.schedule, cfg.timesteps)
    diffusion = Diffusion(betas, device)

    global_step = 0
    for epoch in range(cfg.epochs):
        running = 0.0
        steps = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            t = torch.randint(0, cfg.timesteps, (imgs.size(0),), device=device)
            loss = diffusion.p_losses(model, imgs, t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            running += loss.item()
            steps += 1
        avg_loss = running / max(steps, 1)
        print(f"epoch {epoch + 1}/{cfg.epochs} - loss: {avg_loss:.4f}")

    # --- 生成样本 ---
    model.eval()
    samples = []
    with torch.no_grad():
        remaining = cfg.fid_samples
        while remaining > 0:
            batch = min(cfg.batch_size, remaining)
            gen = diffusion.sample(model, (batch, 1, 28, 28), cfg.sampler)
            samples.append(to_zero_one(gen).cpu())
            remaining -= batch
    samples = torch.cat(samples)

    # --- 计算FID ---
    fid_score = compute_fid_with_pytorch_fid(samples, fid_loader)

    # --- 远离latent生成 ---
    bad_latent = diffusion.sample(model, (cfg.batch_size, 1, 28, 28), cfg.sampler, latent_scale=cfg.latent_far_scale)
    bad_latent = to_zero_one(bad_latent).cpu()
    far_fid = compute_fid_with_pytorch_fid(bad_latent, fid_loader)

    # 保存输出
    os.makedirs(cfg.out_dir, exist_ok=True)
    grid_path = os.path.join(cfg.out_dir, "samples.png")
    far_path = os.path.join(cfg.out_dir, "far_latent.png")
    vutils.save_image(samples[:64], grid_path, nrow=8)
    vutils.save_image(bad_latent[:64], far_path, nrow=8)

    state_path = os.path.join(cfg.out_dir, "model.pt")
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, state_path)

    metrics = {
        "schedule": cfg.schedule,
        "timesteps": cfg.timesteps,
        "sampler": cfg.sampler,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "fid": fid_score,
        "latent_far_scale": cfg.latent_far_scale,
        "far_fid": far_fid,
        "samples_path": grid_path,
        "far_latent_path": far_path,
        "model_path": state_path,
        "steps": global_step,
    }

    with open(os.path.join(cfg.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"{cfg.schedule} T={cfg.timesteps} {cfg.sampler} -> fid {fid_score:.3f} | far fid {far_fid:.3f}")
    return metrics


# =============================
# Main
# =============================

def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST diffusion experiments")
    parser.add_argument("--schedule", default="linear", choices=["linear", "cosine", "sigmoid"])
    parser.add_argument("--timesteps", type=int, default=400)
    parser.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fid-samples", type=int, default=1024)
    parser.add_argument("--latent-far-scale", type=float, default=5.0)
    parser.add_argument("--out", default="results")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, fid_loader = create_dataloaders(args.batch_size, num_workers=2)
    os.makedirs(args.out, exist_ok=True)

    setups = []
    if args.all:
        for schedule in ["linear", "cosine"]:
            for steps in [400, 1000]:
                for sampler in ["ddpm", "ddim"]:
                    setups.append((schedule, steps, sampler))
    else:
        setups = [(args.schedule, args.timesteps, args.sampler)]

    results = []
    for schedule, steps, sampler in setups:
        name = f"{schedule}_T{steps}_{sampler}"
        cfg = Config(
            schedule=schedule,
            timesteps=steps,
            sampler=sampler,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            fid_samples=args.fid_samples,
            latent_far_scale=args.latent_far_scale,
            out_dir=os.path.join(args.out, name),
        )
        metrics = train_one(cfg, device, train_loader, fid_loader)
        results.append(metrics)

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()


# import argparse
# import json
# import math
# import os
# from dataclasses import dataclass
# from typing import Iterable, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, utils as vutils
# from torchvision.models import inception_v3, Inception_V3_Weights
# from torchvision.models.feature_extraction import create_feature_extractor


# def set_seed(seed: int) -> None:
#     if seed < 0:
#         return
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim: int) -> None:
#         super().__init__()
#         self.dim = dim

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         half = self.dim // 2
#         freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / (half - 1))
#         angles = x[:, None] * freqs[None]
#         emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
#         if self.dim % 2:
#             emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
#         return emb


# class ResidualBlock(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
#         super().__init__()
#         self.norm1 = nn.GroupNorm(8, in_ch)
#         self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
#         self.norm2 = nn.GroupNorm(8, out_ch)
#         self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
#         self.time = nn.Linear(time_dim, out_ch)
#         self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

#     def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
#         h = self.conv1(F.silu(self.norm1(x)))
#         h = h + self.time(t_emb)[:, :, None, None]
#         h = self.conv2(F.silu(self.norm2(h)))
#         return h + self.skip(x)


# class DownBlock(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
#         super().__init__()
#         self.res1 = ResidualBlock(in_ch, out_ch, time_dim)
#         self.res2 = ResidualBlock(out_ch, out_ch, time_dim)

#     def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         h = self.res1(x, t_emb)
#         h = self.res2(h, t_emb)
#         return F.avg_pool2d(h, 2), h


# class UpBlock(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int, skip_ch: int, time_dim: int) -> None:
#         super().__init__()
#         self.res1 = ResidualBlock(in_ch + skip_ch, out_ch, time_dim)
#         self.res2 = ResidualBlock(out_ch, out_ch, time_dim)

#     def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
#         x = F.interpolate(x, scale_factor=2, mode="nearest")
#         x = torch.cat([x, skip], dim=1)
#         x = self.res1(x, t_emb)
#         return self.res2(x, t_emb)


# class SimpleUNet(nn.Module):
#     def __init__(self, img_ch: int = 1, base_ch: int = 64, time_dim: int = 256) -> None:
#         super().__init__()
#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(time_dim // 2),
#             nn.Linear(time_dim // 2, time_dim),
#             nn.SiLU(),
#             nn.Linear(time_dim, time_dim),
#         )
#         self.entry = nn.Conv2d(img_ch, base_ch, 3, padding=1)
#         self.down1 = DownBlock(base_ch, base_ch * 2, time_dim)
#         self.down2 = DownBlock(base_ch * 2, base_ch * 4, time_dim)
#         self.mid1 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim)
#         self.mid2 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim)
#         self.up1 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 4, time_dim)
#         self.up2 = UpBlock(base_ch * 2, base_ch, base_ch * 2, time_dim)
#         self.exit = nn.Sequential(
#             nn.GroupNorm(8, base_ch),
#             nn.SiLU(),
#             nn.Conv2d(base_ch, img_ch, 3, padding=1),
#         )

#     def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         t_emb = self.time_mlp(t)
#         x = self.entry(x)
#         x, skip1 = self.down1(x, t_emb)
#         x, skip2 = self.down2(x, t_emb)
#         x = self.mid1(x, t_emb)
#         x = self.mid2(x, t_emb)
#         x = self.up1(x, skip2, t_emb)
#         x = self.up2(x, skip1, t_emb)
#         return self.exit(x)


# def get_beta_schedule(kind: str, timesteps: int) -> torch.Tensor:
#     if kind == "linear":
#         return torch.linspace(1e-4, 2e-2, timesteps)
#     if kind == "cosine":
#         s = 0.008
#         steps = torch.arange(timesteps + 1, dtype=torch.float32)
#         f = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
#         betas = 1 - f[1:] / f[:-1]
#         return torch.clamp(betas, 1e-4, 0.999)
#     if kind == "sigmoid":
#         steps = torch.linspace(-6, 6, timesteps)
#         betas = torch.sigmoid(steps) * (0.02 - 1e-4) + 1e-4
#         return betas
#     raise ValueError(f"unknown schedule {kind}")


# class Diffusion:
#     def __init__(self, beta_schedule: torch.Tensor, device: torch.device) -> None:
#         self.device = device
#         self.betas = beta_schedule.to(device)
#         self.num_steps = self.betas.shape[0]
#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
#         self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
#         self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
#         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
#         self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
#         self.posterior_variance[0] = 1e-20
#         self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
#         self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)

#     def gather(self, a: torch.Tensor, t: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
#         return a.gather(0, t).view(-1, *([1] * (len(shape) - 1)))

#     def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
#         if noise is None:
#             noise = torch.randn_like(x_start)
#         return (
#             self.gather(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
#             + self.gather(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
#         )

#     def p_losses(self, model: nn.Module, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         noise = torch.randn_like(x_start)
#         x_noisy = self.q_sample(x_start, t, noise)
#         predicted = model(x_noisy, t.float())
#         return F.mse_loss(predicted, noise)

#     @torch.no_grad()
#     def predict_x0(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         pred_noise = model(x, t.float())
#         sqrt_alpha = self.gather(self.sqrt_alphas_cumprod, t, x.shape)
#         sqrt_one_minus = self.gather(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
#         return (x - sqrt_one_minus * pred_noise) / torch.clamp(sqrt_alpha, min=1e-12)

#     @torch.no_grad()
#     def p_sample_ddpm(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         x0 = torch.clamp(self.predict_x0(model, x, t), -1.0, 1.0)
#         coef1 = self.gather(self.posterior_mean_coef1, t, x.shape)
#         coef2 = self.gather(self.posterior_mean_coef2, t, x.shape)
#         model_mean = coef1 * x0 + coef2 * x
#         noise = torch.randn_like(x)
#         mask = (t > 0).float().view(-1, 1, 1, 1)
#         sigma = torch.sqrt(self.gather(self.posterior_variance, t, x.shape))
#         return model_mean + mask * sigma * noise

#     @torch.no_grad()
#     def p_sample_ddim(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
#         alpha = self.gather(self.alphas_cumprod, t, x.shape)
#         alpha_prev = self.gather(self.alphas_cumprod_prev, t, x.shape)
#         pred_noise = model(x, t.float())
#         sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=1e-12))
#         sqrt_one_minus = torch.sqrt(torch.clamp(1 - alpha, min=1e-12))
#         pred_x0 = torch.clamp((x - sqrt_one_minus * pred_noise) / sqrt_alpha, -1.0, 1.0)
#         sqrt_alpha_prev = torch.sqrt(torch.clamp(alpha_prev, min=1e-12))
#         sigma = eta * torch.sqrt(torch.clamp((1 - alpha_prev) / (1 - alpha), min=1e-12)) * torch.sqrt(torch.clamp(1 - alpha / alpha_prev, min=1e-12))
#         noise = torch.randn_like(x)
#         noise_coeff = torch.sqrt(torch.clamp(1 - alpha_prev - sigma ** 2, min=1e-12))
#         sample = sqrt_alpha_prev * pred_x0 + noise_coeff * pred_noise
#         mask = (t > 0).float().view(-1, 1, 1, 1)
#         return sample + mask * sigma * noise

#     @torch.no_grad()
#     def sample(self, model: nn.Module, shape: Tuple[int, ...], sampler: str, latent_scale: float = 1.0) -> torch.Tensor:
#         x = torch.randn(shape, device=self.device) * latent_scale
#         timesteps = torch.arange(self.num_steps - 1, -1, -1, device=self.device)
#         for step in timesteps:
#             t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
#             if sampler == "ddpm":
#                 x = self.p_sample_ddpm(model, x, t)
#             elif sampler == "ddim":
#                 x = self.p_sample_ddim(model, x, t)
#             else:
#                 raise ValueError(f"unknown sampler {sampler}")
#         return x


# class FIDScore:
#     def __init__(self, device: torch.device) -> None:
#         base = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
#         base.eval()
#         self.extractor = create_feature_extractor(base, {"avgpool": "feat"}).to(device)
#         self.extractor.eval()
#         for p in self.extractor.parameters():
#             p.requires_grad_(False)
#         self.device = device
#         self.resize = transforms.Resize((299, 299), antialias=True)

#     def preprocess(self, x: torch.Tensor) -> torch.Tensor:
#         x = torch.clamp(x, 0.0, 1.0)
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#         x = self.resize(x)
#         return x

#     @torch.no_grad()
#     def features_from_loader(self, loader: DataLoader) -> torch.Tensor:
#         feats = []
#         for imgs, _ in loader:
#             imgs = imgs.to(self.device, dtype=torch.float32)
#             imgs = self.preprocess(imgs)
#             out = self.extractor(imgs)["feat"].reshape(imgs.size(0), -1)
#             feats.append(out)
#         return torch.cat(feats)

#     @torch.no_grad()
#     def features_from_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
#         imgs = imgs.to(self.device, dtype=torch.float32)
#         imgs = self.preprocess(imgs)
#         out = self.extractor(imgs)["feat"].reshape(imgs.size(0), -1)
#         return out

#     def stats(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         mu = feats.mean(0)
#         diff = feats - mu
#         cov = diff.t() @ diff / (diff.size(0) - 1)
#         return mu, cov

#     def fid(self, mu1: torch.Tensor, cov1: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor) -> float:
#         cov_prod = cov1 @ cov2
#         eigvals, eigvecs = torch.linalg.eigh(cov_prod)
#         sqrt_cov_prod = eigvecs @ torch.diag(torch.sqrt(torch.clamp(eigvals, min=0))) @ eigvecs.t()
#         diff = mu1 - mu2
#         trace = torch.trace(cov1 + cov2 - 2 * sqrt_cov_prod.real)
#         return float(diff.dot(diff) + trace)


# def create_dataloaders(batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
#     train_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#     fid_tf = transforms.ToTensor()
#     train_ds = datasets.MNIST("datasets/MNIST", train=True, download=True, transform=train_tf)
#     fid_ds = datasets.MNIST("datasets/MNIST", train=False, download=True, transform=fid_tf)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#     fid_loader = DataLoader(fid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
#     return train_loader, fid_loader


# def to_zero_one(x: torch.Tensor) -> torch.Tensor:
#     return torch.clamp((x + 1) / 2, 0, 1)


# @dataclass
# class Config:
#     schedule: str
#     timesteps: int
#     sampler: str
#     epochs: int
#     batch_size: int
#     lr: float
#     seed: int
#     fid_samples: int
#     latent_far_scale: float
#     out_dir: str


# def train_one(cfg: Config, device: torch.device, train_loader: DataLoader, real_feats: Tuple[torch.Tensor, torch.Tensor], fid: FIDScore) -> dict:
#     set_seed(cfg.seed)
#     model = SimpleUNet().to(device)
#     model.train()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
#     betas = get_beta_schedule(cfg.schedule, cfg.timesteps)
#     diffusion = Diffusion(betas, device)
#     global_step = 0
#     for epoch in range(cfg.epochs):
#         running = 0.0
#         steps = 0
#         for imgs, _ in train_loader:
#             imgs = imgs.to(device)
#             t = torch.randint(0, cfg.timesteps, (imgs.size(0),), device=device)
#             loss = diffusion.p_losses(model, imgs, t)
#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()
#             global_step += 1
#             running += loss.item()
#             steps += 1
#         avg_loss = running / max(steps, 1)
#         print(f"epoch {epoch + 1}/{cfg.epochs} - loss: {avg_loss:.4f}")
#     samples = []
#     model.eval()
#     with torch.no_grad():
#         remaining = cfg.fid_samples
#         while remaining > 0:
#             batch = min(cfg.batch_size, remaining)
#             gen = diffusion.sample(model, (batch, 1, 28, 28), cfg.sampler)
#             samples.append(to_zero_one(gen).cpu())
#             remaining -= batch
#     samples = torch.cat(samples)
#     fake_feats = fid.features_from_tensor(samples)
#     mu_fake, cov_fake = fid.stats(fake_feats)
#     mu_real, cov_real = real_feats
#     fid_score = fid.fid(mu_real, cov_real, mu_fake, cov_fake)
#     bad_latent = diffusion.sample(model, (cfg.batch_size, 1, 28, 28), cfg.sampler, latent_scale=cfg.latent_far_scale)
#     bad_latent = to_zero_one(bad_latent).cpu()
#     bad_feats = fid.features_from_tensor(bad_latent)
#     mu_bad, cov_bad = fid.stats(bad_feats)
#     far_fid = fid.fid(mu_real, cov_real, mu_bad, cov_bad)
#     grid_path = os.path.join(cfg.out_dir, "samples.png")
#     far_path = os.path.join(cfg.out_dir, "far_latent.png")
#     os.makedirs(cfg.out_dir, exist_ok=True)
#     vutils.save_image(samples[:64], grid_path, nrow=8)
#     vutils.save_image(bad_latent[:64], far_path, nrow=8)
#     state_path = os.path.join(cfg.out_dir, "model.pt")
#     torch.save({"model": model.state_dict(), "config": cfg.__dict__}, state_path)
#     metrics = {
#         "schedule": cfg.schedule,
#         "timesteps": cfg.timesteps,
#         "sampler": cfg.sampler,
#         "epochs": cfg.epochs,
#         "batch_size": cfg.batch_size,
#         "fid": fid_score,
#         "latent_far_scale": cfg.latent_far_scale,
#         "far_fid": far_fid,
#         "samples_path": grid_path,
#         "far_latent_path": far_path,
#         "model_path": state_path,
#         "steps": global_step,
#     }
#     with open(os.path.join(cfg.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2)
#     print(f"{cfg.schedule} T={cfg.timesteps} {cfg.sampler} -> fid {fid_score:.3f} | far fid {far_fid:.3f}")
#     return metrics


# def main() -> None:
#     parser = argparse.ArgumentParser(description="MNIST diffusion experiments")
#     parser.add_argument("--schedule", default="linear", choices=["linear", "cosine", "sigmoid"])
#     parser.add_argument("--timesteps", type=int, default=400)
#     parser.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
#     parser.add_argument("--epochs", type=int, default=1)
#     parser.add_argument("--batch-size", type=int, default=128)
#     parser.add_argument("--lr", type=float, default=2e-4)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--fid-samples", type=int, default=1024)
#     parser.add_argument("--latent-far-scale", type=float, default=5.0)
#     parser.add_argument("--out", default="results")
#     parser.add_argument("--all", action="store_true")
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     train_loader, fid_loader = create_dataloaders(args.batch_size, num_workers=2)
#     fid_helper = FIDScore(device)
#     with torch.no_grad():
#         real_feats = fid_helper.features_from_loader(fid_loader)
#     real_stats = fid_helper.stats(real_feats)
#     del real_feats
#     os.makedirs(args.out, exist_ok=True)

#     if args.all:
#         setups = []
#         for schedule in ["linear", "cosine"]:
#             for steps in [400, 1000]:
#                 for sampler in ["ddpm", "ddim"]:
#                     setups.append((schedule, steps, sampler))
#     else:
#         setups = [(args.schedule, args.timesteps, args.sampler)]

#     results = []
#     for schedule, steps, sampler in setups:
#         name = f"{schedule}_T{steps}_{sampler}"
#         cfg = Config(
#             schedule=schedule,
#             timesteps=steps,
#             sampler=sampler,
#             epochs=args.epochs,
#             batch_size=args.batch_size,
#             lr=args.lr,
#             seed=args.seed,
#             fid_samples=args.fid_samples,
#             latent_far_scale=args.latent_far_scale,
#             out_dir=os.path.join(args.out, name),
#         )
#         metrics = train_one(cfg, device, train_loader, real_stats, fid_helper)
#         results.append(metrics)

#     with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)


# if __name__ == "__main__":
#     main()
