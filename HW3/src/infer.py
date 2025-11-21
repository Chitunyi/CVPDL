import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

from train import NoiseScheduler, UNet2D


@torch.no_grad()
def ddpm_sample(model, noise_scheduler, num_samples, batch_size, device):
    """
    model: Input (x_t, t) / Output Noise, eps_theta
    noise_scheduler: same as training, NoiseScheduler
    num_samples:  10000
    """
    model.eval()

    T = noise_scheduler.num_train_timesteps
    betas = noise_scheduler.betas.to(device)
    alphas = noise_scheduler.alphas.to(device)
    alpha_bars = noise_scheduler.alphas_cumprod.to(device)

    all_images = []

    pbar = tqdm(range(0, num_samples, batch_size), desc="Sampling", ncols=120)
    for start in pbar:
        bs = min(batch_size, num_samples - start)

        # x_T ~ N(0, I)
        x = torch.randn(bs, 3, 28, 28, device=device)

        # Invert diffusion: T-1 ... 0
        for t in reversed(range(T)):
            t_batch = torch.full((bs,), t, device=device, dtype=torch.long)

            # Predict noise eps_theta(x_t, t)
            eps_theta = model(x, t_batch)

            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]
            if t > 0:
                alpha_bar_prev = alpha_bars[t - 1]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # DDPM eq. p(x_{t-1} | x_t)
            # μ_t = 1/sqrt(alpha_t) * ( x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta )
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mean = coef1 * (x - coef2 * eps_theta)

            if t > 0:
                # posterior variance:
                # σ_t^2 = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
                var = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean

        x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
        all_images.append(x.cpu())

    return torch.cat(all_images, dim=0)[:num_samples]


def load_model(ckpt_path, time_emb_dim, device, num_channels=3):

    model = UNet2D(
        in_ch=num_channels,
        out_ch=num_channels,
        base_ch=128,             
        channel_mults=(1, 2, 4), 
        time_emb_dim=time_emb_dim,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)  
    return model



def save_images_to_folder(images, out_dir):
    """
    images: Tensor [N, 3, 28, 28], in range [0,1]
    save as 00001.png ~ N.png, in RGB
    """
    os.makedirs(out_dir, exist_ok=True)

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    for idx, img in enumerate(images, start=1):
        filename = f"{idx:05d}.png"
        path = os.path.join(out_dir, filename)
        save_image(img, path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_timesteps", type=int, default=800, help="must match training NoiseScheduler")
    parser.add_argument("--time_emb_dim", type=int, default=256, help="must match training SimpleUNet time_emb_dim")
    parser.add_argument("--ckpt", type=str, required=True, help="path to trained model checkpoint (.pth)")
    parser.add_argument("--num_samples", type=int, default=10000, help="number of images to generate")
    parser.add_argument("--batch_size", type=int, default=256, help="sampling batch size")
    parser.add_argument("--out_dir", type=str, default="./generated", help="directory to save generated images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device for sampling")

    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    # noise_scheduler = NoiseScheduler(num_timesteps=args.num_timesteps)
    noise_scheduler = NoiseScheduler(
        num_train_timesteps=args.num_timesteps,
        beta_start=1e-4,
        beta_end=0.02,
        device=device,
    )

    model = load_model(args.ckpt, args.time_emb_dim, device)

    images = ddpm_sample(
        model=model,
        noise_scheduler=noise_scheduler,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
    )

    save_images_to_folder(images, args.out_dir)

    print(f"Saved {len(images)} images to {args.out_dir}")


if __name__ == "__main__":
    main()
