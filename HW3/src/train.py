import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import math

from PIL import Image
from torch.utils.data import Dataset
import glob

class CustomMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        exts = ["*.png", "*.jpg", "*.jpeg"]
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(root_dir, ext)))
        paths.sort()
        self.image_paths = paths

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}. Check path and extension.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        dummy_label = 0
        return img, dummy_label

# class NoiseScheduler:
#     def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
#         self.num_timesteps = num_timesteps
        
#         self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
#         self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
#         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

class NoiseScheduler:
    def __init__(self, num_train_timesteps, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_train_timesteps = num_train_timesteps
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def add_noise(self, x0, noise, t):
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise


def forward_diffusion(x0, noise_scheduler, t, device):

    sqrt_alphas_cumprod = noise_scheduler.sqrt_alphas_cumprod.to(device)[t]
    sqrt_one_minus_alphas_cumprod = noise_scheduler.sqrt_one_minus_alphas_cumprod.to(device)[t]
    
    noise = torch.randn_like(x0)
    
    sqrt_alphas_cumprod = sqrt_alphas_cumprod[:, None, None, None]
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[:, None, None, None]
    
    noisy_x = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
    
    return noisy_x, noise


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.LongTensor):
        # t: [B], 0..T-1
        half = self.dim // 2
        device = t.device
        
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1: 
            emb = F.pad(emb, (0, 1))
        return emb  


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.use_skip = (in_ch != out_ch)
        if self.use_skip:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_emb):
        # t_emb: [B, time_emb_dim]
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        t = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        h = h + t

        h = self.conv2(h)
        h = self.norm2(h)

        if self.use_skip:
            x = self.skip(x)

        return self.act(h + x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.res1 = ResBlock(out_ch + skip_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x

class UNet2D(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        base_ch=128,
        channel_mults=(1, 2, 4),
        time_emb_dim=256,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # chs: 32, 64, 128 
        chs = [base_ch * m for m in channel_mults]

        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        # down: 28 -> 14 -> 7
        self.down1 = DownBlock(chs[0], chs[0], time_emb_dim)
        self.down2 = DownBlock(chs[0], chs[1], time_emb_dim)

        # mid
        self.mid1 = ResBlock(chs[1], chs[2], time_emb_dim)
        self.mid2 = ResBlock(chs[2], chs[2], time_emb_dim)

        # up: 7 -> 14 -> 28
        self.up1 = UpBlock(chs[2], chs[1], chs[1], time_emb_dim)
        self.up2 = UpBlock(chs[1], chs[0], chs[0], time_emb_dim)

        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(chs[0], out_ch, 3, padding=1)

    def forward(self, x, t):
        # t: [B] int
        t_emb = self.time_emb(t)
        t_emb = self.time_mlp(t_emb)

        x = self.in_conv(x)

        x, skip1 = self.down1(x, t_emb)  # 28->14
        x, skip2 = self.down2(x, t_emb)  # 14->7

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.up1(x, skip2, t_emb)    # 7->14
        x = self.up2(x, skip1, t_emb)    # 14->28

        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)
        return x  



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="The directory of the dataset")
    parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_timesteps", type=int, default=800, help="step of diffusion process")
    parser.add_argument("--time_emb_dim", type=int, default=256, help="dimension of time embedding",)
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--save_model_path", type=str, default="./img_gen/", help="path to save the trained model")
    return parser.parse_args()


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    # train_dataset = datasets.MNIST(
    #     root=args.data_dir,
    #     train=True,
    #     download=True,
    #     transform=transform
    # )
    train_dataset = CustomMNISTDataset(
        root_dir=args.data_dir,
        transform=transform
    )


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # noise_scheduler = NoiseScheduler(num_timesteps=args.num_timesteps)
    # model = SimpleUNet(time_emb_dim=args.time_emb_dim).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    noise_scheduler = NoiseScheduler(num_train_timesteps=args.num_timesteps, device=device)
    model = UNet2D(
        in_ch=3,
        out_ch=3,
        base_ch=128,                 
        channel_mults=(1, 2, 4),
        time_emb_dim=args.time_emb_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            ncols=120
        )

        for step, (images, _) in enumerate(pbar):
            images = images.to(device)

            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            t = torch.randint(
                low=0,
                high=args.num_timesteps,
                size=(images.size(0),),
                device=device
            )

            # forward diffusion 
            noisy_images, noise = forward_diffusion(images, noise_scheduler, t, device)

            # predict noise
            pred_noise = model(noisy_images, t)

            # MSE 
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if device == "cuda": 
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}"})
            else:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}"})
    
    os.makedirs(args.save_model_path, exist_ok=True)
    ckpt_path = os.path.join(args.save_model_path, "ddpm_mnist.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")



if __name__ == "__main__":
    args = parse_args()
    main(args)