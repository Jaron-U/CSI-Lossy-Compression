import argparse
import torch
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.compress_modules import ResnetCompressor
from ema_pytorch import EMA
from process_data import load_data, cal_mean_std
import torch.nn.functional as F
import pandas as pd
import os
import math
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns 

ptpath = "params_cdc_ldm/CSI-l2-use_weight5-challenge_data_3-d32-t8193-b0.0128-x-cosine-01-float32-aux0/CSI-l2-use_weight5-challenge_data_3-d32-t8193-b0.0128-x-cosine-01-float32-aux0_0.pt"
parser = argparse.ArgumentParser(description="values from bash script")

parser.add_argument("--ckpt", type=str, default=ptpath) # ckpt path
parser.add_argument("--gamma", type=float, default=0.8) # noise intensity for decoding
parser.add_argument("--n_denoise_step", type=int, default=65) # number of denoising step
parser.add_argument("--device", type=int, default=0) # gpu device index
dim = 32
context_channels = 32
dim_mults = [1,2,3,4]
data_channels = 2
context_dim_mults=[1,2]
reverse_context_dim_mults=[2,1]
hyper_dim_mults=[2,2,2]

config = parser.parse_args()

def main(rank):
    
    denoise_model = Unet(
        dim=dim,
        channels=data_channels,
        context_channels=context_channels,
        dim_mults=dim_mults,
        context_dim_mults=context_dim_mults,
        embd_type="01",
    )

    context_model = ResnetCompressor(
        dim=dim,
        dim_mults=context_dim_mults,
        reverse_dim_mults=reverse_context_dim_mults,
        hyper_dims_mults=hyper_dim_mults,
        channels=data_channels,
        out_channels=context_channels,
    )

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        ae_fn=None,
        num_timesteps=8193,
        loss_type="l2",
        lagrangian=0.0032,
        pred_mode="x",
        aux_loss_weight=0,
        aux_loss_type="lpips",
        var_schedule="cosine",
        use_loss_weight=True,
        loss_weight_min=5,
        use_aux_loss_weight_schedule=False,
    )
    loaded_param = torch.load(
        config.ckpt,
        map_location=lambda storage, loc: storage,
    )
    ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
    ema.load_state_dict(loaded_param["ema"], strict=False)
    
    diffusion = ema.ema_model
    diffusion.to(rank)
    diffusion.eval()

    _, test_loader = load_data(
        1,
        num_workers=1,
        shuffle=False
    )

    writer = SummaryWriter('evaluateBlog/cdc')
    mean, std = cal_mean_std(test_loader, test_loader.__len__())
    mean = mean.to(rank)
    std = std.to(rank)


    for i, data in enumerate(test_loader):
        if i >= 10:
            break
        data = data.to(rank)
        data_norm = (data - mean) / std
        data_norm = data_norm.to(rank)
        compressed, bpp = diffusion.compress(
            data_norm,
            sample_steps=config.n_denoise_step,
            bpp_return_mean=True,
            init=torch.randn_like(data_norm) * config.gamma
        )
        mse = torch.nn.functional.mse_loss(data, compressed * std + mean)
        get_diff_heatmap(writer, data, compressed, mse, bpp, i)
    
    writer.close()

def get_diff_heatmap(writer, original, compressed, mse, bpp, step):
    abs_difference = torch.abs(original - compressed)[0]
    sq_difference = ((original - compressed) ** 2)[0]
    fig, axes = plt.subplots(4, 1, figsize=(10, 32))  # 四行一列

    sns.heatmap(abs_difference.cpu().numpy()[0], ax=axes[0], cmap="viridis")
    axes[0].set_title("Channel 1 - Absolute Difference Heatmap")

    sns.heatmap(sq_difference.cpu().numpy()[0], ax=axes[1], cmap="viridis")
    axes[1].set_title("Channel 1 - Squared Difference Heatmap")

    sns.heatmap(abs_difference.cpu().numpy()[1], ax=axes[2], cmap="viridis")
    axes[2].set_title("Channel 2 - Absolute Difference Heatmap")

    sns.heatmap(sq_difference.cpu().numpy()[1], ax=axes[3], cmap="viridis")
    axes[3].set_title("Channel 2 - Squared Difference Heatmap")

    writer.add_figure("Difference Heatmaps by Channel", fig, global_step=step)
    plt.close(fig)

    writer.add_scalar('MSE', mse.item(), step)
    writer.add_scalar('BPP', bpp, step)

if __name__ == "__main__":
    main(config.device)
