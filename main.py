import argparse
import sys
import time
from pathlib2 import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from net.VQVAE import VQVAE
from net.losses import FiedelityLoss, TVLoss_jit, BrightLoss
from utils.image_io import np_to_torch, torch_to_np, save_image, prepare_image
from utils.imresize import np_imresize
from utils.gaussian import gaussian_blur
from copy import deepcopy

torch.manual_seed(384)
torch.cuda.manual_seed_all(384)

def downsample(image):
    return F.avg_pool2d(image, kernel_size=32, stride=16, padding=0)

class EngineModule(nn.Module):
    def __init__(self, input_path, output_dir, device, num_iter=15000, show_every=1000, drop_tau=0.1,
                 drop_mod_every=10000, num_inf_iter=100, input_depth=8, n_scale=5, lr=2e-3, illum_threshold=False):
        super(EngineModule, self).__init__()
        print(f"Processing {input_path}")

        self.output_dir = output_dir
        self.num_iter = num_iter
        self.show_every = show_every
        self.drop_mod_every = drop_mod_every
        self.num_inf_iter = num_inf_iter
        self.learning_rate = lr
        self.illum_threshold = illum_threshold

        # Load input image
        self.image_name = Path(input_path).stem
        self.image = prepare_image(input_path)
        self.original_image = self.image.copy()

        # Resize if necessary
        factor = 1
        if self.image.shape[1] >= 800 or self.image.shape[2] >= 800:
            new_shape_x = self.image.shape[1] / factor - (self.image.shape[1] % 32) + 128
            new_shape_y = self.image.shape[2] / factor - (self.image.shape[2] % 32) + 128
            self.image = np_imresize(self.image, output_shape=(new_shape_x, new_shape_y))
            factor += 1

        self.image_torch = np_to_torch(self.image).float().to(device)
        self.illum_ref = downsample(self.image_torch.max(dim=1, keepdim=True)[0]).detach()
        self.fixed_illum = gaussian_blur(self.image_torch.max(dim=1, keepdim=True)[0], kernel_size=25, sigma=2.0)

        # Load models
        self.reflect_net = self.load_reflect_net(device)
        self.illum_net = self.load_illum_net(device)
        self.BNNet = nn.BatchNorm2d(128).to(device)
        self.BNNet_ill = nn.BatchNorm2d(128).to(device)

        # Initialize inputs
        image_mean, image_std = self.image.mean(), self.image.std()
        self.reflect_net_inputs = torch.randn((1, 128, self.image.shape[1] // 4, self.image.shape[2] // 4)).to(device)
        self.reflect_net_inputs = self.reflect_net_inputs * image_std + image_mean
        self.illum_net_inputs = self.reflect_net_inputs.clone()
        self.reflect_net_inputs.requires_grad = True
        self.illum_net_inputs.requires_grad = True

        self.gamma = torch.randn(1).to(device)
        self.gamma.requires_grad = True

        # Parameters for optimization
        self.parameters = [
            self.reflect_net_inputs,
            self.illum_net_inputs,
            self.gamma,
            *self.BNNet.parameters(),
            *self.BNNet_ill.parameters(),
        ]

    def load_reflect_net(self, device):
        ref_VQVAE = VQVAE(need_dropout=False)
        ckpt = torch.load('vqvae_560.pt')
        model_state_dict = ref_VQVAE.state_dict()
        for name, param in ckpt.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
            else:
                print(f"Warning: Parameter '{name}' not found in the model.")
        return ref_VQVAE.dec.to(device)

    def load_illum_net(self, device):
        illum_VQVAE = VQVAE(out_channels=1, need_sigmoid=True)
        return illum_VQVAE.dec.to(device)

    def forward(self):
        illum_input = self.BNNet_ill(self.illum_net_inputs)
        illum_out = self.illum_net(illum_input).clamp(min=0.0) * 0.95 + 0.05

        reflect_input = self.BNNet(self.reflect_net_inputs)
        reflect_out = self.reflect_net(reflect_input)

        illum_en = illum_out**torch.sigmoid(self.gamma)
        image_en = reflect_out * illum_en
        image_out = reflect_out * illum_out

        return illum_out, illum_en, reflect_out, image_out, image_en

    '''def save_images(self, step, image_en):
        """Save intermediate results during training."""
        image_en_np = np.clip(torch_to_np(image_en.detach()), 0, 1)
        image_en_np = np_imresize(image_en_np, output_shape=self.original_image.shape[1:])
        save_image(f"{self.image_name}_enhanced_{step}", image_en_np, self.output_dir)'''
    def save_images(self, step, image_en):
        """Save intermediate results during training."""
        image_en_np = np.clip(torch_to_np(image_en.detach()), 0, 1)
        image_en_np = np_imresize(image_en_np, output_shape=self.original_image.shape[1:])
        save_image(f"{self.image_name}", image_en_np, self.output_dir)

def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of "Discrepant Untrained Network Priors"')
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU instead of GPU.')
    parser.add_argument('--input_path', default='images/input3.png', help='Path to the input image.')
    parser.add_argument('--output_dir', default='output', help='Path to save results.')
    parser.add_argument('--num_iter', type=int, default=5000, help='Number of iterations.')
    parser.add_argument('--E', type=float, default=0.6, help='Hyperparameter for loss term E.')
    parser.add_argument('--tv', type=float, default=0.003, help='Hyperparameter for TV loss.')
    parser.add_argument('--mse', type=float, default=12, help='Hyperparameter for MSE loss.')
    parser.add_argument('--bri', type=float, default=0.05, help='Hyperparameter for Brightness loss.')
    parser.add_argument('--ill', type=float, default=0.02, help='Hyperparameter for Illumination loss.')
    parser.add_argument('--adj', type=float, default=0.001, help='Hyperparameter for Adjustment loss.')
    parser.add_argument('--illum_threshold', type=bool, default=False, help='Enable illumination thresholding.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--show_every', type=int, default=5000, help='Frequency to display results.')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    engine = EngineModule(
        input_path=args.input_path,
        output_dir=args.output_dir,
        device=device,
        num_iter=args.num_iter,
        lr=args.lr,
        show_every=args.show_every,
        illum_threshold=args.illum_threshold
    )

    optimizer = torch.optim.Adam([
        {'params': engine.parameters, 'lr': engine.learning_rate},
    ])

    start_time = time.time()
    adjillum_loss = BrightLoss(args.E).to(device)
    tv_criterion = TVLoss_jit(engine.image.shape[1],engine.image.shape[2]).to(device)
    recon_criterion = FiedelityLoss().to(device)

    for step in range(1, args.num_iter + 1):
        optimizer.zero_grad()
        illum_out, illum_en, reflect_out, image_out, image_en = engine()

        # Loss computations
        recon_loss = F.mse_loss(engine.image_torch, image_out)
        bright_loss = F.mse_loss(illum_out, engine.fixed_illum)
        adjill_loss = adjillum_loss(image_en)
        tv_loss = tv_criterion(reflect_out)
        ill_smo_loss = tv_criterion(illum_out, reflect_out)
        total_loss = (
            args.mse * recon_loss +
            args.bri * bright_loss +
            args.ill * ill_smo_loss +
            args.adj * adjill_loss +
            args.tv * tv_loss
        )
        total_loss.backward(retain_graph=True)
        optimizer.step()

        if step % args.show_every == 0:
            print('Iteration: %05d    Loss: %.6f    BriLoss: %.6f    AdjLoss: %.6f' % (step, total_loss.item(), bright_loss.item(),adjill_loss.item()), '\r', end='')
            engine.save_images(step, image_en)

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
