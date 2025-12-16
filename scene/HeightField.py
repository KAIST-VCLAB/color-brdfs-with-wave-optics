from scene.Gabor import *
import os
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import torch.nn.functional as F
from utils.general_utils import get_expon_lr_func


class Heightfield:
    def __init__(
        self,
        heightfieldImage,
        texelWidth,
        vertScale,
        padding=True,
        norm=False,
        norm_scale=0.5,
    ):
        self.trainhmap = heightfieldImage
        self.heightfieldImage = heightfieldImage
        self.texelWidth = texelWidth
        self.vertScale = vertScale
        if padding:
            self.H = heightfieldImage.shape[0] * 3
            self.W = heightfieldImage.shape[1] * 3
        else:
            self.H = heightfieldImage.shape[0]
            self.W = heightfieldImage.shape[1]
        self.padding = padding
        self.scale = 1
        self.norm = norm
        self.norm_scale = norm_scale
        self.smooth = False
        self.gs_sigma = 0.13
        self.noise_factor = 0.0

    def update_hmap(self, scale, H, W, norm=False, norm_scale=0.5):
        """
        Add normalization and noise to updated heightmap

        Parameters:
        - scale: upsample the heightmap.
        - H: height of the heightmap image.
        - W: width of the heightmap image.
        - norm: do normalization or not.
        - norm_scale: height range after normalization

        """

        # add noise
        train_map_noise = self.trainhmap + \
                        self.noise_factor * torch.randn(self.trainhmap.shape).cuda() * self.trainhmap
        
        # padding
        if self.padding:
            grids = torch.cat([train_map_noise, train_map_noise, train_map_noise], 0)
            hmap_grid = torch.cat([grids, grids, grids], 1)
            scaledmap = hmap_grid.reshape(1, 1, self.H, self.W)
            upsample = nn.Upsample(scale_factor=scale, mode="nearest")
            scaledmap = upsample(scaledmap).reshape(H, W)
        else:
            upsample = nn.Upsample(scale_factor=scale, mode="nearest")
            scaledmap = upsample(train_map_noise.reshape(1, 1, self.H, self.W)).reshape(H, W)

        # normalization
        if norm:
            scaler = norm_scale / (scaledmap.max() - scaledmap.min())
            self.heightfieldImage = scaledmap * scaler
        else:
            self.heightfieldImage = scaledmap

        # smooth (gaussian blur)
        if self.smooth:
            if scale >= 16:
                size = 17 * (scale // 4)
            else:
                size = 17
            ax = torch.linspace(-(size // 2), size // 2, size) * self.texelWidth / scale
            xx, yy = torch.meshgrid(ax, ax)
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.gs_sigma**2))

            kernel = kernel / kernel.sum()
            kernel = kernel.to(device=self.heightfieldImage.device)

            img = self.heightfieldImage.unsqueeze(0).unsqueeze(0)
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            img = F.conv2d(img, kernel, padding=kernel.shape[-1] // 2)

            self.heightfieldImage = img.squeeze(0).squeeze(0)

    def G_Prime(self, device, istrain=True, scale=1):
        """
        Calculate G_prime of the whole height map in parallel.

        Parameters:
        - device: cpu or gpu.
        - istrain: self.heightfieldImage need to be updated in a training iter.
        - scale: upsample the heightmap.

        Returns:
        - GaborKernelPrime class: wavelength independent Gabor info

        """

        H = self.H * scale
        W = self.W * scale
        texelWidth = self.texelWidth / scale

        # self.heightfieldImage need to be updated in a training iter.
        if istrain:
            self.update_hmap(scale=scale, H=H, W=W, norm=self.norm, norm_scale=self.norm_scale)

        # calculate Gabor_prime parameter
        X, Y = torch.meshgrid(torch.arange(H, device=device) + 0.5, torch.arange(W, device=device) + 0.5)
        mu_k = torch.stack([X, Y], dim=-1).reshape(H * W, 2) * texelWidth
        l_k = texelWidth
        sigma_k = torch.ones((H * W, 1), dtype=torch.float32, device=device) * l_k / 2

        H_mk = (self.heightfieldImage * self.vertScale).reshape(H * W, 1)
        L_Hmap = F.pad(self.heightfieldImage, pad=(1, 1, 1, 1), mode="constant", value=0)
        HPrime_mk = (
            torch.stack(
                [
                    (L_Hmap[2:, 1:-1] - L_Hmap[0:-2, 1:-1]) * self.vertScale,
                    (L_Hmap[1:-1, 2:] - L_Hmap[1:-1, 0:-2]) * self.vertScale,
                ],dim=-1,).reshape(H * W, 2) / 2 / texelWidth
        )

        C_info = H_mk - (HPrime_mk * mu_k).sum(dim=-1, keepdim=True)
        a_info = 2.0 * HPrime_mk

        return GaborKernelPrime(mu_k, sigma_k, a_info, C_info)

    # set up training parameters
    def training_setup(self, training_args, device):
        heightfieldImage = np.asarray(self.trainhmap.cpu().data)
        self.trainhmap = nn.Parameter(
            torch.tensor(heightfieldImage, dtype=torch.float32, device=device)
        ).requires_grad_(True)
        l = [
            {
                "params": [self.trainhmap],
                "lr": training_args.hmap_lr_init,
                "name": "hmap",
            },
        ]
        self.optimizer = torch.optim.Adam(l, eps=1e-15)
        self.hmap_scheduler = get_expon_lr_func(
            lr_freeze=training_args.hmap_freeze_step,
            lr_init=training_args.hmap_lr_init,
            lr_final=training_args.hmap_lr_final,
            lr_delay_steps=training_args.hmap_delay_step,
            lr_delay_mult=training_args.hmap_delay_mult,
            max_steps=training_args.hmap_max_steps,
        )

    # update learning rate according to step (iter)
    def update_lr(self, step):
        hmap_lr = self.hmap_scheduler(step)
        self.optimizer.param_groups[0]["lr"] = hmap_lr

    def capture(self):
        return (
            self.trainhmap,
            self.heightfieldImage,
            self.texelWidth,
            self.vertScale,
            self.H,
            self.W,
            self.scale,
            self.norm,
            self.norm_scale,
            self.smooth,
            self.gs_sigma,
        )

    def save_model(self, path):
        torch.save(self.capture(), path)

    def load_model(self, path):
        ckpt = torch.load(path)
        (
            self.trainhmap,
            self.heightfieldImage,
            self.texelWidth,
            self.vertScale,
            self.H,
            self.W,
            self.scale,
            self.norm,
            self.norm_scale,
            self.smooth,
            self.gs_sigma,
        ) = ckpt
