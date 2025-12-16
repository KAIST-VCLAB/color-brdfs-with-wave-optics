import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from scene.helper import *


class GaborKernel:

    # mu: (N,2) sigma: (N, ) a: (N,2)  C: (N,2) two channel complex float
    def __init__(self, mu, sigma, a, C):
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.C = C

    # s: (N,M,2)
    def eval(self, mu, sigma, a, C, s):
        N = mu.shape[0]
        M = s.shape[1]
        aDots = torch.einsum("NX,NMX->NMX", a, s).sum(dim=-1)
        exp_complex = cnis((2.0 * math.pi * aDots)).reshape(N, M, 2)
        complex_term = complex_mul(exp_complex, C.reshape(N, 1, 2).expand(N, M, 2))
        return complex_term * G(s, mu, sigma)

    def change_device(self, device):
        self.mu = self.mu.to(device=device)
        self.sigma = self.sigma.to(device=device)
        self.a = self.a.to(device=device)
        self.C = self.C.to(device=device)

    # evaluate FT(G)
    # u: (N,M,2)
    def xform(self, u):
        sigmaPrime = 1.0 / (2.0 * math.pi * self.sigma)
        CPrime = complex_mul(
            self.C,
            (1.0 / (2.0 * math.pi * self.sigma * self.sigma))
            * cnis(2.0 * math.pi * (self.a * self.mu).sum(dim=-1)),
        )
        return self.eval(-self.a, sigmaPrime, self.mu, CPrime, u)


# a and C of Gabor kernel changes with different lambda
# GaborKernelPrime stores the information needed to calculate a and C before lambda is given
# (to avoid redundant computation)
class GaborKernelPrime:

    # mu: (N,2) sigma: (N, ) a: (N,2)  C: (N,1) two channel complex float
    def __init__(self, mu, sigma, aInfo, cInfo):
        self.mu = mu
        self.sigma = sigma
        self.aInfo = aInfo
        self.cInfo = cInfo

    def change_device(self, device):
        self.mu = self.mu.to(device=device)
        self.sigma = self.sigma.to(device=device)
        self.aInfo = self.aInfo.to(device=device)
        self.cInfo = self.cInfo.to(device=device)

    # save the model as a tensor catted by G_Prime parameter
    def save_model(self, path):
        ckpt = torch.cat([self.mu, self.sigma, self.aInfo, self.cInfo], dim=-1)
        torch.save(ckpt, path)

    # load model from tensor catted by G_Prime parameter
    def load_model(self, path):
        ckpt = torch.load(path)
        self.mu = ckpt[:, 0:2]
        self.sigma = ckpt[:, 2:3]
        self.aInfo = ckpt[:, 3:5]
        self.cInfo = ckpt[:, 5:6]

    # lambda: (N,)
    # convert to Gabor kernel
    def toGaborKernel(self, lam, C3=2):
        l = self.sigma * 2
        C = l * l * cnis(C3 * 2.0 * math.pi / lam * self.cInfo)
        a = self.aInfo / lam / 2 * C3

        return GaborKernel(self.mu, self.sigma, a, C)
