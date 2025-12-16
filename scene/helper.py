import torch
import math


# x1/x2/y1/y2: (N,)
def dist(x1, y1, x2, y2):
    return torch.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


# 1D Gaussian function
# x: (N,M,)  mu:(N,)  sigma:(N,)
# return: (N,M,1)
def G_1D(x, mu, sigma):
    N = mu.shape[0]
    M = x.shape[1]
    x = x.reshape(N, M, 1)
    mu = mu.reshape(N, 1, 1).expand(N, M, 1)
    sigma = sigma.reshape(N, 1, 1).expand(N, M, 1)
    return (
        1.0
        / (math.sqrt(2.0 * math.pi) * sigma)
        * torch.exp(-0.5 * torch.pow((x - mu) / sigma, 2.0))
    )


# 2D Gaussian function
# x: (N,M,2)  mu: (N,2)   sigma: (N,1)
# return: (N,M,1)
def G(x, mu, sigma):
    return G_1D(x[:, :, 0], mu[:, 0], sigma) * G_1D(x[:, :, 1], mu[:, 1], sigma)


# e^(ix) = cos x + i sin x
# x: (N,)
def cis(x):
    return torch.cat([torch.cos(x).reshape(-1, 1), torch.sin(x).reshape(-1, 1)], dim=1)


# e^(-ix) = cos x - i sin x
# x: (N,)
def cnis(x):
    return torch.cat([torch.cos(x).reshape(-1, 1), -torch.sin(x).reshape(-1, 1)], dim=1)


# camplex multiply
# x: (,,2) y: (,,2)
def complex_mul(x, y):
    shape = x.shape
    x = x.reshape(-1, 2)
    y = y.reshape(-1, 2)
    real = x[:, 0] * y[:, 0] - x[:, 1] * y[:, 1]
    virtual = x[:, 1] * y[:, 0] + x[:, 0] * y[:, 1]
    return torch.cat([real.reshape(-1, 1), virtual.reshape(-1, 1)], dim=1).reshape(shape)
