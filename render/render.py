from scene.HeightField import *
from render.GaborRender_cuda import GaborRenderFunction

def render_cuda(
    cam_pos,
    pl_pos,
    pixel_size,
    imgH,
    imgW,
    sigma_p,
    hmap,
    lam,
    G_Prime,
    device="cpu",
    pipe_param=None,
):
    """
    Render wave optics based reflectance under single wavelength.

    Parameters:
    - cam_pos: tensor of shape (3,), camera position.
    - pl_pos: tensor of shape (3,), point light source position..
    - pixel_size: pixel size of the height map.
    - imgH: height of the height map image.
    - imgW: width of the height map image.
    - sigma_p: size of the coherence area.
    - hmap: Heightfield class.
    - lam: wavelength.
    - G_Prime: Gabor model.
    - device: cpu or gpu.
    - pipe_param: parameters for rendering pipeline.

    Returns:
    - Tensor of shape (1,): single-wavelength reflectance rendered

    """

    # generate query location
    scale = pipe_param.render_scale
    if hmap.padding:
        padding = imgH * pixel_size
    else:
        padding = 0
    X, Y = torch.meshgrid(
        torch.arange(imgH * scale, device=device),
        torch.arange(imgW * scale, device=device),
    )
    query = (torch.stack([X, Y], dim=-1) + 0.5) * pixel_size / scale + padding
    query = torch.cat([query, torch.zeros((imgH * scale, imgW * scale, 1), device=device)], dim=-1)

    # generate query angle
    omega_i = F.normalize(pl_pos - query, eps=1e-8, dim=-1)[:, :]
    omega_o = F.normalize(cam_pos - query, eps=1e-8, dim=-1)[:, :]
    a = omega_i + omega_o
    az = a[:, :, 2].mean()
    a = a / lam
    a = a[:, :, :2].reshape(imgH * imgW * scale * scale, 2).contiguous()

    # generate new gabor kernel under temporal wave length
    if pipe_param.C3 == "nonparaxial":
        g = G_Prime.toGaborKernel(lam, C3=az)
    else:
        g = G_Prime.toGaborKernel(lam)

    # render
    brdfValue = GaborRenderFunction.apply(
        query[:, :, :2].reshape(-1, 2).contiguous(),
        a,
        g.mu,
        g.sigma,
        g.a,
        g.C,
        sigma_p,
        imgH * pixel_size + 2 * padding,
        imgW * pixel_size + 2 * padding,
    )

    # calculate ceofficient
    if pipe_param.C1 == "Kirchhoff":
        oDotN = omega_o[:, :, 2]
        iDotN = omega_i[:, :, 2]
        C1 = (iDotN + oDotN) * (iDotN + oDotN) / (lam * lam * 4.0 * iDotN * oDotN)
        C1 = C1.reshape(imgH * scale * imgW * scale)
    else:
        oDotN = omega_o[:, :, 2]
        iDotN = omega_i[:, :, 2]
        C1 = (1 + (omega_o * omega_i).sum(dim=-1)) / (lam * lam * iDotN * oDotN * (iDotN + oDotN))
        C1 = C1.reshape(imgH * scale * imgW * scale)

    # return the energy
    energy = C1 * ((brdfValue * brdfValue).sum(dim=-1))
    return energy


def render(
    cam_pos,
    pl_pos,
    pixelsize,
    imgH,
    imgW,
    sigma_p,
    hmap,
    spectrum,
    device="cpu",
    pipe_param=None,
    datatype="BGR",
):
    """
    Render wave optics based reflectance.

    Parameters:
    - cam_pos: tensor of shape (3,), camera position.
    - pl_pos: tensor of shape (3,), point light source position..
    - pixelsize: pixel size of the height map.
    - imgH: height of the height map image.
    - imgW: width of the height map image.
    - sigma_p: size of the coherence area.
    - hmap: Heightfield class.
    - spectrum: Spectrum class.
    - device: cpu or gpu.
    - pipe_param: parameters for rendering pipeline.
    - datatype: return type, BGR or XYZ or spectrum.

    Returns:
    - Tensor of shape (3,) or (spectrumSamples, ): reflectance rendered

    """

    scale = pipe_param.render_scale
    spectrumSamples = torch.zeros(imgH * imgW * scale * scale, spectrum.SPECTRUM_SAMPLES, device=device)

    # initialize G_Prime
    G_Prime = hmap.G_Prime(device, scale=scale)

    for k in range(spectrum.SPECTRUM_SAMPLES):
        lam = (k + 0.5) / spectrum.SPECTRUM_SAMPLES * (0.68 - 0.42) + 0.42
        brdfValue = render_cuda(
            cam_pos,
            pl_pos,
            pixelsize,
            imgH,
            imgW,
            sigma_p / 0.5 * lam,
            hmap,
            lam,
            G_Prime,
            device,
            pipe_param,
        )
        spectrumSamples[:, k] = brdfValue

    if datatype == "BGR":
        # calculate rgb color according to spectrum
        r, g, b = spectrum.SpectrumToRGB(spectrumSamples)
        # opencv use bgr channel
        return torch.stack([b, g, r], dim=-1).reshape(imgH * scale, imgW * scale, 3)
    if datatype == "XYZ":
        x, y, z = spectrum.SpectrumToXYZ(spectrumSamples)
        return torch.stack([x, y, z], dim=-1).reshape(imgH, imgW, 3)
    if datatype == "spectrum":
        return spectrumSamples.reshape(imgH * scale, imgW * scale, spectrum.SPECTRUM_SAMPLES)
