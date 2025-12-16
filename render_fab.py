from scene.spectrum import *
from render.render import *
import cv2
from arguments import *
import matplotlib.pyplot as plt
import sys
from utils.general_utils import draw_heatmap

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def render_all(model_param, opti_param, pipe_param):
    """
    Rendering pipeline.

    Parameters:
    - model_param: parameters for heghtfield model.
    - opti_param: parameters for optimization.
    - pipe_param: parameters for rendering pipeline.

    """

    resolution = model_param.resolution
    output_path = (
        model_param.model_path
        + "/sigma="
        + str(pipe_param.blur_sigma)
        + "_noise="
        + str(pipe_param.noise)
        + "_range="
        + str(opti_param.sample_range)
        + "/"
    )
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # load model
    spectrum = Spectrum(pipe_param.material)
    hmap = Heightfield(torch.zeros(resolution, resolution), 1, 1)
    hmap.load_model(model_param.loadmodel)

    # set up rendering parameters
    brdf_resolution = 64
    scale = 8
    hmap.padding = True
    hmap.H = 32 * 3
    hmap.W = 32 * 3
    hmap.update_hmap(
        scale,
        hmap.H * scale,
        hmap.W * scale,
        norm=hmap.norm,
        norm_scale=hmap.norm_scale,
    )
    hmap.H *= scale
    hmap.W *= scale
    hmap.texelWidth = hmap.texelWidth / scale
    distance = 500000

    # visualize hmap
    img = np.asarray(hmap.heightfieldImage.cpu().data)[
        : resolution * scale, : resolution * scale
    ]
    cv2.imwrite(output_path + "/hmap.exr", img)
    np.savetxt(output_path + "/hmap.txt", img, delimiter=",", fmt="%.08f")
    cv2.imwrite(output_path + "/hmap.png", draw_heatmap(img))

    # render
    spectrumSamples = torch.zeros(brdf_resolution * brdf_resolution, spectrum.SPECTRUM_SAMPLES)
    with torch.no_grad():
        for i in range(brdf_resolution):
            for j in range(brdf_resolution):
                sample_range = opti_param.sample_range
                omega_o = (
                    np.asarray(
                        [
                            (i + 0.5) / brdf_resolution * 2.0 - 1.0,
                            (j + 0.5) / brdf_resolution * 2.0 - 1.0,
                            0,
                        ]
                    )
                    * sample_range
                )
                if omega_o[0] * omega_o[0] + omega_o[1] * omega_o[1] > 1:
                    continue
                omega_o[2] = np.sqrt(1 - omega_o[0] * omega_o[0] - omega_o[1] * omega_o[1])
                omega_i = torch.tensor([0, 0, 1], device="cuda", dtype=torch.float32)
                omega_o = torch.tensor(omega_o, device=device, dtype=torch.float32)

                omega_i = F.normalize(omega_i, eps=1e-8, dim=-1)
                omega_o = F.normalize(omega_o, eps=1e-8, dim=-1)

                for k in range(spectrum.SPECTRUM_SAMPLES):
                    lam = (k + 0.5) / spectrum.SPECTRUM_SAMPLES * (0.68 - 0.42) + 0.42

                    img = render_cuda(
                        torch.tensor([0, 0, 1], device="cuda", dtype=torch.float32) * distance,
                        torch.tensor(omega_o, device="cuda", dtype=torch.float32) * distance,
                        pipe_param.pixelsize / scale,
                        resolution * scale,
                        resolution * scale,
                        pipe_param.sigma / 0.5 * lam,
                        hmap,
                        lam,
                        hmap.G_Prime(device, istrain=False),
                        device,
                        pipe_param=pipe_param,
                    )

                    spectrumSamples[i * brdf_resolution + j][k] = img.mean()

    # # visualize specturm intensity with normalization
    # for i in range(spectrum.SPECTRUM_SAMPLES):
    #     lam=(i + 0.5) / spectrum.SPECTRUM_SAMPLES * (0.68 - 0.42) + 0.42
    #     pred=spectrumSamples[:,i].reshape(brdf_resolution,brdf_resolution)
    #     cv2.imwrite(output_path+"scale="+str(scale)+"_sigma_g_="+str(lam)+".png",np.asarray(255*pred/pred.max()))

    # visualize rendering result
    r, g, b = spectrum.SpectrumToRGB(spectrumSamples)
    img = torch.stack([b, g, r], dim=-1).reshape(brdf_resolution, brdf_resolution, 3).cpu().data
    cv2.imwrite(output_path + "scale=" + str(scale) + "_render.png", np.asarray(255 * img / img.max()))
    cv2.imwrite(output_path + "scale=" + str(scale) + "_render.exr", np.asarray(img))
    cv2.imwrite(output_path + "scale=" + str(scale) + "_render_gamma.exr", np.power(np.asarray(img), 1 / 2.2))

    # visualize specturm intensity with color
    spectrumSamples = spectrumSamples / spectrumSamples.max() * 2
    for k in range(spectrum.SPECTRUM_SAMPLES):
        lam = (k + 0.5) / spectrum.SPECTRUM_SAMPLES * (0.68 - 0.42) + 0.42
        spectrumSamples_single = torch.zeros(brdf_resolution * brdf_resolution, spectrum.SPECTRUM_SAMPLES)
        spectrumSamples_single[:, k] = spectrumSamples[:, k]
        r, g, b = spectrum.SpectrumToRGB(spectrumSamples_single)
        img = torch.stack([b, g, r], dim=-1).reshape(brdf_resolution, brdf_resolution, 3).cpu().data
        cv2.imwrite(output_path + "sclae="+ str(scale) + "_sigma_g_"+ str(lam) + "_single.png", np.asarray(255 * img))


parser = ArgumentParser(description="Rendering scripts parameters")
mp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)

args = parser.parse_args(sys.argv[1:])
model_param = mp.extract(args)
opti_param = op.extract(args)
pipe_param = pp.extract(args)
render_all(model_param, opti_param, pipe_param)
