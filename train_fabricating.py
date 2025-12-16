from render.render import *
from tqdm import tqdm
from scene.spectrum import *
from torch.utils.tensorboard import SummaryWriter
from arguments import *
import time
import cv2
import sys
import matplotlib.pyplot as plt
from utils.general_utils import L2_loss, draw_heatmap
from utils.brdf_utils import brdf_function, color_lobe

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# generate tensorboard training report
def training_report(
    tb_writer, iteration, loss, elapsed, dotest, hmap, spectrum, pipe_param, model_param
):
    """
    Generate tensorboard training report

    Parameters:
    - tb_writer: tensorboard writer.
    - iteration: int representing the training iteration.
    - loss: float representing loss of this iteration.
    - elapsed: float representing the elapsed time of this iteration.
    - dotest: bool controls whether this is a test report.
                In a test report, render the 2D BRDF slice, calculate
                the test loss, and visualize the heightmap and rendered
                BRDF.
    - hmap: Heightfield class.
    - spectrum: Spectrum class.
    - pipe_param: parameters for rendering pipeline.
    - model_param: parameters for heghtfield model.

    """

    resolution = model_param.resolution
    # read 2D BRDF slice image is needed
    if pipe_param.BRDF_name == "image_scale" or pipe_param.BRDF_name == "image_inverse":
        brdf_img = cv2.imread(pipe_param.BRDF_image_path)
        brdf_img = cv2.resize(brdf_img, (512, 512)) / 255
    else:
        brdf_img = None

    # write relevant parameters to tensorboard
    tb_writer.add_scalar("loss", loss, iteration)
    tb_writer.add_scalar("iteration_time", elapsed, iteration)
    tb_writer.add_scalar("lr", hmap.optimizer.param_groups[0]["lr"], iteration)
    g = hmap.G_Prime(device).toGaborKernel(0.5)
    tb_writer.add_scalar("param/scale", hmap.scale, iteration)
    tb_writer.add_scalar("param/a", g.a.abs().mean().cpu().data, iteration)
    tb_writer.add_scalar("param/mu", g.mu.abs().mean().cpu().data, iteration)
    tb_writer.add_scalar("param/sigma", g.sigma.abs().mean().cpu().data, iteration)
    tb_writer.add_scalar("param/coeff", g.C.abs().mean().cpu().data, iteration)

    # in a test iteration, render BRDF image, write the image and loss to tensorboard
    if dotest:
        distance = 500000

        # initialize gt (target) and pred (rendered) for BRDF image
        gt = np.zeros((64, 64, 3))
        pred = np.zeros((64, 64, 3))

        # (angular) resolution of the tested 2D BRDF slice image is 64
        for i in range(64):
            for j in range(64):

                # angular range of the tested 2D BRDF slice image is [-0.5, 0.5] ([-30, 30] degree)
                sample_range = 0.5
                omega_o = (
                    np.asarray(
                        [
                            (i + 0.5 + 32) / 128 * 2.0 - 1.0,
                            (j + 0.5 + 32) / 128 * 2.0 - 1.0,
                            0,
                        ]
                    ) * sample_range
                )
                if omega_o[0] * omega_o[0] + omega_o[1] * omega_o[1] > 1:
                    continue
                omega_o[2] = np.sqrt(1 - omega_o[0] * omega_o[0] - omega_o[1] * omega_o[1])

                # calculate gt (target)
                # TODO: check whether we still need to ifelse on ret.shape
                ret = brdf_function(pipe_param.BRDF_name, np.asarray([0, 0, 1]), np.asarray(omega_o), brdf_img, pipe_param.BRDF_image_scale)
                ret = np.asarray(ret)
                if ret.shape == 1:
                    gt[i][j][0] = ret[0]
                    gt[i][j][1] = ret[0]
                    gt[i][j][2] = ret[0]
                else:
                    gt[i][j] = ret
                gt[i][j] = color_lobe(pipe_param.color_type, gt[i][j], 
                                      np.asarray([0, 0, 1]), np.asarray(omega_o), pipe_param.iri_range,)

                # calculate pred (rendered)
                rendered = hmap.scale * render(
                    torch.tensor([0, 0, 1], device="cuda", dtype=torch.float32) * distance,
                    torch.tensor(omega_o, device="cuda", dtype=torch.float32) * distance,
                    pipe_param.pixelsize,
                    resolution,
                    resolution,
                    pipe_param.sigma,
                    hmap,
                    spectrum,
                    device,
                    pipe_param=pipe_param,
                )
                pred[i][j] = rendered.mean(dim=0).mean(dim=0).cpu().data

        # clip minus value
        pred[pred < 0] = 0
        # gamma correction
        pred = np.power(pred, 1 / pipe_param.pred_gamma)
        gt = np.power(gt, 1 / pipe_param.gt_gamma)

        loss = L2_loss(pred, gt)

        # write height map to tensorboard
        img = np.asarray(hmap.trainhmap.data.cpu())
        img = (img - img.min()) / (img.max() - img.min())
        tb_writer.add_image("test/hmap", draw_heatmap(img), dataformats="HWC", global_step=iteration)

        # write gt and pred to tensorboard
        pred = pred[..., ::-1]
        gt = gt[..., ::-1]
        tb_writer.add_images(
            "test/render",
            pred.reshape(1, 64, 64, 3).transpose(0, 3, 1, 2).clip(0, 1),
            global_step=iteration,
        )
        tb_writer.add_images(
            "test/ground_truth",
            gt.reshape(1, 64, 64, 3).transpose(0, 3, 1, 2).clip(0, 1),
            global_step=iteration,
        )

        tb_writer.add_scalar("test/l2_loss", loss, iteration)
        print("test loss:", loss, "at iter", iteration)


def train(model_param, opti_param, pipe_param):
    """
    Training pipeline.

    Parameters:
    - model_param: parameters for heghtfield model.
    - opti_param: parameters for optimization.
    - pipe_param: parameters for rendering pipeline.

    """

    output_path = model_param.model_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_path + "tensorboard"):
        os.mkdir(output_path + "tensorboard")
    if not os.path.exists(output_path + "model"):
        os.mkdir(output_path + "model")

    resolution = model_param.resolution
    iternum = opti_param.iteration_num
    test_interval = opti_param.test_interval
    save_interval = opti_param.save_interval
    distance = pipe_param.distance

    # read 2D BRDF slice image is needed
    if pipe_param.BRDF_name == "image_scale" or pipe_param.BRDF_name == "image_inverse":
        brdf_img = cv2.imread(pipe_param.BRDF_image_path)
        brdf_img = cv2.resize(brdf_img, (512, 512)) / 255
    else:
        brdf_img = None

    # initialize spectrum
    spectrum = Spectrum(pipe_param.material)
    spectrum.np2cuda(device=device)

    # initialize heightmap model 1. from exsiting ckpt 2. randomly
    if not model_param.loadmodel == "":
        hmap = Heightfield(torch.zeros(resolution, resolution), 1, 1)
        hmap.load_model(model_param.loadmodel)
    else:
        hmap_img = 0.25 * (torch.rand((resolution, resolution), device=device) - 0.5)
        hmap = Heightfield(hmap_img, pipe_param.pixelsize, 1)

    hmap.training_setup(opti_param, device)
    hmap.norm = model_param.norm_hmap
    hmap.norm_scale = model_param.norm_factor_hmap
    hmap.scale = model_param.intensity_scaler
    hmap.smooth = True
    hmap.gs_sigma = pipe_param.blur_sigma
    if hmap.gs_sigma == 0.0:
        hmap.smooth = False
    hmap.noise_factor = pipe_param.noise

    # initialize tensorboar writer
    tb_writer = SummaryWriter(output_path + "tensorboard")
    progress_bar = tqdm(range(0, iternum), desc="Training progress")

    # record BRDF image before the training start
    with torch.no_grad():
        training_report(tb_writer, 0, 0, 0, 1, hmap, spectrum, pipe_param, model_param)

    # start training
    iter = 0
    while iter < iternum:
        start = time.time()

        # random sample with importance sampling
        sample = np.random.rand(4)
        sample_range = opti_param.sample_range
        omega_o = np.asarray([sample[0] - 0.5, sample[1] - 0.5, 0]) * 2
        omega_i = np.asarray([sample[2] - 0.5, sample[3] - 0.5, 0]) * 2
        sign_o = np.sign(omega_o)
        sign_i = np.sign(omega_i)
        omega_o = (
            np.power(np.abs(omega_o), opti_param.sample_exp) / 2 * sample_range * sign_o
        )
        omega_i = (
            np.power(np.abs(omega_i), opti_param.sample_exp) / 2 * sample_range * sign_i
        )
        if omega_o[0] * omega_o[0] + omega_o[1] * omega_o[1] > 1:
            continue
        if omega_i[0] * omega_i[0] + omega_i[1] * omega_i[1] > 1:
            continue
        omega_o[2] = np.sqrt(1 - omega_o[0] * omega_o[0] - omega_o[1] * omega_o[1])
        omega_i[2] = np.sqrt(1 - omega_i[0] * omega_i[0] - omega_i[1] * omega_i[1])

        # calculate gt (target)
        gt = brdf_function(pipe_param.BRDF_name, np.asarray(omega_i), np.asarray(omega_o), brdf_img, pipe_param.BRDF_image_scale)
        gt = torch.tensor(gt, device="cuda", dtype=torch.float32)
        if gt.numel() == 1:
            gt = gt.expand(3).contiguous()
        gt = color_lobe(pipe_param.color_type, gt,
                        np.asarray(omega_i), np.asarray(omega_o), pipe_param.iri_range)
        gt = torch.pow(gt, 1 / pipe_param.gt_gamma)

        # calculat pred (rendered)
        rendered = hmap.scale * render(
            torch.tensor(omega_i, device="cuda", dtype=torch.float32) * distance,
            torch.tensor(omega_o, device="cuda", dtype=torch.float32) * distance,
            pipe_param.pixelsize,
            resolution,
            resolution,
            pipe_param.sigma,
            hmap,
            spectrum,
            device,
            pipe_param=pipe_param,
        )
        if rendered.max() > 1e8 or torch.isnan(rendered).any():
            continue
        pred_brdf = rendered.mean(dim=0).mean(dim=0)

        # gamma correction
        if not pipe_param.pred_gamma == 1.0:
            pred_brdf[pred_brdf < 0] = 0
            pred_brdf = torch.pow(pred_brdf, 1 / pipe_param.pred_gamma)

        loss = L2_loss(pred_brdf, gt)
        loss.backward()
        end = time.time()

        with torch.no_grad():
            # pixel value that is too small will possibly cause NaN in grad, prune it
            for group in hmap.optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad[torch.isnan(param.grad)] = 0

            # update lr according to iteration number
            hmap.update_lr(iter)
            hmap.optimizer.step()
            hmap.optimizer.zero_grad(set_to_none=True)

            if iter % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)

            elapsed_time = end - start
            dotest = iter % test_interval == 0
            training_report(
                tb_writer,
                iter,
                loss.item(),
                elapsed_time,
                dotest,
                hmap,
                spectrum,
                pipe_param,
                model_param,
            )

        if iter % save_interval == 0:
            hmap.save_model(output_path + "model/ckpt" + str(iter) + ".tsr")

        if iter == iternum:
            progress_bar.close()

        iter = iter + 1


parser = ArgumentParser(description="Training scripts parameters")
mp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)

args = parser.parse_args(sys.argv[1:])
model_param = mp.extract(args)
opti_param = op.extract(args)
pipe_param = pp.extract(args)
train(model_param, opti_param, pipe_param)
