from argparse import ArgumentParser


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser: ArgumentParser, fill_none=False):
        # load exsisting checkpoint
        self.loadmodel = ""
        self.resolution = 128
        # dataset path
        self.source_path = ""
        # path to save model and ckpt and tensorboard data
        self.model_path = ""
        # add relative gaussian noise when load exsisting ckpt
        self.noise_factor = 0.0

        # normalize the hmap or not
        self.norm_hmap = False
        # scale the normalized hmap
        self.norm_factor_hmap = 0.5

        self.intensity_scaler = 0.25
        super().__init__(parser, "Loading Parameters", fill_none)


class PipelineParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        # use a larger sampling rate in render
        # apply to both hmap and position query
        self.render_scale = 1

        # render spectrum or single_wavelenth
        # if single_wavelenth, need lambda
        self.render_type = "spectrum"
        self.lam = 0.5

        # use different C1
        # Kirchhoff: Kirchhoff model from Linqi Yan
        # Stam: Kirchhoff from Stam
        self.C1 = "Kirchhoff"

        # use different C3
        # paraxial: paraxial model from Linqi Yan
        # nonparaxial: nonparaxial from Stam
        self.C3 = "nonparaxial"

        self.sigma = 2.6
        self.blur_sigma = 0.52
        # noise to hmap
        self.noise = 0.0

        self.pixelsize = 1.0

        self.BRDF_name = "anti_mirror"
        # if it is image, define image path
        self.BRDF_image_path = "./data/batman.png"
        self.BRDF_image_scale = 1.5

        self.color_type = "white"
        # if it is disk_iri, define the angle range of the center color
        self.iri_range = 0.5

        self.material = "Silver"

        # apply gamma correction to GT and pred
        self.gt_gamma = 1.0
        self.pred_gamma = 1.0

        # fixed scaler or not
        self.fix_scaler = False
        
        self.distance=500000

        if not parser == None:
            super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # learning rate param for hmap
        self.hmap_freeze_step = 0
        self.hmap_lr_init = 0.001
        self.hmap_lr_final = 0.00001
        self.hmap_delay_step = 0
        self.hmap_delay_mult = 1.0
        self.hmap_max_steps = 200000

        # define angular range
        self.sample_range = 1.0
        # define importance sampling
        self.sample_exp = 2

        self.epoch_num = 200
        self.iteration_num = 2000000
        self.test_interval = 5
        self.save_interval = 10
        self.batch_size = 200

        if not parser == None:
            super().__init__(parser, "Optimization Parameters")
