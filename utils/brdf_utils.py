from scene.fabricating_BRDF import *


def color_lobe(color_type, reflectance, omega_i, omega_o, iri_range=0.5):
    """
    Control color of the BRDF

    Parameters:
    - color_type: name of the color type.
    - reflectance: current grayscale reflectance (intensity only).
    - omega_i: incident angle.
    - omega_o: outgoing angle.
    - iri_range: if disk-like iridescent color, what's the angle range of the disk.

    Returns:
    - Numpy array or tensor of shape (3,): reflectance (intensity and color)

    """

    if color_type == "red":
        lobe = [0, 0, 1]
    elif color_type == "blue":
        lobe = [1, 0, 0]
    elif color_type == "white":
        lobe = [1, 1, 1]
    elif color_type == "orange":
        lobe = [0, 0.5, 1]
    elif color_type == "yellow":
        lobe = [0, 1, 1]
    elif color_type == "green":
        lobe = [0, 1, 0]
    elif color_type == "cyan":
        lobe = [1, 1, 0]
    elif color_type == "purple":
        lobe = [1, 0, 0.5]

    # iridescent color
    elif color_type == "iri":
        a = omega_i[:2] + omega_o[:2]
        if a[1] > 0:
            lobe = [1, 0, 0]
        else:
            lobe = [0, 0, 1]
    elif color_type == "disk_iri":
        a = omega_i[:2] + omega_o[:2]
        a_abs = a[0] * a[0] + a[1] * a[1]
        if a_abs > iri_range * iri_range:
            lobe = [1, 0, 0]
        else:
            lobe = [0, 0, 1]
    elif color_type == "inv_disk_iri":
        a = omega_i[:2] + omega_o[:2]
        a_abs = a[0] * a[0] + a[1] * a[1]
        if a_abs > iri_range * iri_range:
            lobe = [0, 0, 1]
        else:
            lobe = [1, 0, 0]
    else:
        lobe = color_type

    reflectance[0] *= lobe[0]
    reflectance[1] *= lobe[1]
    reflectance[2] *= lobe[2]
    return reflectance


def brdf_function(BRDF_name, wi, wo, brdf_img=None,brdf_img_scale=1.5):
    """
    Calculate intensity of the BRDF

    Parameters:
    - BRDF_name: name of the BRDF type.
    - wi: incident angle.
    - wo: outgoing angle.
    - brdf_img: required if the BRDF is controled by a 2D image.

    Returns:
    - float: reflectance (intensity only)

    """

    if BRDF_name == "anti_mirror":
        return anti_mirror_brdf(wi, wo, np.asarray([0, 0, 1]), 0.05, 0.12)
    if BRDF_name == "aniso_anti_mirror":
        return aniso_anti_mirror_brdf(wi, wo, np.asarray([0, 0, 1]), 0.7, 0.015, 0.06) / 1.8
    if BRDF_name == "diffuse":
        return mirror_brdf(wi, wo, np.asarray([0, 0, 1]), 0.15)
    if BRDF_name == "uniform":
        return 1

    if BRDF_name == "image_scale":
        return BRDF_from_image_scale(wi, wo, np.asarray([0, 0, 1]), brdf_img, brdf_img_scale)
    if BRDF_name == "image_inverse":
        return BRDF_from_image_inverse(wi, wo, np.asarray([0, 0, 1]), brdf_img, 0.20)
