import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def mirror_brdf(incident_vector, outgoing_vector, normal, spread_outer=0.22):
    """
    Compute the anti-mirror BRDF value.

    Parameters:
    - incident_vector: numpy array of shape (3,) representing the incident light direction.
    - outgoing_vector: numpy array of shape (3,) representing the outgoing view direction.
    - normal: numpy array of shape (3,) representing the surface normal at the intersection point.
    - spread: float, controls the spread of the reflectance away from the mirror direction.

    Returns:
    - float: The BRDF value for the given directions and surface properties.
    """

    # Normalize vectors
    i = normalize(incident_vector)
    o = normalize(outgoing_vector)
    n = normalize(normal)

    # Compute the mirror reflection direction
    mirror_direction = 2 * np.dot(n, i) * n - i
    mirror_direction = normalize(mirror_direction)

    # Compute the angle to the mirror direction
    dot_product = np.dot(o, mirror_direction)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Invert reflectance: more intensity away from the mirror direction
    mirror_value = np.exp(-(angle**2) / (2 * spread_outer**2))

    return mirror_value


def anti_mirror_brdf(
    incident_vector, outgoing_vector, normal, spread=1.0, spread_outer=0.22
):
    """
    Compute the anti-mirror BRDF value.

    Parameters:
    - incident_vector: numpy array of shape (3,) representing the incident light direction.
    - outgoing_vector: numpy array of shape (3,) representing the outgoing view direction.
    - normal: numpy array of shape (3,) representing the surface normal at the intersection point.
    - spread: float, controls the spread of the reflectance away from the mirror direction.

    Returns:
    - float: The BRDF value for the given directions and surface properties.
    """

    # Normalize vectors
    i = normalize(incident_vector)
    o = normalize(outgoing_vector)
    n = normalize(normal)

    # Compute the mirror reflection direction
    mirror_direction = 2 * np.dot(n, i) * n - i
    mirror_direction = normalize(mirror_direction)

    # Compute the angle to the mirror direction
    dot_product = np.dot(o, mirror_direction)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Anti-mirror reflectance calculation (higher angle to mirror direction means higher reflectance)
    brdf_value = np.exp(-(angle**2) / (2 * spread**2))

    # Invert reflectance: more intensity away from the mirror direction
    anti_mirror_value = (1 - brdf_value) * np.exp(-(angle**2) / (2 * spread_outer**2))

    return anti_mirror_value


def anisotropic_gaussian_brdf(
    incident_vector, outgoing_vector, normal, sigma_u, sigma_v
):
    """
    Compute the anisotropic Gaussian BRDF value.

    Parameters:
    - incident_vector: numpy array of shape (3,) representing the incident light direction.
    - outgoing_vector: numpy array of shape (3,) representing the outgoing view direction.
    - normal: numpy array of shape (3,) representing the surface normal at the intersection point.
    - sigma_u: float, standard deviation of the Gaussian along the tangent u direction.
    - sigma_v: float, standard deviation of the Gaussian along the tangent v direction.

    Returns:
    - float: The BRDF value for the given directions and surface properties.
    """

    # Normalize vectors
    i = normalize(incident_vector)
    o = normalize(outgoing_vector)
    n = normalize(normal)

    # Compute halfway vector
    h = normalize(i + o)

    # Define tangent and bitangent vectors for anisotropy directions
    tangent_u = np.array([1.0, 0.0, 0.0])  # Example tangent vector (u direction)
    tangent_v = np.array([0.0, 1.0, 0.0])  # Example bitangent vector (v direction)

    # Project halfway vector onto the tangent directions
    h_u = np.dot(h, tangent_u)
    h_v = np.dot(h, tangent_v)

    # Compute anisotropic Gaussian BRDF
    exponent = -((h_u**2) / (2 * sigma_u**2) + (h_v**2) / (2 * sigma_v**2))
    brdf_value = (1 / (2 * np.pi * sigma_u * sigma_v)) * np.exp(exponent)

    return brdf_value


def aniso_anti_mirror_brdf(
    incident_vector, outgoing_vector, normal, spread, sigma_u, sigma_v
):
    """
    Compute the anti-mirror BRDF value.

    Parameters:
    - incident_vector: numpy array of shape (3,) representing the incident light direction.
    - outgoing_vector: numpy array of shape (3,) representing the outgoing view direction.
    - normal: numpy array of shape (3,) representing the surface normal at the intersection point.
    - spread: float, controls the spread of the reflectance away from the mirror direction.

    Returns:
    - float: The BRDF value for the given directions and surface properties.
    """

    # Normalize vectors
    i = normalize(incident_vector)
    o = normalize(outgoing_vector)
    n = normalize(normal)

    # Compute the mirror reflection direction
    mirror_direction = 2 * np.dot(n, i) * n - i
    mirror_direction = normalize(mirror_direction)

    # Compute the angle to the mirror direction
    dot_product = np.dot(o, mirror_direction)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Anti-mirror reflectance calculation (higher angle to mirror direction means higher reflectance)
    brdf_value = np.exp(-(angle**2) / (2 * spread**2))

    # Invert reflectance: more intensity away from the mirror direction
    anti_mirror_value = (1 - brdf_value) * anisotropic_gaussian_brdf(
        incident_vector, outgoing_vector, normal, sigma_u, sigma_v
    )

    return anti_mirror_value


def BRDF_from_image_inverse(incident_vector, outgoing_vector, normal, img, size):
    """
    Compute the BRDF value from a inversed image.

    Parameters:
    - incident_vector: numpy array of shape (3,) representing the incident light direction.
    - outgoing_vector: numpy array of shape (3,) representing the outgoing view direction.
    - normal: numpy array of shape (3,) representing the surface normal at the intersection point.
    - img: the image to be inversed.
    - size: the size of the white ring around the inversed image.

    Returns:
    - float: The BRDF value for the given directions and surface properties.
    """

    diffuse = mirror_brdf(incident_vector, outgoing_vector, normal, size)
    img = BRDF_from_image_scale(incident_vector, outgoing_vector, normal, img)
    anti_mirror_value = diffuse * (1 - img)

    return anti_mirror_value


def BRDF_from_image_scale(incident_vector, outgoing_vector, normal, img, img_scale=1.5):
    """
    Compute the BRDF value from an image.

    Parameters:
    - incident_vector: numpy array of shape (3,) representing the incident light direction.
    - outgoing_vector: numpy array of shape (3,) representing the outgoing view direction.
    - normal: numpy array of shape (3,) representing the surface normal at the intersection point.
    - img: the BRDF image.
    - img_scale: size of the BRDF image.

    Returns:
    - float: The BRDF value for the given directions and surface properties.
    """

    # Normalize vectors
    i = normalize(incident_vector)
    o = normalize(outgoing_vector)

    # Compute halfway vector
    h = (i + o)[:2] / 2

    limit = 0.0625 * img_scale
    # limit = 0.0625 * 2.5

    idx = h / limit * 256 + 256
    idx0 = int(np.floor(idx[0]))
    if idx0 < 0 or idx0 >= 511:
        return np.zeros(3)
    f0 = idx[0] - idx0
    idx1 = int(np.floor(idx[1]))
    if idx1 < 0 or idx1 >= 511:
        return np.zeros(3)
    f1 = idx[1] - idx1

    anti_mirror_value = (
        img[idx0][idx1] * (1 - f0) * (1 - f1)
        + img[idx0 + 1][idx1] * (f0) * (1 - f1)
        + img[idx0][idx1 + 1] * (1 - f0) * (f1)
        + img[idx0 + 1][idx1 + 1] * (f0) * (f1)
    )

    return anti_mirror_value
