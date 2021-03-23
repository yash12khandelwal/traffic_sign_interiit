import json
import os
import imgaug.augmenters as iaa
import cv2
import random

"""
Maps the given string to the corresponding imgaug function.

Note: Add the function call when adding new augmentations
"""
augments = {
    "Fog": iaa.weather.Fog,
    "Snowflakes": iaa.Snowflakes,
    "GaussianNoise": iaa.imgcorruptlike.GaussianNoise,
    "Rain": iaa.Rain,
    "FastSnowyLandscape": iaa.FastSnowyLandscape,
    "JpegCompression": iaa.JpegCompression,
    "CoarsePepper": iaa.CoarsePepper,
    "Invert": iaa.Invert
}


def load_augments(augments_conf: dict, top=1):
    """
    Function that returns the augmentatons list from available ones

    Args:
        augments_conf (dict): Augmentations dictionary of the config file.
        top (int, optional): Number of augmentations to apply on a single image. Defaults to 1.

    Returns:
        iaa.Sequential: Sequential of list of transforms according to augments_conf and top
    """

    transforms = []

    total_augments = [k for (k, v) in augments_conf.Use.items() if v]

    # Randomly chooses 'top' augmentations from the list of avialable augmentations
    apply_augments = random.choices(total_augments, k=top)

    for k in apply_augments:
        # Applies the randomly chosen augmentations with given probability
        aug = iaa.Sometimes(augments_conf.probability,
                            augments[k](**augments_conf[k]))
        transforms.append(aug)

    seq = iaa.Sequential(transforms)
    return seq
