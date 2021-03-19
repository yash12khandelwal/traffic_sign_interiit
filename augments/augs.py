import json
import os
import imgaug.augmenters as iaa
import cv2
import random

"""
Add the function call when adding new augmentations
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


def load_augments(augments_conf, top=1):
    """ Function that returns the augmentatons list from available ones

    Args:
        config_path (str, optional): path to augmentations configs. Defaults to '../config/default_augment_conf.json'.
        rand (bool, optional): Flag to use a single random augmentations. Defaults to False.

    Returns:
        iaa.Sequential: Sequential of list of tranforms according to config file and rand
    """

    transforms = []

    total_augments = [ k for (k, v) in augments_conf.Use.items() if v ]

    apply_augments = random.choices(total_augments, k=top)

    for k in apply_augments:
        aug = iaa.Sometimes(augments_conf.probability, augments[k](**augments_conf[k]))
        transforms.append(aug)

    seq = iaa.Sequential(transforms)
    return seq
