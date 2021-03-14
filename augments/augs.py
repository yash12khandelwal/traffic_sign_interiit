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


def load_augments(config_path='../config/default_augment_conf.json', rand=False):
    """ Function that returns the augmentatons list from available ones

    Args:
        config_path (str, optional): path to augmentations configs. Defaults to '../config/default_augment_conf.json'.
        rand (bool, optional): Flag to use a single random augmentations. Defaults to False.

    Returns:
        iaa.Sequential: Sequential of list of tranforms according to config file and rand
    """

    with open(config_path, 'r') as f:
        augments_conf= json.load(f)

    transforms = []

    total_augments = []
    for (k, v) in augments_conf['Augmentation'].items():
        if v:
            total_augments.append(k)

    if rand:
        choice = random.choice(total_augments)
        aug = iaa.Sometimes(augments_conf['probability'], augments[choice](**augments_conf[choice]))
        transforms.append(aug)
    else:
        for k in total_augments:
            aug = iaa.Sometimes(augments_conf['probability'], augments[k](**augments_conf[k]))
            transforms.append(aug)

    seq = iaa.Sequential(transforms)
    return seq
