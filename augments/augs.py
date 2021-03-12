import yaml
import os
import imgaug.augmenters as iaa
import cv2

augments = {"Fog": iaa.weather.Fog, "Snowflakes": iaa.Snowflakes,
            "GaussianNoise": iaa.imgcorruptlike.GaussianNoise}


def load_augments(config_path='../config/configs.yaml'):

    with open(config_path, "r") as file:
        try:
            hyperparams = yaml.load(file, Loader=yaml.FullLoader)
        except:
            hyperparams = yaml.load(file)
    file.close()

    transforms = []

    for key, val in hyperparams['Augmentation'].items():
        if val[0] == True:
            transforms.append(augments[key](**hyperparams[key]))

    seq = iaa.Sequential(transforms)#.augment_image
    return seq
