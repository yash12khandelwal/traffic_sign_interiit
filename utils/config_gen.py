import json
import argparse

parser = argparse.ArgumentParser(description="Config Path")
parser.add_argument("--save-path", "-fn", type=str,
                    default="../config/default_augment_conf.json")

args = parser.parse_args()

contents = dict()

# Dictionary to define which augmentation to execute
contents["Augmentation"] = {
    "Fog": True,
    "Snowflakes": True,
    "GaussianNoise": True,
    "Rain": True,
    "FastSnowyLandscape": True,
    "JpegCompression": True,
    "CoarsePepper": True,
    "Invert": True
}

# Parameters related to augmentations
contents["Fog"] = {}

contents["Snowflakes"] = {
    "flake_size": (0.1, 0.4),
    "speed": (0.01, 0.05)
}

contents["GaussianNoise"] = {"severity": 2}

contents["Rain"] = {
    "speed": (0.1, 0.3)
}

contents["FastSnowyLandscape"] = {
    "lightness_threshold": (100, 255),
    "lightness_multiplier": (1.0, 4.0)
}

contents["JpegCompression"] = {
    "compression": (70, 99)
}

contents["CoarsePepper"] = {
    "p": 0.25,
    "size_percent": (0.01, 0.1)
}

contents["Invert"] = {
    "p": 0.25,
    "per_channel": 0.5
}

# Probability for applying augmentations
contents["probability"] = 0.3

with open(args.save_path, "w") as file:
    json.dump(contents, file, indent=4)

file.close()
