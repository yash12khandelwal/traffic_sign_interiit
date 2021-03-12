import yaml
import argparse

parser = argparse.ArgumentParser(description="Config Path")
parser.add_argument("--save-path", "-fn", type=str,
                    default="../config/configs.yaml")

args = parser.parse_args()


contents = {}

contents["Augmentation"] = {
    "Fog": [False],
    "Snowflakes": [False],
    "GaussianNoise": [True]
}

contents["Fog"] = {}

contents["Snowflakes"] = {
    "flake_size": [0.1, 0.4],
    "speed": [0.01, 0.05]
}

contents["GaussianNoise"] = {"severity": [2]}


with open(args.save_path, "w") as file:
    yaml.dump(contents, file)
file.close()
