import random
import numpy as np
import torch
import os


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def convert_onnx(model, args):
    """
    Converts the model to .onnx format and saves it

    Args:
        model (Torch Model): Trained Model
        args (TrainOptions): TrainOptions class (refer options/train_options.py)

    Returns:
        str: path to converted .onnx file
    """

    batch_size = 1
    inputs = torch.randn(
        batch_size, 3, args.size[0], args.size[1], requires_grad=True).to(args.device)
    name = f'opt_{args.wandb_name}.onnx'
    save_path = os.path.join(args.snapshot_dir, name)

    torch.onnx.export(model, inputs, save_path, export_params=True, input_names=['input'], output_names=[
                      'output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    return save_path
