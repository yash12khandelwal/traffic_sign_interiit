import wandb
import os


def init_wandb(model, args=None) -> None:
    """
    Initialize project on Weights & Biases
    """
    wandb.login(key=args.wandb_api_key)
    wandb.init(
        name=args.wandb_name,
        project="traffic-sign",
        id=args.wandb_id,
        dir="~/",
    )
    if args:
        wandb.config.update(args)

    wandb.watch(model, log="all")


def save_model_wandb(save_path: str):
    """
    Save model weights to wandb
    """
    wandb.save(os.path.abspath(save_path))


def wandb_log(train_loss: float, val_loss: float, val_acc: float, epoch: int):
    """
    Logs the accuracy and loss to wandb
    """
    wandb.log({
        'Training loss': train_loss,
        'Validation loss': val_loss,
        'Validation Accuracy': val_acc
    })
