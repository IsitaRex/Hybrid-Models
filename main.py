import wandb
import torch
import argparse
import src.utils as utils

# parameters
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

SEED = 121212


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main(config):

    wandb.init(project=config["task"], config=config)

    if config["task"] == "Hybrid-Models-CNN":
        utils.task_cnn(config)
    elif config["task"] == "Hybrid-Models-GAN":
        utils.task_gan(config)
    elif config["task"] == "Hybrid-Models-AUTOENCODER":
        utils.task_autoencoder(config)
    else:
        print("No model defined for this task")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--task", type=str, default="Hybrid-Models-CNN")
    parser.add_argument("--seed", type=int, default=121212)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--use_wandb_offline", type=bool, default=False)
    parser.add_argument("--plot_roc", type=bool, default=False)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default=DEVICE)
    config = vars(parser.parse_args())
    utils.setup_wandb_logging(config)
    main(config)
