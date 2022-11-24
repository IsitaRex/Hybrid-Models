import wandb
import torch
import argparse 
from src.utils.load_mnist import load_mnist
from src.models.lenet5 import LeNet5
from src.utils.train import training_loop, predict
from src.utils.wandb_setup import setup_wandb_logging

# parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 121212


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def main(config):

    wandb.init(project=config["task"], config=config)
    # load data
    trainloader, testloader = load_mnist(config["batch_size"])
    if config["task"] == "Hybrid-Models-CNN":
        # define model
        model = LeNet5(config["n_classes"])
        # define loss function
        criterion = torch.nn.CrossEntropyLoss()
        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        # train model
        training_loop(model, criterion, optimizer, trainloader, testloader, config["epochs"], device=DEVICE, print_every=1, plot_roc=config["plot_roc"])

        # predict class of an image
        print("Predicting class of an image")
        image = "data/2.jpg"
        clas =predict(model, image, device=DEVICE)
        print(f"The predicted class is {clas}")
    elif config["task"] == "Hybrid-Models-GAN":
        pass
    elif config["task"] == "Hybrid-Models-AUTOENCODER":
        pass
    else :
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
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--use_wandb_offline", type=bool, default=False)
    parser.add_argument("--plot_roc", type=bool, default=False)
    config = vars(parser.parse_args())
    setup_wandb_logging(config)
    main(config)


