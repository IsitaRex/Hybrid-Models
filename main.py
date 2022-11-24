import wandb
import torch
from src.utils.load_mnist import load_mnist
from src.models.lenet5 import LeNet5
from src.utils.train import training_loop

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 5
IMG_SIZE = 32
N_CLASSES = 10
TASK = "Hybrid-Models-CNN"
SEED = 121212
DEVICE = "mps" 
#if torch.cuda.is_available() else "cpu"
breakpoint()
config = {
    "seed": RANDOM_SEED,
    "lr": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": N_EPOCHS,
    "img_size": IMG_SIZE,
    "n_classes": N_CLASSES,
    "seed": SEED
}


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def main(config):

    wandb.init(project=TASK, config=config)
    # load data
    trainloader, testloader = load_mnist(config["batch_size"])
    # define model
    model = LeNet5(config["n_classes"])
    # define loss function
    criterion = torch.nn.CrossEntropyLoss()
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # train model
    training_loop(model, criterion, optimizer, trainloader, testloader, config["epochs"], device=DEVICE, print_every=1)

    wandb.finish()


if __name__ == "__main__":

    main(config)