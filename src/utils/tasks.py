import wandb
import torch
import src.models as models
import src.utils as utils

SEED = 121212


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def task_cnn(config):
        trainloader, testloader = utils.load_mnist(config["batch_size"], "resize")
        # define model
        model = models.LeNet5(config["n_classes"])
        # define loss function
        criterion = torch.nn.CrossEntropyLoss()
        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        # train model
        utils.training_loop_cnn(
            model,
            criterion,
            optimizer,
            trainloader,
            testloader,
            config["epochs"],
            device=config["device"],
            print_every=1,
            plot_roc=config["plot_roc"],
        )

        # # predict class of an image
        # print("Predicting class of an image")
        # for i in range(10):
        #     image = f"data/my_numbers_black/{i}.jpg"
        #     clas = utils.predict(model, image, device=DEVICE)
        #     print(f"The predicted class is {clas}, the actual class is {i}")

def task_gan(config):
        trainloader, testloader = utils.load_mnist(config["batch_size"])
        # define model
        generator = models.Generator(config["latent_dim"]).to(device=config["device"])
        discriminator = models.Discriminator().to(device=config["device"])
        # define loss function
        criterion = torch.nn.BCELoss()

        # optimizers
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config["lr"])
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=config["lr"]
        )
        generator, discriminator, G_losses, D_losses = utils.training_loop_gan(
            trainloader,
            discriminator,
            generator,
            discriminator_optimizer,
            generator_optimizer,
            config["device"],
            criterion,
            config["epochs"],
            config["latent_dim"],
        )

def task_autoencoder(config):
    trainloader, testloader = utils.load_mnist(config["batch_size"], "resize")
    # define model
    model = models.Autoencoder(1024, 784, 1024).to(device=config["device"])
    # define loss function
    criterion = torch.nn.MSELoss()
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # train model
    print("Training autoencoder")
    model, _,_,_ = utils.training_loop_autoencoder(
        model,
        criterion,
        optimizer,
        trainloader,
        testloader,
        config["epochs"],
        device=config["device"],
        print_every=1,
    )
    print("Training autoencoder done")
    wandb.finish()
    # wandb.init(project=config["task"], config=config)
    wandb.init(project="Hybrid-Models-CNN", config=config)
    # Use encoder data to train a LeNet5 model
    encoder = model.encoder_hidden
    encoder = encoder.to(device=config["device"])
    # define model
    model = models.LeNet5(config["n_classes"])
    # define loss function
    criterion = torch.nn.CrossEntropyLoss()
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # train model
    print("Training LeNet5 on encoder data")
    utils.training_loop_cnn(
        model,
        criterion,
        optimizer,
        trainloader,
        testloader,
        config["epochs"],
        device=config["device"],
        print_every=1,
        plot_roc=config["plot_roc"],
        encode_data=True,
        encoder=encoder
    )
    print("Training LeNet5 on encoder data done")
