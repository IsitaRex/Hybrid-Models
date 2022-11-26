import torch
import wandb
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from torch.functional import F
from sklearn.preprocessing import label_binarize
from datetime import datetime 
from sklearn.metrics import roc_curve, auc, confusion_matrix
from PIL import Image


def plot_roc_curve(fpr, tpr, roc_auc, n_classes=10, show=False):
    '''
    Function for plotting the ROC curve
    '''
    fig, ax = plt.subplots()
    lw = 2
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow', 'black', 'pink', 'purple', 'brown']
    for i,color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color,
                 lw=lw, label=f'ROC curve of class {i} (area = {roc_auc[i]:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc="lower right")
    if show:
        plt.show()

    # log the ROC curve to wandb
    wandb.log({"ROC Curve": wandb.Image(fig)})

def get_roc_curve(model, data_loader, device, n_classes=10, encode_data = False, encoder = None):
    '''
    Function for computing the ROC curve of the predictions over the entire data_loader
    '''

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_all = None
    y_score_all = None
    with torch.no_grad():
        model.eval()
        for X, y in data_loader:

            X = X.to(device)
            y = y.to(device)
            if encode_data:
                encoder.to(device)
                X = X.view(-1, 1024).to(device)
                X = encoder(X)
                X = X.view(-1, 1, 28, 28).to(device)

            _, y_prob_batch = model(X)
            y = y.cpu().numpy()
            y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            y_prob_batch = y_prob_batch.cpu().numpy()
            if y_all is None:
                y_all = y
                y_score_all = y_prob_batch.copy()
            else:
                y_all = np.concatenate((y_all, y), axis=0)
                y_score_all = np.concatenate((y_score_all, y_prob_batch), axis=0)

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_all[:,i],y_score_all[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # create confusion matrix and log it to wandb
        # transform y_all and y_score_all to 1D arrays
        y_all = np.argmax(y_all, axis=1)
        y_score_all = np.argmax(y_score_all, axis=1)
        cm = confusion_matrix(y_all,y_score_all)
        df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
        f, ax = plt.subplots(figsize=(5, 5))
        cmap = sn.cubehelix_palette(light=1, as_cmap=True)

        sn.heatmap(df_cm, cbar=False, annot=True, cmap=cmap, square=True, fmt='.0f',
                    annot_kws={'size': 10})
        ax.set_title("Confusion Matrix")
        wandb.log({"Confusion Matrix": wandb.Image(f)})
    
    return fpr, tpr, roc_auc

# from: https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
def get_accuracy(model, data_loader, device, encode_data = False, encoder = None):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)
            if encode_data:
                encoder.to(device)
                X = X.view(-1, 1024).to(device)
                X = encoder(X)
                X = X.view(-1, 1, 28, 28).to(device)


            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

# from: https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
def train(train_loader, model, criterion, optimizer, device, encode_data = False, encoder = None):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        X = X.to(device)
        if encode_data:
            encoder.to(device)
            X = X.view(-1, 1024).to(device)
            
            X = encoder(X)
            X = X.view(-1, 1, 28, 28).to(device)
        y_true = y_true.to(device)
        model.to(device)
    
        # Forward pass
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

# from: https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
def validate(valid_loader, model, criterion, device, encode_data = False, encoder = None):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        X = X.to(device)
        if encode_data:
            encoder.to(device)
            X = X.view(-1, 1024).to(device)
            X = encoder(X)
            X = X.view(-1, 1, 28, 28).to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)
        

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss

def training_loop_cnn(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1, plot_roc=True, encode_data = False, encoder = None):

    '''
    Function defining the entire training loop
    '''
    
    # raise error ir endoce_data is True but no encoder is given
    if encode_data and encoder is None:
        raise ValueError("An encoder must be provided to encode data.")

    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device, encode_data, encoder)
        train_losses.append(train_loss)
        
        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device, encode_data, encoder)
            valid_losses.append(valid_loss)

        wandb.log({"CNN/train_loss": float(train_loss), "CNN/val_loss": float(valid_loss)}, step=epoch)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device, encode_data=encode_data, encoder=encoder)
            valid_acc = get_accuracy(model, valid_loader, device=device, encode_data=encode_data, encoder=encoder)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

            wandb.log({"Train accuracy": float(100 * train_acc), "Valid accuracy": float(100 * valid_acc)}, step=epoch)
            # get ROC curve
    fpr, tpr, roc_auc = get_roc_curve(model, valid_loader, device=device, n_classes=10, encode_data=encode_data, encoder=encoder)
    plot_roc_curve(fpr, tpr, roc_auc, show=plot_roc)
           
    return model, optimizer, (train_losses, valid_losses)

def predict(model, image, device):
    '''
    Function for predicting the class of an image in .jpg format
    '''

    # load image and import the libraries

    # define the transformations
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # load the image
    image = Image.open(image)

    # turn to grayscale
    image = image.convert('1')

    # apply the transformations
    image = transform(image)

    # add a batch dimension
    image = image.unsqueeze(0)

    # move the input and model to GPU for speed if available

    image = image.to(device)

    # predict the class of the image
    with torch.no_grad():
        model.eval()
        _, output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted


                                
