import torch
import wandb
from datetime import datetime

def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    
    for X, _ in train_loader:

        X = X.view(-1, 1024).to(device)
        optimizer.zero_grad()
        
        X = X.to(device)
        model.to(device)
    
        # Forward pass
        outputs = model(X) 
        loss = criterion(outputs, X) 

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = loss / len(train_loader)
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, _ in valid_loader:
        X = X.view(-1, 1024).to(device)
        X = X.to(device)

        # Forward pass and record loss
        outputs = model(X) 
        loss = criterion(outputs, X) 
        
    epoch_loss = loss/ len(valid_loader)
        
    return model, epoch_loss

def training_loop_autoencoder(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1, plot_roc=True):

    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        wandb.log({"train_loss": float(train_loss), "val_loss": float(valid_loss)}, step=epoch)

        print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t')

        # log to wandb
        wandb.log({"Autoencoder/train_loss": float(train_loss), "Autoencoder/val_loss": float(valid_loss)}, step=epoch)
           
    return model, optimizer, train_losses, valid_losses
