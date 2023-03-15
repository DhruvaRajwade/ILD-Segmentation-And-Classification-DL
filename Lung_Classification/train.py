import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import *
from data_helpers import *


def train(model,train_loader, val_loader, num_epochs, optimizer, criterion, scheduler=None, early_stopping_patience=None, plot=True, device='cpu', **kwargs):
    """
    Trains a PyTorch model using the given train and validation dataloaders for the specified number of epochs.
    
    Args:
        model (torch.nn.Module): A PyTorch model
        train_loader (torch.utils.data.DataLoader): A PyTorch DataLoader object for the training dataset.
        val_loader (torch.utils.data.DataLoader): A PyTorch DataLoader object for the validation dataset.
        num_epochs (int): The number of epochs to train for.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer object.
        criterion (torch.nn.Module): A PyTorch loss function object.
        scheduler (torch.optim.lr_scheduler.*): A PyTorch learning rate scheduler object. (default: None)
        early_stopping_patience (int): The number of epochs to wait for validation loss to improve before early stopping. (default: None)
        plot (bool): Whether to plot the train and validation losses and accuracies. (default: True)
        device (str): The device to run the model on. (default: 'cpu')
        **kwargs: Any additional arguments that may be required by the optimizer or the model.
    
    Returns:
        train_losses (list): A list of training losses for each epoch.
        val_losses (list): A list of validation losses for each epoch.
        train_accs (list): A list of training accuracies for each epoch.
        val_accs (list): A list of validation accuracies for each epoch.
    """
    Net=model()
    net=Net.to(device)
    # Initialize variables for early stopping
    if early_stopping_patience is not None:
        best_val_loss = float('inf')
        counter = 0

    # Initialize lists for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        net.train()  # Set the network to training mode
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):

            # Get the inputs and labels
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).long()
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize

            outputs = net(inputs)
           # print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_acc = 100 * correct / total
            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                # print('[Epoch %d, Batch %d] Loss: %.3f' %
                #     (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Evaluate the network on the validation set
        net.eval()  # Set the network to evaluation mode
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(val_loader, total=len(val_loader)):
                images, labels = data
                inputs, labels = inputs.to(device), labels.to(device).long()
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_acc = 100 * correct / total
            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                # print('[Epoch %d, Batch %d] Loss: %.3f' %
                #      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Evaluate the network on the validation set
        net.eval()  # Set the network to evaluation mode
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device).long()
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print statistics
        train_loss = running_loss / len(train_loader)
        val_loss /= len(val_loader)
        # train_acc = 100 * correct / total
        val_acc = 100 * correct / total
        print('[Epoch %d] Train Loss: %.3f, Val Loss: %.3f, Train Acc: %d %%, Val Acc: %d %%' %
              (epoch + 1, train_loss, val_loss, train_acc, val_acc))

        # Early stopping
        if early_stopping_patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= early_stopping_patience:
                    print("Early stopping after epoch ", epoch + 1)
                    break

        # Update lists for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update the learning rate scheduler
        #scheduler.step()
        if scheduler is not None:
            scheduler.step()

    if plot:

        # Plot train and val acc vs epoch
        plt.plot(range(1, epoch+2), train_accs, label="Train")
        plt.plot(range(1, epoch+2), val_accs, label="Validation")
        plt.title("Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.show()

        # Plot train and val loss vs epoch
        plt.plot(range(1, epoch+2), train_losses, label="Train")
        plt.plot(range(1, epoch+2), val_losses, label="Validation")
        plt.title("Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
