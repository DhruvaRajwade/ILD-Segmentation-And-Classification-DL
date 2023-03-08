from models import *
from loss import *   # All losses except CrossEntropy2D() require One Hot Encoded labels, refer to utils.py for the one hot-encode function
#from metrics import compute_dice, compute_iou, compute_accuracy  :For Evaluation
from data_helpers import *

# Code assumes you have a train_dataloader and a val_dataloader containing images and labels, both with shape [b,n,h,w]
import csv
from tqdm import tqdm

def train(model, device, train_dataloader, val_dataloader, epochs, criterion, optimizer, loss_fn, lr_scheduler=None, lr_rate=None, save_path=None, patience=10):
"""Args:
    model (nn.Module): The neural network model to train.
    device (torch.device): The device to train the model on (e.g. CPU or GPU).
    train_dataloader (DataLoader): The dataloader for the training set.
    val_dataloader (DataLoader): The dataloader for the validation set.
    optimizer (torch.optim.Optimizer): The optimizer to use for training the model.
    criterion (callable): The loss function to use for training the model.
    epochs (int): The number of epochs to train for.
    lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler to use (default: None).
    lr_rate (float, optional): The learning rate to use (default: None).
    save_path (str, optional): The path to save the trained model to (default: None).    """
    # Added distributed GPU training
    if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    model.to(device)
    #criterion.to(device) ##Depends on Selected Loss Function
    # Initialize CSV file for storing results
    csv_file = None
    csv_writer = None
    if save_path is not None:
        csv_file = open(save_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])

    #Learning_Rate scheduler Example
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer, gamma=0.98)

    best_val_loss = float("inf")
    counter = 0

    # Set the model to training mode
    model.train()

    for epoch in range(epochs):
        # Use tqdm to display a progress bar during training
        with tqdm(total=len(train_dataloader)) as pbar:
            # Initialize variables for computing train loss and accuracy
            train_loss = 0
            correct = 0
            total = 0

            # Iterate over the training data
            for i, (images, labels) in enumerate(train_dataloader):
                # Move the data to the device
                images = images.to(device)
                labels = labels.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)

                # Compute the loss
                loss = criterion(outputs, labels[:,0,:,:], device)

                # Backward pass
                loss.backward()

                # Optimize the model
                optimizer.step()

                # Update train loss and accuracy
                train_loss += loss.item()

                # Update the progress bar
                pbar.update(1)

            # Compute train loss and accuracy
            train_loss /= len(train_dataloader)

            # Perform validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (val_images, val_labels) in enumerate(val_dataloader):
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_images)
                    val_loss += loss_fn(val_outputs, val_labels[:,0,:,:], device)
                    _, predicted = torch.max(val_outputs.data, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels[:,0,:,:]).sum().item()
            val_loss /= len(val_dataloader)
            v_loss=float(val_loss)

            print(f'Validation loss for epoch {epoch+1}: {val_loss:.4f}')

            # Write results to CSV file
            if csv_writer is not None:
                csv_writer.writerow([epoch+1,train_loss, v_loss])

            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    if csv_file is not None:
                        csv_file.close()
                    return

            if lr_scheduler is not None:
                lr_scheduler.step()

    if csv_file is not None:
        csv_file.close()
    
