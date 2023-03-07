from models import *
from loss import CrossEntropy2d
from metrics import compute_dice, compute_iou, compute_accuracy
from data_helpers import *

# Use Example script in data_helpers.py to preprocess and load your dataset
# User should set hyperparameters like lr(learning rate), num_epochs
# Code assumes you have a train_dataloader and a val_dataloader containing images and labels, both with shape [b,n,h,w]

model=my_model() # Choice Of ('U_Net', 'AttU_Net', 'R2AttU_Net', 'R2U_Net')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Added GPU support, as well as distributed GPU training (Todo: Add Multiworker distributed strategy support)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    #model.load_state_dict(torch.load('best_model.pth')) ##To resume training from a saved load_state_dict
    model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def loss_calc(pred, label, device):
    criterion = CrossEntropy2d().cuda(device)
    return criterion(pred, label)



###Training Loop###
import csv
from tqdm import tqdm
#Learning_Rate scheduler Example (Uncomment below line and scheduler.step() line to implement)
#from torch.optim.lr_scheduler import ExponentialLR

#scheduler = ExponentialLR(optimizer, gamma=0.98)
num_epochs = 100
patience = 10 #Patience parameter for Early Stopping (To prevent Overfitting)
best_val_loss = float("inf")
counter = 0

# Set the model to training mode
model.train()

# Initialize CSV file for storing results
csv_file = open('/kaggle/working/results.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])

for epoch in range(num_epochs):
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
            #print(outputs.shape)

            # Compute the loss
            loss = loss_calc(outputs, labels[:,0,:,:], device)

            # Backward pass
            loss.backward()

            # Optimize the model
            optimizer.step()
            #scheduler.step()

            # Update train loss and accuracy
            train_loss += loss.item()


            # Update the progress bar
            pbar.update(1)

        # Compute train loss and accuracy
        train_loss /= len(train_dataloader)


        # Perform validation
        #lrz=scheduler.get_last_lr() (Get learning_rate for epoch)
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (val_images, val_labels) in enumerate(val_dataloader):
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_images)
                val_loss += loss_calc(val_outputs, val_labels[:,0,:,:], device)
                _, predicted = torch.max(val_outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels[:,0,:,:]).sum().item()
        val_loss /= len(val_dataloader)
        v_loss=float(val_loss)

        print(f'Validation loss for epoch {epoch+1}: {val_loss:.4f}')

        # Write results to CSV file
        csv_writer.writerow([epoch+1,train_loss, v_loss])

        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

# Close CSV file
csv_file.close()
