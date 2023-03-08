from models import *
from loss import CrossEntropy2d
from metrics import compute_dice, compute_iou, compute_accuracy
from data_helpers import *
import matplotlib.pyplot as plt
# Define A test_dataloader using the data_helpers.py csv_file

# Define Model, and load Saved Weights
#Some previous steps to follow
"""model= model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
# Uncomment the below line if the model was trained using the DataParallel Strategy [The state_dict keys are stored with the prefix module (eg; module.conv.conv0..., and loading the saved model will prove unsuccessful] )
#model = nn.DataParallel(model)  ##Even if you do not use >1 GPUs for evaluation, uncomment if you used >1 GPUs for training
model.load_state_dict(torch.load('best_model.pth'))   
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)"""


def evaluate_model(model, dataloader, device, save_predictions=False, save_path=None):
'''
This function evaluates the trained model on the given dataloader and returns the mean IoU, mean Dice coefficient, and mean accuracy.
rgs:
- model: Trained PyTorch model
- dataloader: PyTorch DataLoader containing the evaluation data
- device: Device to run the evaluation on (GPU or CPU)
- save_predictions: Boolean indicating whether to save the predicted masks
- save_path: Path to save the predicted masks (required if save_predictions is True)

Returns:
- mean_iou: Mean Intersection over Union (IoU) score
- mean_dice: Mean Dice coefficient score
- mean_acc: Mean accuracy score
- predictions: Numpy array containing the predicted masks
'''

# Set the model to evaluation mode
    model.eval()

    # Initialize lists for storing metrics and predictions
    predictions = []
    ious = []
    dices = []
    accs = []

    # Iterate over the dataloader
    for inputs, labels in dataloader:
        # Move data to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            # Forward pass
            output_mask = model(inputs)

            # Get the predicted mask
            _, predicted_mask = torch.max(output_mask, 1)
            predicted_mask = predicted_mask.squeeze(1)

            # Convert the predicted mask to a numpy array
            predicted_mask = predicted_mask.to('cpu').numpy()

            # Convert the labels to a numpy array
            labels = labels.to('cpu').numpy()

            # Calculate the IoU
            iou = compute_iou(predicted_mask, labels)

            # Calculate the Dice coefficient
            dice = compute_dice(predicted_mask, labels)

            # Calculate the accuracy
            accuracy = compute_accuracy(predicted_mask, labels)

            ious.append(iou)
            dices.append(dice)
            accs.append(accuracy)

            predictions.append(predicted_mask)

    # Calculate mean metrics
    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)
    mean_acc = np.mean(accs)

    if save_predictions:
        # Save predictions to file
        predictions = np.asarray(predictions)
        predictions = np.squeeze(predictions, axis=1)
        np.save(save_path, predictions)

    return mean_iou, mean_dice, mean_acc, predictions
