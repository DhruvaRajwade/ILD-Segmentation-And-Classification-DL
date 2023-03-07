from models import *
from loss import CrossEntropy2d
from metrics import compute_dice, compute_iou, compute_accuracy
from data_helpers import *
import matplotlib.pyplot as plt
# Define A test_dataloader using the data_helpers.py csv_file

# Define Model, and load Saved Weights

model= model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
# Uncomment the below line if the model was trained using the DataParallel Strategy [The state_dict keys are stored with the prefix module (eg; module.conv.conv0..., and loading the saved model will prove unsuccessful] )
#model = nn.DataParallel(model)  ##Even if you do not use >1 GPUs for evaluation, uncomment if you used >1 GPUs for training
model.load_state_dict(torch.load('best_model.pth'))   
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)


predictions = []
ious=[]
dices=[]
accs=[]


for inputs, labels in train_dataloader:
    # Move data to the device (GPU or CPU)
    inputs = inputs.to(device)
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

# Print the mean IoU, Dice coefficient, and accuracy
print("Mean IoU: {:.4f}".format(np.mean(ious)))
print("Mean Dice: {:.4f}".format(np.mean(dices)))
print("Mean Accuracy: {:.4f}".format(np.mean(accs)))


#Save Model Prediction Images For Visualization And Analysis
predictions=np.asarray(predictions)
predictions.shape
predictions=np.squeeze(predictions, axis=1)
np.save('predictions_path', predictions)
