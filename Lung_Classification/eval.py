import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from models import *
from data_helpers import *



def evaluate(net, test_loader, device, print_report=True, print_auc=True, print_confusion_matrix=True):
    """
    Args:

    net: PyTorch model class.

    test_loader: PyTorch DataLoader class.

    device: PyTorch device class

    print_report (optional): Boolean flag, with the option to print metrics like precision, recall, accuracy and F1 score

    print_auc (optional): Boolean flag, with the option to print the ROC curve

    print_confusion_matrix (optional): Boolean flag, with the option to print the Confusion Matrix

    """
    net.eval()  # Set the network to evaluation mode

    # Initialize variables for accuracy and predictions
    correct = 0
    total = 0
    y_true = []
    y_score = []

    # Iterate over the test data
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device).long()
            outputs = net(images)

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Append true labels and predicted scores for AUC calculation
            y_true.extend(labels.cpu().numpy())
            y_score.extend(outputs.cpu().numpy())

    # Compute and print accuracy
    accuracy = 100 * correct / total
    print('Accuracy on the test set: %d %%' % accuracy)

    # Calculate AUC
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_score = np.argmax(y_score, axis=1)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Print AUC curve if requested
    if print_auc:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) curve')
        plt.legend(loc="lower right")
        plt.show()

    # Print classification report if requested
    if print_report:
        target_names = ['HP', 'IPF']  # Replace with your class names
        print(classification_report(y_true, y_score, target_names=target_names))

    # Print confusion matrix if requested
    if print_confusion_matrix:
        cm = confusion_matrix(y_true, y_score)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()
