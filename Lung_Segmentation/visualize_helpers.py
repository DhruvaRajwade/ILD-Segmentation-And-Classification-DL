import cv2
import matplotlib.pyplot as plt
import numpy as np


#Plot Original_Image, Ground_Truth And Predictions
def visualize_segmentation(predictions: Numpy array, my_dataloader: Torch Dataloader)
    count = 0
    for inputs, labels in my_dataloader:
        for i in range(len(labels)):
            img = predictions[count]
            count += 1
            original_label = labels[i]
            original_label = torch.squeeze(original_label, 0)
            orig = inputs[i]
            orig = torch.squeeze(orig, 0)

            fig = plt.figure(figsize=(8, 8))
            plt.subplot(1, 3, 1)
            plt.imshow(original_label, cmap='gray')
            plt.title("Target Mask")
            plt.subplot(1, 3, 2)
            plt.imshow(orig, cmap='gray')
            plt.title("Original image")
            plt.subplot(1, 3, 3)
            plt.imshow(img, cmap='gray')
            plt.title("Predicted Mask")
            plt.show()
            plt.close()


###Code For Prediction_Mask_Comparision Vizualization

#Load Saved Prediction images of the respective models, as well as original images and labels which we have saved as numpy arrays


def plot_model_predictions(model_names, original_images_path, ground_truth_path, *predictions_paths):
"""
Plot model predictions and save the combined figure for each image
Parameters:
model_names (list): List of names for the models to be plotted
original_images_path (str): Path to the numpy array of original images
ground_truth_path (str): Path to the numpy array of ground truth masks
*predictions_paths (tuple): Tuple containing paths to numpy arrays of predictions for each model

Returns:
None
"""
    num_models = len(prediction_paths)
    assert num_models == len(model_names), "Number of model names should match number of prediction arrays"
    original_images = np.load(original_images_path)
    ground_truth = np.load(ground_truth_path)
    predictions = [np.load(path) for path in predictions_paths]
    
    
    
    for i in range(ground_truth.shape[0]):
        fig, axs = plt.subplots(1, len(model_names)+2, figsize=(15, 3))
        axs[0].imshow(original_images[i], cmap='gray')
        axs[1].imshow(ground_truth[i], cmap='gray')
        axs[0].title.set_text("Original")
        axs[1].title.set_text("Ground Truth")
    
    for j, pred in enumerate(predictions):
        axs[j+2].imshow(pred[i], cmap='gray')
        axs[j+2].title.set_text(model_names[j])
    
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f'combined_figure{i}.png', bbox_inches='tight')

    
###Todo Code For Contouring
        
