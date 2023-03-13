from torch.utils.data import DataLoader
import cv2
from typing import Optional
import matplotlib.pyplot as plt
import torch
###Code To One Hot Encode Labels


def one_hot_encode(labels, num_classes):
    # Reshape the labels to (batch_size, height, width)
    labels = labels.reshape(labels.size(0), labels.size(2), labels.size(3))
    
    # Create a tensor of shape (batch_size, num_classes, height, width) filled with zeros
    one_hot = torch.zeros(labels.size(0), num_classes, labels.size(1), labels.size(2))
    
    # Use torch.scatter to set the appropriate indices to 1
    index = torch.unsqueeze(labels, 1)
    one_hot.scatter_(1, index, 1)
    
    return one_hot

### Code for Bitwise Segmentation Of Input Images with reference to Predicted Lung Masks
# (Basically Lung Extraction :D )

def visualize_segmented_lungs(my_dataloader: DataLoader, predictions: np.ndarray, save_path: Optional[str] = None, visualize: bool = False):
    count = 0
    segmented_lungs_list = []
    for inputs, labels in my_dataloader:
        for i in range(len(labels)):
            pred_image_masks = predictions[count]
            count += 1
            input_images = inputs[i]
            input_images = torch.squeeze(input_images, 0)
            input_images = input_images.numpy()
            input_images = np.expand_dims(input_images, 2)
            pred_image_masks = np.expand_dims(pred_image_masks, 2)
            pred_image_masks = pred_image_masks.astype(np.uint8)
            input_images = input_images.astype(np.uint8)
            segmented_region = cv2.bitwise_and(input_images, input_images, mask=pred_image_masks)
            segmented_lungs_list.append(segmented_region)

    segmented_lungs_bitwise = np.asarray(segmented_lungs_list)
    if save_path is not None:
        np.save(save_path, segmented_lungs_bitwise)

    if visualize:
        for i in range(segmented_lungs_bitwise.shape[0]):
            plt.figure(figsize=(15, 5))

            # Plot original image
            plt.subplot(131)
            plt.imshow(inputs[i][0], cmap='gray')
            plt.title('Original Image')

            # Plot mask
            plt.subplot(132)
            plt.imshow(predictions[i], cmap='gray')
            plt.title('Mask')

            # Plot segmented lung
            plt.subplot(133)
            plt.imshow(segmented_lungs_bitwise[i], cmap='gray')
            plt.title('Segmented Lungs')

            plt.show()          



### Code To Calculate Mean and Standard Deviation for a given Image dataset, serialized with a Dataloader
# For data intensive images, use a batch_size >4 for the data_loader, and average the below function outputs, else: batch_size=1

def mean_std(loader):
  images, labels = next(iter(loader))
  # shape of images = [b,c,w,h]
  mean, std = images.mean([0,2,3]), images.std([0,2,3])
  return mean, std

mean, std = mean_std(image_data_loader)
print("mean and std: \n", mean, std)

#Now, images can be normalized if required, using:
#transforms.Normalize(( mean,), (std,))]) (Code Valid For single channel images)
   
