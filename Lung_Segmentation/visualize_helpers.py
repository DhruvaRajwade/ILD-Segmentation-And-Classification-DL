
#Plot Original_Image, Ground_Truth And Predictions
count=0
for inputs, labels in my_dataloader:
    for i in range(len(labels)):

        img=predictions[count]
        count+=1
        original_label=labels[i]
        original_label=torch.squeeze(original_label, 0)
        orig=inputs[i]
        orig=torch.squeeze(orig, 0)

        fig = plt.figure(figsize=(8, 8))
        plt.subplot(1,3,1)
        plt.imshow(original_label, cmap='gray')
        plt.title("Target Mask")
        plt.subplot(1,3,2)
        plt.imshow(orig, cmap='gray')
        plt.title("Original image")
        plt.subplot(1,3,3)
        plt.imshow(img, cmap='gray')
        plt.title("Predicted Mask")
        plt.show()
        plt.close()

###Code For Prediction_Mask_Comparision Vizualization (Used in the paper)

#Load Saved Prediction images of the respective models, as well as original images and labels which we have saved as numpy arrays

import numpy as np
import matplotlib.pyplot as plt

# Load the arrays of images
#array1 = np.load('pathoriginal_images.npy')
array1 = np.load('path/original_labels.npy')
array2 = np.load('path/unet.npy')
array4 = np.load('path/R2_unet.npy')
array5 = np.load('path/attn_r2_unet.npy')
array3 = np.load('path/attn_unet.npy')
array3=np.squeeze(array3,1)
array6=np.load('path/original_images.npy')
# Assume array.shape[0] is same for all above arrays)
#Below code saves comparitive plot figures for all images in the arrays: Using a high RAM runtime is recommended for iterating over the whole datasets
for i in range(array1.shape[0]):
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    axs[0].imshow(array6[i],cmap='gray')
    axs[1].imshow(array1[i],cmap='gray')
    axs[2].imshow(array2[i],cmap='gray')
    axs[3].imshow(array3[i],cmap='gray')
    axs[4].imshow(array4[i],cmap='gray')
    axs[5].imshow(array5[i],cmap='gray')
    axs[0].title.set_text("Original")
    axs[1].title.set_text("Ground Truth")
    axs[2].title.set_text("UNet")
    axs[3].title.set_text("Attention_UNet")
    axs[4].title.set_text("R2_UNet")
    axs[5].title.set_text("Attention_R2_UNet")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f'combined_figure{i}.png', bbox_inches='tight')

###Todo Code For Contouring
        
