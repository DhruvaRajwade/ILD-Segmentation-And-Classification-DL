### Code for Bitwise Segmentation Of Input Images with reference to Predicted Lung Masks
# (Basically Lung Extraction :D )
from torch.utils.data import DataLoader
import cv2
count=0
segmented_lungs_list = []
for inputs, labels in my_dataloader:
    for i in range(len(labels)):

        pred_image_masks=predictions[count]
        count+=1
        input_images=inputs[i]
        input_images=torch.squeeze(input_images, 0)
        input_images=input_images.numpy()
        input_images=np.expand_dims(input_images,2)
        pred_image_masks=np.expand_dims(pred_image_masks,2)
        pred_image_masks=pred_image_masks.astype(np.uint8)
        input_images=input_images.astype(np.uint8)
        segmented_region = cv2.bitwise_and(input_images,input_images, mask=img)
        segmented_lungs_list.append(segmented_region)

segmented_lungs_bitwise = np.asarray(segmented_lungs_list)
np.save('PATH', segmented_lungs_bitwise)
#Visualize Bitwise_Segmented Lungs

for i in range(segmented_lungs_bitwise.shape[0]):
    plt.imshow(segmented_lungs_bitwise[i], cmap='gray')
    plt.show()

###Code to save original images and labels as numpy arrays for visualization
input_image_list=[]
input_label_list=[]
for images, labels in train_data:
    input_image_list.append(images.numpy())
    input_label_list.append(labels.numpy())

input_image_list=np.asarray(input_image_list)
input_label_list=np.asarray(input_label_list)
print(input_image_list.shape, input_label_list.shape)
original_images=np.squeeze(input_image_list,1)
original_labels=np.squeeze(input_label_list,1)
np.save('path/input_images', original_images)
np.save('path/input_labels', original_labels)

### Code To Calculate Mean and Standard Deviation for a given Image dataset, serialized with a Dataloader
#Recommended transforms for input images before calculating Mean and SD
""" transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor()]) """

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
   
