import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import torchvision
from torchvision import transforms


#Pytorch dataset Class For Numpy Image arrays

class NumpyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return (image, label)
    
    
#Function to convert numpy inputs to a torch dataset
    
def numpy_to_dataset(images, labels, batch_size=None, shuffle=False, transform=None):
    # Convert numpy arrays to PyTorch tensors
    images = torch.from_numpy(images).float()
    labels = torch.from_numpy(labels).long()

    # Create dataset from tensors
    dataset = Dataset(images, labels, transform)

    if batch_size is not None:
        # Create dataloader from dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset, dataloader
    else:
        return dataset    
    
#Example Code For Converting a Numpy image array with images and labels of shape [num_images,h,w] to a torch Dataset manually
#Labels are set to torch.long() and images to torch.float()


"""with np.load('images_path', allow_pickle= True) as data:
    X = data['arr_0'][0:]
with np.load('labels_path', allow_pickle= True) as data:
    Y = data['arr_0'][0:]
y=np.reshape(Y, (Y.shape[0],Y.shape[1],Y.shape[2]))
X = np.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()
y=torch.unsqueeze(y,axis=1)
#transforms=transforms.compose[()]   (Your transforms, suggested transforms include normalization, cropping and reshaping)
torch_dataset=NumpyDataset(X,y)
torch.save(torch_dataset,'save_path')

#Load a saved dataset and serialize it as a torch_dataloader
my_data=torch.load('save_path')
my_dataloader= DataLoader(my_data, batch_size=batch_size, shuffle=True)"""
