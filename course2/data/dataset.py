import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder


import numpy as np
import os
import cv2
import tqdm
from PIL import Image

class MeeroRoomsDataset(Dataset):
    """
    Main DataLoader class to perform rooms classification.
    For speed up conveniance, the whole dataset is stored through .npy files, that are
    build if they do not exist at the given root path (indir).

    Images are resized to (456,456) and randomly horizontally flipped for data augmentation.

    Labels are one-hot encoded.
    """

    def __init__(self, indir: str, is_train: bool, transform_fn):
        """_summary_

        Args:
            indir (str): _description_
            is_train (bool): _description_
            transform_fn (_type_): _description_
        """
        # Input directory where all the images are stored.
        self.indir = indir

        # Set the flag to know if we are training or testing.
        self.is_train = is_train

        # Transformation pipeline.
        self.transform = transform_fn

        # Get the list of rooms.
        self.rooms = self.get_rooms_list()
        
        # Load the images and their corresponding labels.
        self.imgs, self.labels = self.get_images_list()
        
    def get_rooms_list(self) -> list:
        rooms =  sorted(
            [
                dir
                for dir in os.listdir(self.indir)
                if os.path.isdir(os.path.join(self.indir, dir))
            ]
        )
        
        return rooms
        
    def get_images_list(self):
        # Build up two empty lists.
        imgs = []
        labels = []
        
        #Iterate over each room
        for room in self.rooms:
            y = self.rooms.index(room)
            
            for img in os.listdir(os.path.join(self.indir, room)):
                full_path_img = os.path.join(self.indir, room, img)
                imgs.append(full_path_img)
                labels.append(y)
                
        # Shuffle the images and labels (in the same order)
        data = list(zip(imgs, labels))
        imgs, labels = zip(*data)
        
        return imgs, labels
    
    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]
        
        # Read the image / apply the transformation pipeline.
        img = cv2.imread(img)[:,:,::-1]  # BGR to RGB
        img = self.transform(img)
        
        # Convert the [int] label into a [torch.tensor] (required for loss computation)
        label = torch.tensor(label).long()

        return img, label


if __name__ == "__main__":
    import sys 
    sys.path.append('..')
    from utils.utils_data import transform_fn
    
    transform = transform_fn()['train']
    dataset = MeeroRoomsDataset(indir="/data2/datasets/sourceImgs/all", is_train=True, transform_fn=transform)
    print(len(dataset))
    
    y = torch.tensor([4])
    #y[1] = 1.
    
    y_pred = torch.tensor([[ 0.1040,  0.0115,  0.0186,  0.0653,  0.0202,  0.0427,  0.0075,  0.0189,
          0.0579, -0.0183, -0.0208]])
    y_pred_exp = torch.exp(y_pred)
    
    y_pred_soft = torch.nn.Softmax(dim=-1)(y_pred)
    
    sum_exp = torch.sum(y_pred_exp)
    
    print(torch.log(y_pred_exp/sum_exp))
    print(torch.log(y_pred_soft))
    
    #y1 = torch.tensor([0.,0.91,0.02,0.03,0.0,0.,0.02,0.01,0.01,0.,0.]).float()
    #y1 = torch.tensor([0.,1.,0.0,0.0,0.0,0.,0.0,0.0,0.0,0.,0.]).float()

    #y2 = torch.tensor([0.,0.12,0.0,0.21,0.09,0.01,0.27,0.03,0.17,0.,0.]).float()   
    
    from torch.nn import CrossEntropyLoss
    
    l1 = CrossEntropyLoss()(y_pred, y)
    
    #l2 = CrossEntropyLoss()(y2.unsqueeze(0), y.unsqueeze(0))
    print(l1)
    #print(l1)
    #print(l2)
    
    #x,y = dataset[80]
    
    
    