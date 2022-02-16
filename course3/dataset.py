import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F
import numpy as np

class OurMeeroRoomsDataset(Dataset):
    def __init__(self, np_X_path,np_Y_path,test_size):
        """
        The class OurMeeroRoomsDataset is responsible for creating 
        the batch of data that are going to be feed to our neural network.
        
        Args:
            np_file_path ([str]): Path toward the .npz file that contain both our SIFT descriptor and labels.
            test_size ([float]): Proportion of data kept for testing the neural network performance. 
        """
        
        dataX = np.load(np_X_path)
        labelY = np.load(np_Y_path)
        
        self.X = dataX
        self.Y = np.expand_dims(labelY, -1)
        
        self.train_idx, self.test_idx = self.get_train_test_split_idx(test_size)
        
    def get_train_test_split_idx(self,test_size):
        nb_samples = self.__len__()
        indices = list(range(nb_samples))
        np.random.shuffle(indices)
        split = int(np.floor(test_size * nb_samples))
        
        return  indices[split:], indices[:split]
        
    def __getitem__(self, index):
       
        x = torch.from_numpy(self.X[index,:].astype(np.float32))
        y = torch.from_numpy(self.Y[index].astype(np.int64)) 
        
        return x,y
    
    def __len__(self):
        return self.X.shape[0]
    
if __name__== '__main__': 
   
    dataset = OurMeeroRoomsDataset(np_X_path = '/data2/datasets/x.npy',
                                   np_Y_path= '/data2/datasets/y.npy',
                                   test_size = 0.2)
   
    train_sampler = SubsetRandomSampler(dataset.train_idx)
    test_sampler = SubsetRandomSampler(dataset.test_idx)

    train_loader = DataLoader(dataset, batch_size = 32, sampler=train_sampler)
    test_loader  = DataLoader(dataset, batch_size = 32, sampler=test_sampler)
    
    for (x,y) in train_loader: 
        print(x.shape,y.shape)