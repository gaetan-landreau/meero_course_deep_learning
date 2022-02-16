import torch 
from dataset import OurMeeroRoomsDataset , DataLoader, SubsetRandomSampler
import torch.nn as nn
from mlp import MLP
from tqdm import tqdm 
import numpy as np 

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4

# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    
    # Model instantiation. 
    mlp = MLP(input_dim= 1001,output_dim=11)
    
    
    
    # Dataset building. 
    
    dataset = OurMeeroRoomsDataset(np_X_path = '/data2/datasets/x.npy',
                                   np_Y_path= '/data2/datasets/y.npy',
                                   test_size = 0.2)
   
    train_sampler = SubsetRandomSampler(dataset.train_idx)
    test_sampler = SubsetRandomSampler(dataset.test_idx)

    train_loader = DataLoader(dataset, batch_size =32, sampler=train_sampler)
    test_loader  = DataLoader(dataset, batch_size = 32, sampler=test_sampler)
    
    # Define the loss function and optimizer    
    loss_function = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
    
    for _ in tqdm(range(EPOCHS)):
        
        
        for data in train_loader: 
            
            x,ytrue = data
            
            optimizer.zero_grad()
            
            # Perform forward pass
            ypred = mlp(x)
        
            # Compute loss
            loss = loss_function(ypred, ytrue.squeeze(1))
      
            # Perform backward pass
            loss.backward()

            
            # Perform optimization
            optimizer.step()

        # Print statistics
        print(f'Training loss: {loss.item()}')
        

    # Process is complete.
    print('Training process has finished.')
    
if __name__ == '__main__':
    
    """import numpy as np 
    
    y = np.load('/data2/datasets/y.npy')
    print(type(y[0]))"""
    train()