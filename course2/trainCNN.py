
from dataset import MeeroRoomsDataset
from modelCNN import RoomsModel

import torch 
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import Adam
from torch import nn
from torchmetrics.classification import Accuracy

import numpy as np
import yaml
from tqdm import trange 

def train(config):
    ##################
    ## Main parameters
    ##################
    num_epochs = config["hyperparameters"]["epochs"]
    bs = config["hyperparameters"]["batch_size"]
    learning_rate = config['hyperparameters']['learning_rate']
   
    indir = config["data"]["indir"]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Define your execution device
    
    train_accuracy = Accuracy().to(device)
    val_accuracy = Accuracy().to(device)
    
    ################################
    # Data loading and preprocessing
    ################################

    dataset = MeeroRoomsDataset(indir)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(np.floor(config["split"]["val"] * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=bs, sampler=train_sampler
    )

    validation_loader = DataLoader(
        dataset, batch_size=bs, sampler=valid_sampler
    )

    #########################################
    ## Model instantiation - Loss / Optimizer
    #########################################
    
    model = RoomsModel()
    model.to(device)
    model.train()
    
    lossCE = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    
    pbar = trange(num_epochs, desc="Epochs")
    for _ in pbar:
        # Train:   
        for batch_index, (imgs, labels) in enumerate(train_loader):
            
           
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            # predict classes using images from the training set
            preds = model(imgs)
            # compute the loss based on model output and real labels
            loss = lossCE(preds, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
          
            train_batch_acc = train_accuracy(preds, labels.type(torch.int8))
           
        # Validation stage.  
        model.eval()
        for batch_index, (imgs, labels) in enumerate(validation_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # predict classes using images from the training set
            preds = model(imgs)
            val_batch_acc = val_accuracy(preds, labels.type(torch.int8))
            
        train_acc = train_accuracy.compute()
        val_acc = val_accuracy.compute().cpu().numpy()
        
        pbar.set_postfix({"Train loss": loss.item(), "Val. acc": val_acc})
        model.train()
        
        train_accuracy.reset()
        val_accuracy.reset()

            
if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))["train"]

    train(config)
