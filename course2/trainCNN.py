
from data.dataset import MeeroRoomsDataset
from models.cnn import RoomsModel

from utils.utils_data import create_loaders
from utils.utils_metrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

import torch 
from torch.optim import Adam
from torch import nn


import numpy as np
import yaml
from tqdm import trange 

def train_epoch():
    return 
def train(config):
    ##################
    ## Main parameters
    ##################
    num_epochs = config["hyperparameters"]["epochs"]
    bs = config["hyperparameters"]["batch_size"]
    learning_rate = config['hyperparameters']['learning_rate']
   
    nb_classes= config["model"]["num_classes"]
    fine_tune = config["model"]["fine_tunning"]
    
    indir = config["data"]["indir"]
    outdir = config["data"]["outdir"]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Define your execution device
    
    train_accuracy = Accuracy().to(device)
    val_accuracy = Accuracy().to(device)
    
    ############################
    # Dataset creation / loading
    ############################
    dataset = MeeroRoomsDataset(indir)
    train_loader, validation_loader = create_loaders(dataset, bs, split= config['split'])
    
    ######################
    ## Model instantiation
    ######################
    model = RoomsModel(fine_tune=fine_tune, num_classes=nb_classes).to(device)
    model.train()
    
    ###################
    ## Loss / optimizer
    ###################
    lossCE = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    
    ################
    ## Training loop
    ################
    pbar = trange(num_epochs, desc="Epochs")
    for _ in pbar:
        # Train stage
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
