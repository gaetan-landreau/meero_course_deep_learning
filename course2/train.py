from data.dataset import MeeroRoomsDataset
from models.cnn import RoomsModel

from utils.utils_data import create_loaders, transform_fn
from utils.utils_metrics import classificationMetrics
from utils.utils_train import EarlyStopping,LogsTrain


import torch
from torch.optim import Adam
from torch import nn


import numpy as np
import yaml
from tqdm import trange,tqdm
import setproctitle

# To know who is running the script (used when typing 'nvidia-smi' on a terminal)
setproctitle.setproctitle('[Gaetan - DL course - Training]')


def train_one_epoch(model, optimizer, train_loader, lossCE, device, train_metrics,logs):
    model.train()
    running_train_loss = 0.0
    with tqdm(train_loader,unit='batch') as tepoch:
        
        for idx,(imgs, labels) in enumerate(tepoch):

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # predict classes using images from the training set
            preds = model(imgs)
           
            # compute the loss based on model output and real labels
            train_loss = lossCE(preds, labels)
       
            # backpropagate the loss
            train_loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
            # update training metrics
            train_metrics.update(labels,preds)

            running_train_loss += train_loss.item()
        
            if (idx +1)%50 ==0:
                avg_training_loss = running_train_loss / 50.
                logs.log({'train_loss': avg_training_loss})
                print(f'Training Loss: {avg_training_loss}')
                running_train_loss = 0.0
            
        return avg_training_loss
    
def train(config):
    ##################
    ## Main parameters
    ##################
    num_epochs = config["hyperparameters"]["epochs"]
    bs = config["hyperparameters"]["batch_size"]
    learning_rate = config["hyperparameters"]["learning_rate"]
    img_size = config["hyperparameters"]["image_size"]
    
    nb_classes = config["model"]["num_classes"]
    is_pretrained = config["model"]["pretrained"]

    indir = config["data"]["indir"]
    outdir = config["data"]["outdir"]

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # Define your execution device

    ############################
    # Dataset creation / loading
    ############################
    transform = transform_fn(image_size = img_size, pretrained = is_pretrained)
    dataset = MeeroRoomsDataset(indir, is_train=True, transform_fn=transform["train"])
    train_loader, validation_loader = create_loaders(dataset, bs, split=config["split"])

    ######################
    ## Model instantiation
    ######################
    model = RoomsModel(pretrained=is_pretrained, num_classes=nb_classes).to(device)
    model.train()

    ###################
    ## Loss / Optimizer
    ###################
    lossCE = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    ##########
    ## Metrics
    ##########
    train_metrics = classificationMetrics(device)
    val_metrics = classificationMetrics(device)
    
    #################
    ## Early stopping
    #################
    early_stopping = EarlyStopping(tolerance=5, min_delta=1)


    ##################
    ## Logs with WandB
    ##################
    logs = LogsTrain(run_name='my_first_cnn_training',dict_config=config)
    run_logs = logs.get_run()
    
    ################
    ## Training loop
    ################
    pbar = trange(num_epochs, desc="Epochs")
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        
        # Train stage.
        avg_training_loss = train_one_epoch(model, optimizer, train_loader, lossCE, device, train_metrics,run_logs)
        
        # Validation stage.
        with torch.no_grad():
            running_val_loss = 0.0
            for imgs, labels in validation_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                # predict classes using images from the validation set
                preds = model(imgs)
                
                val_loss = lossCE(preds, labels)
                
                running_val_loss += val_loss.item()
                val_metrics.update(labels,preds)
            avg_val_loss = running_val_loss / len(validation_loader)  
            run_logs.log({'val_loss': avg_val_loss})
             
        # Early stopping criteria. 
        e_stop = EarlyStopping()
        if e_stop(validation_loss=avg_val_loss):
            print("End the training at epoch:, {epoch} because of early stopping.")
            # Save model (to update for best weights...)
            torch.save(model.state_dict(), outdir + f"final_model_epoch_{epoch}.pt")
            break
        
        train_metrics_dict = train_metrics.compute()
        val_metrics_dict = val_metrics.compute()
        
        run_logs.log(val_metrics_dict)
        run_logs.log(train_metrics_dict)
       
        train_metrics.reset()
        val_metrics.reset()

    # Save model (to update for best weights...)
    torch.save(model.state_dict(), outdir + "final_model.pt")


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))["train"]

    train(config)
