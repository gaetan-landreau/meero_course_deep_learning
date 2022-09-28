import yaml
import torch
from torch import nn 

from utils.utils_data import transform_fn
from data.dataset import MeeroRoomsDataset
from models.cnn import RoomsModel
from utils.utils_metrics import Accuracy

from torch.utils.data import DataLoader


def test(config):

    indir = config["data"]["indir"]
    outdir = config["data"]["outdir"]
    weight_file = config['data']['weight']
    
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # Define your execution device

    ############################
    # Dataset creation / loading
    ############################
    transform = transform_fn()
    dataset = MeeroRoomsDataset(indir, is_train=False, transform_fn=transform["test"])
    test_loader = DataLoader(dataset, batch_size=10)
    
    
    model = RoomsModel(fine_tune=True, num_classes=11).to(device)
    model.load_state_dict(torch.load(weight_file))
    model.eval()
    
    lossCE = nn.CrossEntropyLoss()
    test_accuracy = Accuracy().to(device)
    nb_false = 0 
    with torch.no_grad():
        for _, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs)
            
            
            pred =torch.argmax(preds, dim=-1)
            print(pred.shape)
            print(labels.shape)
            print(pred)
            print(labels)
            loss = lossCE(preds, labels)
            exit()
            #lab = torch.argmax(labels, dim=-1)
            #if lab !=pred:
                #print(preds.unsqueeze(0),labels.type(torch.int8))
                #nb_false+=1
            
            #test_accuracy.update(preds.unsqueeze(0), labels.type(torch.int8))
        #print(f'Nb false: {nb_false}')
        #print(f'Nb images: {len(test_loader)}')
        #print(f'Computed accuracy: {test_accuracy.compute()}')
        
if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))["test"]

    test(config)
