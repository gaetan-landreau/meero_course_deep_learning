import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

from torchsummary import summary

class RoomsModel(nn.Module):
    def __init__(self,fine_tune, num_classes):
        super(RoomsModel, self).__init__()
        
        efficientnetb0 = models.efficientnet_b0(pretrained=True)
        
        self.backbone =efficientnetb0.features # backbone
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(1280,200)
        self.fc2 = nn.Linear(200,num_classes)
        
        self.softmax = nn.Softmax(dim=1)
        
        # Keep the weights from the backbone frozen if fine_tune is True. 
        if fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
 
    def forward(self, x):
        out = self.backbone(x)
        out = self.adaptive_pool(out)
        out = self.dropout(out).squeeze()
        out = self.fc1(out)
        out = self.fc2(out)
        pred = self.softmax(out)
        return pred
    
if __name__ =='__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.zeros((1,3,224,224)).to(device)
    model = RoomsModel().to(device)
    #out = model(x)
    
    
    summary(model, (3, 224,224))
    
    
    