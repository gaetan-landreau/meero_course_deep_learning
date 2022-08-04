import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

class RoomsModel(nn.Module):
    def __init__(self):
        super(RoomsModel, self).__init__()
        efficientnetb0 = models.efficientnet_b0(pretrained=True)
        
        self.backbone =efficientnetb0.features # backbone
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(1280,200)
        self.fc2 = nn.Linear(200,11)
        
        # Keep the weights from the backbone frozen.
        for param in self.backbone.parameters():
            param.requires_grad = False
 
    def forward(self, x):
        out = self.backbone(x)
        out = self.adaptive_pool(out)
        out = self.dropout(out).squeeze()
        out = self.fc1(out)
        pred = self.fc2(out)
     
        return pred
    
if __name__ =='__main__':
    x = torch.zeros((1,3,224,224))
    model = RoomsModel()
    out = model(x)
    print(out.shape)
#model = efficientnet.efficientnet_v2_s()#EfficientNet_V2_S
#model = torch.hub.load("pytorch/vision", "efficientnet_v2_s", weights=".IMAGENET1K_V1")

