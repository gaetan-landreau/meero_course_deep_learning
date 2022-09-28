import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

from torchsummary import summary


class RoomsModel(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(RoomsModel, self).__init__()

        efficientnetb0 = models.efficientnet_b3(pretrained=True)

        self.backbone = efficientnetb0.features  # backbone

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(in_features=1536, out_features=512)
        #self.fc1 = nn.Linear(1280, 200)
        self.fc2 = nn.Linear(512, num_classes)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        # Froze the backbone's weights if pretrained.
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.backbone(x)
        out = self.adaptive_pool(out)
        out = self.dropout(out).squeeze()
        out = self.fc1(out)
        out = self.relu(out)
        logits = self.fc2(out)
        return logits
   
    def infer(self, x):
        out = self.forward(x)
        out = self.softmax(out) # To get probabilities. 
        return out
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.zeros((1, 3, 300, 300)).to(device)
    model = RoomsModel(pretrained=True, num_classes=11).to(device)

    summary(model, (3, 224, 224))
