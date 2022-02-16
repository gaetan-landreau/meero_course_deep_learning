import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    This class defines our Multi Layer Perceptron. 
    The one we implement here as a single Hidden layer
    and perform binary classification: 
     --> Is the input feature feed to the Neural Network correspond to an Indoor or Outdoor image ? 
    """
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__()

        self.fc_in = nn.Linear(input_dim, 500)
        self.fc1 = nn.Linear(500, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc_out = nn.Linear(20, output_dim)
        
        self.relu = nn.ReLU()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor : 

        out = self.fc_in(x)
        out = self.relu(out)
        
        out = self.fc1(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        y_pred = self.fc_out(out)
        
        return y_pred 

if __name__=='__main__': 
    
    ourMLP = MLP(input_dim= 1000,output_dim=2)
    
    x = torch.rand(16,1000)
    y = ourMLP(x)