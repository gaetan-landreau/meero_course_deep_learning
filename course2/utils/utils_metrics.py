from tarfile import XGLTYPE
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
)

import torch 
import numpy as np 

class classificationMetrics:
    def __init__(self, device):
        self.metrics={'accuracy':Accuracy().to(device),
                      'precision':Precision().to(device),
                      'recall':Recall().to(device),
                      'f1score':F1Score().to(device)}
                    
    def update(self, y_true, y_pred):
        for key in self.metrics:
            self.metrics[key].update(y_pred,y_true)
         
    def compute(self):
        l={}
        for key in self.metrics:
            l[key]=self.metrics[key].compute()
        return l 
    
    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()
    
    def on_cpu_numpy(self):
        for key in self.metrics:
            self.metrics[key].cpu().numpy()
        
if __name__ =='__main__':
    
    
    
    #pred = torch.Tensor([[0.95,0.05,0,0,0,0,0,0,0,0,0],[0.98,0.02,0,0,0,0,0,0,0,0,0]])
    #true = np.array([[3],[1]])
    
    #target = torch.tensor([2, 1, 2, 0], device=torch.device("cuda", 0))
    #preds = torch.tensor([2, 3, 2, 0], device=torch.device("cuda", 0))

    #print(target.shape)
    
    
    y_pred = torch.tensor([[ 0.1040,  0.0115,  0.0186,  0.0653,  0.0202,  0.0427,  0.0075,  0.0189,
          0.0579, -0.0183, -0.0208],[ 0.040,  0.115,  0.0186,  0.0653,  0.0202,  0.0427,  0.0075,  0.0189,
          0.0579, -0.0183, -0.0208],[ 0.1040,  0.0115,  0.0186,  0.0653,  0.0202,  0.0427,  0.0075,  0.0189,
          0.0579, -0.0183, 0.208]]).to(torch.device("cuda", 0))
    
    y_pred_soft = torch.nn.Softmax(dim=-1)(y_pred).to(torch.device("cuda", 0))
    
    pred = torch.argmax(y_pred_soft, dim=1).to(torch.device("cuda", 0))
    target = torch.tensor([0, 1, 2], device=torch.device("cuda", 0))
    
    metrics = classificationMetrics(device = 'cuda:0')
    
    metrics.update(target, pred)
    res_metrics = metrics.compute()
    
    #print(res_metrics)
    
    
    acc = Accuracy().to('cuda:0')
    
    res1 = acc(pred, target)
    res2=acc(y_pred, target)

    print(res1)
    print(res2)
    exit()
    y_pred = torch.tensor([[ 0.040,  0.1115,  0.0186,  0.0653,  0.0202,  0.0427,  0.0075,  0.0189,
          0.0579, -0.0183, -0.0208],[ 0.040,  0.115,  0.0186,  0.0653,  0.0202,  0.0427,  0.0075,  0.0189,
          0.0579, -0.0183, -0.0208],[ 0.1040,  0.1115,  0.0186,  0.0653,  0.0202,  0.0427,  0.0075,  0.0189,
          0.0579, -0.0183, 0.98]])
    
    y_pred_soft = torch.nn.Softmax(dim=-1)(y_pred)
    
    pred = torch.argmax(y_pred_soft, dim=1).to(torch.device("cuda", 0))
    target = torch.tensor([1, 1, 10], device=torch.device("cuda", 0))
    
    metrics.update(target, pred)
    res_metrics = metrics.compute()
    print(res_metrics)
    
    #metrics.update(true,pred)
    
    #metrics_dict = metrics.compute()