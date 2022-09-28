import numpy as np 
import wandb

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0.1):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class LogsTrain():
    def __init__(self,run_name,dict_config):
        
        self.config = dict_config
        self.run = wandb.init(project = 'DeepLearning_CourseIntro', 
                              entity = 'gaetanlandreau',
                              config = self.config)
        self.run.name = run_name
        
    def get_run(self):
        return self.run
