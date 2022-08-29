import numpy as np 
from torch.utils.data import DataLoader, SubsetRandomSampler

def create_loaders(dataset,bs, split):
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(np.floor(split["val"] * dataset_size))

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
    
    return train_loader, validation_loader