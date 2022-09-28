import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

import torchvision.transforms as T


def create_loaders(dataset, bs, split):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

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


def transform_fn(image_size, pretrained) -> dict:

    transform = {
        "train": T.Compose(
            [
                T.ToPILImage(),  # required for torchvision.transforms
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalize_transform(pretrained)
            ]
        ),
        "test": T.Compose(
            [
                T.ToPILImage(),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                normalize_transform(pretrained)
            ]
        ),
        "visualize": T.Compose(
                [T.ToPILImage(),]
        )
    }

    return transform

def normalize_transform(pretrained):

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if pretrained else \
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    
    return normalize

if __name__ == "__main__":
    from data.dataset import MeeroRoomsDataset

    dataset = MeeroRoomsDataset(indir="/data2/datasets/sourceImgs/all")
    split = {"val": 0.2}
    bs = 16

    train_loader, validation_loader = create_loaders(dataset, bs, split)
