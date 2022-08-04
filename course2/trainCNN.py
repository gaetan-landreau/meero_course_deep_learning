from dataset import MeeroRoomsDatasetNPY
from torch.utils.data import DataLoader, SubsetRandomSampler

import numpy as np
import yaml


def train(config):

    ################################
    # Data loading and preprocessing
    ################################

    dataset = MeeroRoomsDatasetNPY(config["data"]["indir"])

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(np.floor(config["split"]["val"] * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=config["hyperparams"]["batch_size"], sampler=train_sampler
    )

    validation_loader = DataLoader(
        dataset, batch_size=config["hyperparams"]["batch_size"], sampler=valid_sampler
    )


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))["train"]

    train(config)
