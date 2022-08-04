from modelMLP import MLP
from dataset import MeeroRoomsDatasetNPY

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn

from tqdm import trange
import numpy as np

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3

# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():

    # Model instantiation.
    mlp = MLP(input_dim=1001, output_dim=11)

    # Dataset building.

    dataset = MeeroRoomsDatasetNPY(
        np_X_path="/data2/datasets/x.npy",
        np_Y_path="/data2/datasets/y.npy",
        test_size=0.2,
    )

    train_sampler = SubsetRandomSampler(dataset.train_idx)
    test_sampler = SubsetRandomSampler(dataset.test_idx)

    train_loader = DataLoader(dataset, batch_size=128, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=128, sampler=test_sampler)

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
    pbar = trange(EPOCHS, desc="Epochs")
    for _ in pbar:
        mlp.train()

        for data in train_loader:

            x, ytrue = data

            optimizer.zero_grad()

            # Perform forward pass
            ypred = mlp(x)

            # Compute loss
            loss = loss_function(ypred, ytrue.squeeze(1))

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

        L = []
        mlp.eval()
        for data_val in test_loader:

            x, ytrue = data_val

            # Perform forward pass
            ypred = mlp(x)
            _, predicted_targets = torch.max(ypred, 1)

            L.append((predicted_targets == ytrue.squeeze(1)).sum().item() / 128.0)

        val_acc = np.mean(L)

        # Print statistics
        pbar.set_postfix({"Train loss": loss.item(), "Val. acc": val_acc})

    # Process is complete.
    print("Training process has finished.")


if __name__ == "__main__":
    train()
