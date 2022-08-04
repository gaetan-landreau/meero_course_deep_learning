import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import torch.nn.functional as F

import numpy as np
import os
import cv2
import tqdm
from PIL import Image


class MeeroRoomsDatasetNPY(Dataset):
    def __init__(self, np_X_path, np_Y_path, test_size):
        """
        The class OurMeeroRoomsDataset is responsible for creating
        the batch of data that are going to be feed to our neural network.

        Args:
            np_file_path ([str]): Path toward the .npz file that contain both our SIFT descriptor and labels.
            test_size ([float]): Proportion of data kept for testing the neural network performance.
        """

        dataX = np.load(np_X_path)
        labelY = np.load(np_Y_path)

        self.X = dataX
        self.Y = np.expand_dims(labelY, -1)

        self.train_idx, self.test_idx = self.get_train_test_split_idx(test_size)

    def get_train_test_split_idx(self, test_size):
        nb_samples = self.__len__()
        indices = list(range(nb_samples))
        np.random.shuffle(indices)
        split = int(np.floor(test_size * nb_samples))

        return indices[split:], indices[:split]

    def __getitem__(self, index):

        x = torch.from_numpy(self.X[index, :].astype(np.float32))
        y = torch.from_numpy(self.Y[index].astype(np.int64))

        return x, y

    def __len__(self):
        return self.X.shape[0]


class MeeroRoomsDataset(Dataset):
    """
    Main DataLoader class to perform rooms classification.
    For speed up conveniance, the whole dataset is stored through .npy files, that are
    build if they do not exist at the given root path (indir).

    Images are resized to (456,456) and randomly horizontally flipped for data augmentation.

    Labels are one-hot encoded.
    """

    def __init__(self, indir: str):
        # Input directory where all the images are stored.
        self.indir = indir
        # List the different rooms in the input directory.
        self.rooms = self.get_rooms_list()

        # Pre-load all the images and their corresponding labels.
        npy_files_exist = os.path.exists(self.indir + "/X.npy") and os.path.exists(
            self.indir + "/Y.npy"
        )
        self.X, self.Y = (
            self.load_dataset() if npy_files_exist else self.build_dataset()
        )

    def load_dataset(self):
        print(f"--> Loading dataset from existing .npy files ...")
        X = np.load(self.indir + "/X.npy", allow_pickle=True)
        Y = np.load(self.indir + "/Y.npy", allow_pickle=True)
        return X, Y

    def build_dataset(self):
        X = []
        Y = []
        print(f"--> Building .npy files from rooms contained in {self.indir} ...")
        for room in tqdm.tqdm(self.rooms):
            print(f"--> Processing room {room} ...")
            room_path = os.path.join(self.indir, room)
            imgs = [
                f
                for f in os.listdir(room_path)
                if os.path.isfile(os.path.join(room_path, f))
            ]

            label = self.rooms.index(room)
            label = np.array(label)

            for img in imgs:

                img = cv2.imread(os.path.join(room_path, img))
                img = img[:, :, ::-1]

                X.append(img)
                Y.append(label)

        # Save both x and y as .npy files
        np.save(self.indir + "/X.npy", X, allow_pickle=True)
        np.save(self.indir + "/Y.npy", Y, allow_pickle=True)

        return X, Y

    def get_rooms_list(self) -> list:
        self.rooms_list = sorted(
            [
                dir
                for dir in os.listdir(self.indir)
                if os.path.isdir(os.path.join(self.indir, dir))
            ]
        )
        return self.rooms_list

    @staticmethod
    def transform(img: np.array) -> torch.Tensor:
        # Convert the np.array image to a PIL one. (required for torchvision.transforms)
        img = Image.fromarray(img)

        transform = T.Compose(
            [
                T.Resize((456, 456)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        return transform(img)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index):
        img, label = self.X[index], self.Y[index]

        # Convert to tensor and perform data augmentation.
        label = F.one_hot(
            torch.from_numpy(np.array(label)), num_classes=len(self.rooms)
        ).float()

        img = MeeroRoomsDataset.transform(img)

        return img, label


if __name__ == "__main__":

    dataset = MeeroRoomsDataset(indir="/data2/datasets/sourceImgs/all")
