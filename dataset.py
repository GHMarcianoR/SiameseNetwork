import torch
from torch.utils.data import Dataset
import numpy as np


class SiameseDataset(Dataset):

    def __init__(self, data_set):
        self.data_set = data_set
        self.train = data_set.train
        self.transform = self.data_set.transform
        if self.train:
            self.train_labels = self.data_set.targets
            self.train_data = self.data_set.data
        else:
            self.test_labels = self.data_set.targets
            self.test_data = self.data_set.data

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def selectPair(self, label):
        equal = np.random.randint(2)

        if self.train:
            ind = np.random.randint(0, len(self.train_data))
            if equal:
                while self.train_labels[ind].item() != label:
                    ind = np.random.randint(0, len(self.train_data))
                return self.train_data[ind], 0
            else:
                if self.train_labels[ind].item() != label:
                    pass
                else:
                    while self.train_labels[ind].item() == label:
                        ind = np.random.randint(0, len(self.train_data))

                return self.train_data[ind], 1
        else:
            ind = np.random.randint(0, len(self.test_data))
            if equal:
                while self.test_labels[ind].item() != label:
                    ind = np.random.randint(0, len(self.test_data))
                return self.test_data[ind], 0
            else:
                if self.test_labels[ind].item() != label:
                    pass
                else:
                    while self.test_labels[ind].item() == label:
                        ind = np.random.randint(0, len(self.test_data))
                return self.test_data[ind], 1

    def __getitem__(self, item):
        if self.train:
            img = self.train_data[item]
            label = self.train_labels[item]
        else:
            img = self.test_data[item]
            label = self.test_labels[item]
        img2, y = self.selectPair(label.item())

        return [torch.unsqueeze(img, 0).float(),
                torch.unsqueeze(img2, 0).float(),
                y,
                label]
