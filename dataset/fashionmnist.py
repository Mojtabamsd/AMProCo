import gzip
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from PIL import Image


def load_idxfile(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels


class FashionMNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None, train=True):
        self.images = load_idxfile(images_path)
        self.labels = load_labels(labels_path)
        self.transform = transform
        self.num_classes = 10
        self.train = train

        # Initialize class_data and cls_num_list similar to other datasets
        self.class_data = [[] for _ in range(self.num_classes)]
        for i, label in enumerate(self.labels):
            self.class_data[label].append(i)

        self.cls_num_list = [len(self.class_data[i]) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]

        image = Image.fromarray(image, mode='L')
        label = torch.tensor(label, dtype=torch.int64)

        if self.transform is not None:
            if self.train:
                sample1 = self.transform[0](image)
                sample2 = self.transform[1](image)
                sample3 = self.transform[2](image)
                return [sample1, sample2, sample3], label
            else:
                return self.transform(image), label
        else:

            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            return image, label
