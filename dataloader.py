import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random
import time


class MnistBags(Dataset):

    def __init__(self, target = 1, train = True, mean_bag_length = 10, total_bags = 1000, seed = 7):
        self.target = target
        self.mean_bag_length = mean_bag_length
        self.train = train
        self.random = np.random.RandomState(seed)
        self.total_bags = total_bags
        self.bags, self.bag_labels = self._form_bags()

    def _form_bags(self):
        tic = time.time()
        data_loader = get_data_loader(25, train = self.train)
        valid_bags = 0
        bags = []
        bag_labels = []
        previous_label = 0

        # Creates a balanced dataset with alternating labels
        while valid_bags < self.total_bags:
            for numbers, labels in data_loader:
                bag_length = max(1, self.random.normal(self.mean_bag_length, 2, 1).astype(int))
                indices = torch.LongTensor(self.random.randint(0, len(labels), bag_length))
                labels_in_bag = labels[indices]
                label = 1 if any(labels_in_bag == self.target) else 0

                if label == previous_label:
                    continue
                else:
                    bags.append(numbers[indices])
                    bag_labels.append(label)
                    previous_label = label
                    valid_bags += 1
                    if valid_bags == self.total_bags:
                        break

        # Need to randomize the orderings
        temp = list(zip(bags, bag_labels))
        random.shuffle(temp)
        bags, bag_labels = zip(*temp)

        print("Took {:0.3f}s to create {:s}".format(time.time() - tic, 'train set' if self.train else 'test set'))

        return bags, bag_labels

    def __len__(self):
        return len(self.bag_labels)

    def __getitem__(self, index):
        bag = self.bags[index]
        label = self.bag_labels[index]

        return bag, label

def get_data_loader(batch_size, train = True):

    # Convert the image into float tensor in [0, 1], mean-std normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    # Create train data loader
    if train:
        data_loader = DataLoader(datasets.MNIST(download = True,
                                                root = "../datasets",
                                                transform = data_transform,
                                                train = True),
                                 batch_size = batch_size,
                                 shuffle = True)

    # Create test data loader
    else:
        data_loader = DataLoader(datasets.MNIST(download = True,
                                                root = "../datasets",
                                                transform = data_transform,
                                                train = False),
                                 batch_size = batch_size,
                                 shuffle = True)

    return data_loader
