from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch


class OneShotDataset(object):

    def __init__(self, dataset, target_labels=[0, 1, 2, 3, 4], new_labels=[0, 1, 2, 3, 4], n_data=-1, seed=1):

        np.random.seed(seed)
        self.dataset = dataset

        # Get the labels of the dataset
        dataset_labels = []
        for i in range(len(dataset)):
            _, cur_label = dataset[i]
            dataset_labels.append(cur_label)
        dataset_labels = np.asarray(dataset_labels)

        # Keep only the labels we are interested in
        idx = []
        for cur_label in target_labels:
            cur_idx = dataset_labels == cur_label
            cur_idx = np.where(cur_idx)[0]
            if n_data > 0:
                np.random.shuffle(cur_idx)
                idx.extend(cur_idx[:n_data])
            else:
                idx.extend(cur_idx)

        # Indices of samples with the correct labels
        self.idx = np.asarray(idx)

        # Data to translate the labels
        self.target_labels = np.asarray(target_labels)
        self.new_labels = np.asarray(new_labels)

    def __getitem__(self, index):

        # Get the sample from the original dataset
        idx = self.idx[index]
        data, label = self.dataset[idx]

        if type(label) is torch.Tensor:
            label = label.item()

        # Translate the label
        new_label_idx = np.where(self.target_labels == label)[0]
        label = self.new_labels[new_label_idx]

        return data, label

    def __len__(self):
        return len(self.idx)


def mnist_loaders_unimodal(n_samples=1, batch_size=128, seed=1):
    """
    Load the mnist dataset using a half-train half-test split

    :param n_samples: number of samples to be used for one shot learning
    :return:
    """

    transform = transforms.Compose([transforms.ToTensor()])

    train_classes, train_classes_targets = [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]
    oneshot_classes, oneshot_targets = [5, 6, 7, 8, 9], [5, 6, 7, 8, 9]
    oneshot_classes_all, oneshot_targets_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Datasets
    train_set = datasets.MNIST(root='../datasets/data/mnist', train=True, download=True, transform=transform)
    train_set = OneShotDataset(train_set, target_labels=train_classes, new_labels=train_classes_targets, n_data=-1,
                               seed=seed)

    test_set = datasets.MNIST(root='../datasets/data/mnist', train=False, download=True, transform=transform)
    test_set = OneShotDataset(test_set, target_labels=train_classes, new_labels=train_classes_targets, n_data=-1,
                              seed=seed)

    oneshot_train_set = datasets.MNIST(root='../datasets/data/mnist', train=True, download=True, transform=transform)
    oneshot_train_set = OneShotDataset(oneshot_train_set, target_labels=oneshot_classes, new_labels=oneshot_targets,
                                       n_data=n_samples, seed=seed)

    oneshot_test_set = datasets.MNIST(root='../datasets/data/mnist', train=False, download=True, transform=transform)
    oneshot_test_set = OneShotDataset(oneshot_test_set, target_labels=oneshot_classes, new_labels=oneshot_targets,
                                      n_data=-1, seed=seed)

    oneshot_test_set_all = datasets.MNIST(root='../datasets/data/mnist', train=False, download=True,
                                          transform=transform)
    oneshot_test_set_all = OneShotDataset(oneshot_test_set_all, target_labels=oneshot_classes_all,
                                          new_labels=oneshot_targets_all, n_data=-1, seed=seed)

    # Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=1, shuffle=True)

    oneshot_train_loader = DataLoader(oneshot_train_set, batch_size=batch_size, num_workers=1, shuffle=False)
    oneshot_test_loader = DataLoader(oneshot_test_set, batch_size=batch_size, num_workers=1, shuffle=True)
    oneshot_test_loader_all = DataLoader(oneshot_test_set_all, batch_size=batch_size, num_workers=1, shuffle=True)

    return (train_loader, test_loader), oneshot_train_loader, oneshot_test_loader, oneshot_test_loader_all
