import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MNIST_NET(nn.Module):

    def __init__(self, num_classes=10, initial_c=1):
        super(MNIST_NET, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.fc1 = nn.Linear(in_features=1600, out_features=256)

        self.fc2 = nn.Linear(in_features=256, out_features=num_classes, bias=False)
        self.c = nn.Parameter(data=torch.from_numpy(np.float32([initial_c])))

        self.drop_layer = nn.Dropout(p=0.5)
        self.eps = 1e-7

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        out = self.drop_layer(out)
        out = F.relu(self.fc1(out))

        # Get the embeddings
        out = out / (torch.sqrt(torch.sum(out ** 2, dim=1, keepdim=True)) + self.eps)

        # Get the normalized prototypes
        W = self.fc2.weight

        # Calculate the cosine similarity
        out = self.c * torch.matmul(out, W.t())

        return out

    def get_features(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        out = self.drop_layer(out)
        out = F.relu(self.fc1(out))

        out = out / (torch.sqrt(torch.sum(out ** 2, dim=1, keepdim=True)) + self.eps)

        return out

    def enforce_constraints(self):
        W = self.fc2.weight.data
        W = W / (torch.sqrt(torch.sum(W ** 2, dim=1, keepdim=True)) + self.eps)
        self.fc2.weight.data = W


def sym_distance_matrix(A, B, eps=1e-10, self_similarity=False):
    """
    """

    # Compute the squared distances
    AA = torch.sum(A * A, 1).view(-1, 1)
    BB = torch.sum(B * B, 1).view(1, -1)
    AB = torch.mm(A, B.transpose(0, 1))
    D = AA + BB - 2 * AB

    # Zero the diagonial
    if self_similarity:
        D = D.view(-1)
        D[::B.size(0) + 1] = 0
        D = D.view(A.size(0), B.size(0))

    # Return the square root
    D = torch.sqrt(torch.clamp(D, min=eps))

    return D


class MNIST_HWI_NET(nn.Module):

    def __init__(self, num_classes=10, out=256):
        super(MNIST_HWI_NET, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.fc1 = nn.Linear(in_features=1600, out_features=out)

        # Class centers
        W = np.random.randn(num_classes, out)
        self.centers = nn.Parameter(torch.from_numpy(np.float32(W)))

        self.drop_layer = nn.Dropout(p=0.5)
        self.eps = 1e-7

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        out = self.drop_layer(out)
        out = F.relu(self.fc1(out))

        # Calculate distance to centers
        dists = sym_distance_matrix(out, self.centers)
        out = 1. / (1 + dists)

        return out

    def get_features(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        out = self.drop_layer(out)
        out = F.relu(self.fc1(out))

        return out
