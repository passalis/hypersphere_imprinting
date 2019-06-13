import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch.nn import L1Loss


def save_model(net, output_file='model.state'):
    """
    Saves a pytorch model
    :param net:
    :param output_file:
    :return:
    """
    torch.save(net.state_dict(), output_file)


def load_model(net, input_file='model.state'):
    """
    Loads a pytorch model
    :param net:
    :param input_file:
    :return:
    """
    state_dict = torch.load(input_file)
    net.load_state_dict(state_dict)


def train_model(net, optimizer, criterion, train_loader, epochs=10):
    """
    Trains a pytorch model
    :param net:
    :param optimizer:
    :param criterion:
    :param train_loader:
    :param epochs:
    :return:
    """
    for epoch in range(epochs):
        net.train()

        train_loss, correct, total = 0, 0, 0
        for (inputs, targets) in tqdm(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets).view(-1)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            net.enforce_constraints()

            # Calculate statistics
            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print("\nLoss, acc = ", train_loss, 100.0 * correct / total)


def sym_distance_matrix(A, B, eps=1e-10, self_similarity=False):
    """
    Defines the symbolic matrix that contains the distances between the vectors of A and B
    :param A: the first data matrix
    :param B: the second data matrix
    :param self_similarity: zeros the diagonial to improve the stability
    :params eps: the minimum distance between two vectors (set to a very small number to improve stability)
    :return:
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


def train_model_hypersphere(net, optimizer, train_loader, radius=0, min_dist=10, repulsive_loss_weight=1, std=0.1, epochs=10):
    """
    Trains a pytorch model
    :param net:
    :param optimizer:
    :param criterion:
    :param train_loader:
    :param epochs:
    :return:
    """

    for epoch in range(epochs):
        net.train()

        train_loss, correct, total = 0, 0, 0
        for (inputs, targets) in tqdm(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets).view(-1)
            outputs = net.get_features(inputs)

            # Loss for getting the samples close to their centers
            centers = net.centers[targets]
            centers = centers + torch.randn_like(centers)*std
            sample_dists = torch.sqrt(torch.sum((centers - outputs) ** 2, dim=1))
            sample_targets = torch.rand_like(sample_dists).cuda()
            center_loss = torch.mean((sample_dists - radius*sample_targets) ** 2)

            # Loss for getting the centers apart (if needed)
            center_dists = sym_distance_matrix(centers, centers, self_similarity=True)
            center_dists = center_dists[center_dists <= min_dist]
            repulsive_loss = torch.mean((min_dist - center_dists) ** 2)

            loss = center_loss + repulsive_loss*repulsive_loss_weight
            loss.backward()
            optimizer.step()

            # Calculate statistics
            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            #

        print("\nLoss, acc = ", train_loss, 100.0 * correct / total)


def evaluate_model(net, test_loader, class_ids = None):
    net.eval()
    predicted = []
    labels = []

    with torch.no_grad():
        for (inputs, targets) in tqdm(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            preds = outputs.cpu().numpy().argmax(axis=1)
            if class_ids is not None:
                preds = [class_ids[x] for x in preds]


            predicted.append(preds)
            labels.append(targets.cpu().numpy().reshape((-1,)))
    predicted = np.concatenate(predicted).reshape((-1,))
    labels = np.concatenate(labels).reshape((-1,))
    acc = np.sum(predicted == labels) / len(labels)
    return acc


def extract_representation(net, test_loader):
    net.eval()
    representations = []
    labels = []

    with torch.no_grad():
        for (inputs, targets) in tqdm(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net.get_features(inputs)

            representations.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy().reshape((-1,)))
    representations = np.concatenate(representations, axis=0)
    labels = np.concatenate(labels).reshape((-1,))
    return representations, labels
