from dataset_loaders import mnist_loaders_unimodal
import torch.nn as nn
import torch.optim as optim
from neural_utils import train_model, save_model, evaluate_model, load_model, extract_representation
from model import MNIST_NET
import torch
import pickle
import numpy as np


def train_imprinting_model(initial_c=10):
    """
    Trains the baseline models
    :return:
    """
    (train_loader, test_loader), _, _, _ = mnist_loaders_unimodal()
    net = MNIST_NET(num_classes=5, initial_c=initial_c)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_model(net, optimizer, criterion, train_loader, epochs=20)

    save_model(net, output_file='models/base_cnn_imprinting.model')

    acc = evaluate_model(net, test_loader)
    print("Acc = ", 100 * acc)


def evaluate_one_shot_imprinting(n_samples=[1, 2, 5], seed=1):
    net = MNIST_NET(num_classes=5)
    net.cuda()
    load_model(net, input_file='models/base_cnn_imprinting.model')

    # Extend the weights to handle a total of 10 classes (maximum)
    W_original = net.fc2.weight.data.cpu().numpy()
    W = np.zeros((10, W_original.shape[1]))
    W[:5, :] = W_original

    all_acc = []

    for n in n_samples:
        (train_loader, test_loader), oneshot_train_loader, oneshot_test_loader, oneshot_test_loader_all = \
            mnist_loaders_unimodal(n_samples=n, seed=seed)

        # Calculate the centroids
        train_data_one_shot, train_labels_one_shot = extract_representation(net, oneshot_train_loader)

        # Imprint the weights
        unique_labels = np.unique(train_labels_one_shot)
        for i, cur_lab in enumerate(unique_labels):
            idx = (train_labels_one_shot == cur_lab)
            mean_vec = np.mean(train_data_one_shot[idx, :], axis=0)
            W[5 + i, :] = mean_vec
        net.fc2.weight.data = torch.from_numpy(np.float32(W)).cuda()
        net.enforce_constraints()

        # Evaluate the model
        all_acc.append(evaluate_model(net, oneshot_test_loader_all))

    print("Results all_acc:", all_acc)

    with open("results/imprinting_results_" + str(seed) + ".pickle", "wb") as f:
        pickle.dump([all_acc], f, protocol=pickle.HIGHEST_PROTOCOL)


def print_imprinting_results(repeats=5):
    all_acc_euclidean = []

    for i in range(repeats):
        with open("results/imprinting_results_" + str(i) + ".pickle", "rb") as f:
            [a] = pickle.load(f)
            all_acc_euclidean.append(np.asarray(a))
    all_acc_euclidean = np.asarray(all_acc_euclidean)

    res = [' ${%3.2f} \pm {%3.2f}$ &' % (100 * mean, 100 * std) for mean, std in
           zip(np.mean(all_acc_euclidean, axis=0), np.std(all_acc_euclidean, axis=0))]
    line = 'Imprinting  &' + ''.join(res)[:-2] + ' \\\\'
    print(line)




# Step 1: Train the model
train_imprinting_model()

# Step 2: Imprint the weights and evaluate
for i in range(5):
    evaluate_one_shot_imprinting(seed=i)

# Step 3: Print the evaluation results
print_imprinting_results()
