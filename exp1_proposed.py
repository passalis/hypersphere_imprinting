import torch.optim as optim
from neural_utils import train_model_hypersphere, save_model, load_model, evaluate_model, extract_representation
from dataset_loaders import mnist_loaders_unimodal
from neural_utils import load_model, evaluate_model, extract_representation
from model import MNIST_HWI_NET
import numpy as np
import pickle
import torch


def train_base_model(radius=1, min_dist=2, std=0.1):
    """
    Trains a  HWI model
    :return:
    """
    (train_loader, test_loader), _, _, _ = mnist_loaders_unimodal()
    net = MNIST_HWI_NET(num_classes=5)
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_model_hypersphere(net, optimizer, train_loader, epochs=20, min_dist=min_dist, radius=radius, std=std)
    save_model(net, output_file='models/hwi_cnn_' + str(radius) + '_' + str(min_dist) + '_' + str(std) + '.model')


def evaluate_one_shot_imprinting(n_samples=[1, 2, 5], radius=1, min_dist=2, std=1, seed=0):
    net = MNIST_HWI_NET(num_classes=5)
    net.cuda()
    load_model(net, input_file='models/hwi_cnn_' + str(radius) + '_' + str(min_dist) + '_' + str(std) + '.model')

    # Extend the weights to handle a total of 10 classes (maximum)
    W_original = net.centers.data.cpu().numpy()
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

        net.centers.data = torch.from_numpy(np.float32(W)).cuda()

        # Evaluate the model
        acc4 = evaluate_model(net, oneshot_test_loader_all)

        all_acc.append(acc4)

        print(acc4)

    print("Results all_acc:", all_acc)

    with open("results/hwi_" + str(radius) + '_' + str(min_dist) + '_' + str(std) + '_' + str(seed) + ".pickle",
              "wb") as f:
        pickle.dump([all_acc], f, protocol=pickle.HIGHEST_PROTOCOL)


def print_imprinting_results(radius=1, min_dist=2, std=0.1, repeats=5):
    all_acc_euclidean = []

    for i in range(repeats):
        with open("results/hwi_" + str(radius) + '_' + str(min_dist) + '_' + str(std) + '_' + str(i) + ".pickle",
                  "rb") as f:
            [a] = pickle.load(f)
            all_acc_euclidean.append(np.asarray(a))

    all_acc_euclidean = np.asarray(all_acc_euclidean)

    print("results/hwi_" + str(radius) + '_' + str(min_dist) + '_' + str(std) + '_' + '*' + ".pickle")

    print("All categories: ")
    res = [' ${%3.2f} \pm {%3.2f}$ &' % (100 * mean, 100 * std) for mean, std in
           zip(np.mean(all_acc_euclidean, axis=0), np.std(all_acc_euclidean, axis=0))]

    line = 'Imprinting  &' + ''.join(res)[:-2] + ' \\\\'
    print(line)


# Step 1: Train the models
train_base_model(min_dist=10, radius=0, std=0)
train_base_model(min_dist=10, radius=5, std=0.05)

# Step 2: Imprint the weights and evaluate
for i in range(5):
    evaluate_one_shot_imprinting(min_dist=10, radius=0, std=0, seed=i)
    evaluate_one_shot_imprinting(min_dist=10, radius=5, std=0.05, seed=i)


# Step 3: Print the evaluation results
print_imprinting_results(min_dist=10, radius=0, std=0,)
print_imprinting_results(min_dist=10, radius=5, std=0.05)

