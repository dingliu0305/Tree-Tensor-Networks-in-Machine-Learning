from tree_tensor_network_mnist import TreeTensorNetwork
from utilities_mnist import load_train_data
from utilities_mnist import load_test_data
from itertools import product
import numpy as np

import pickle

data_folder = "/export/tree_tn/data/mnist/"
n_epochs = 3

bond_data=[2,2,2,2,2,2,2,2,2,2]
bond_inner=[2,2,2,2,2,2,2,2,2,2]
bond_label = 2

n_class = 2
n_train_single=10
n_train_each = n_train_single*9
n_train = n_train_each * n_class

n_test_each = 800
n_test = n_test_each * 10

layer_units = [16, 8, 4, 2, 1]



# build tensor network---------------------------------------------------
print("building tensor network")
output = open('tsne.pkl', 'wb')

ttn=[0 for col in range(10)]
acc_train=[0 for col in range(10)]
acc_test1=[0 for col in range(10)]
acc_test2=[0 for col in range(10)]

for i in [0]: # 0,1,2...9 corresponding to each digit
    print("bond_data:",bond_data[i])
    print("bond_inner:", bond_inner[i])
    data, labels = load_train_data(data_folder + "train.mat", n_train, n_train_each, bond_label, bond_data[i], i)
    data_test, labels_test = load_train_data(data_folder + "test.mat", n_test_each*2, n_test_each, bond_label, bond_data[i], i)

    ttn[i] = TreeTensorNetwork(data, labels, bond_data[i], bond_inner[i], bond_label, layer_units)

    # Training
    print("Training number", i, "th binary classifier:")
    acc_train[i] = ttn[i].train(n_epochs)
    print("training inner product:",acc_train[i])

    pickle.dump(ttn, output)
output.close()


