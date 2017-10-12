#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Mon May 15 11:09:35 2017"""
import numpy as np
import tncontract as tn
import scipy.io as sio
from itertools import product
from itertools import combinations


def load_train_data(filename, n_train, n_train_each, bond_label, bond_data, current_class):
    image = sio.loadmat(filename)
    image_group = image['Data_group']
    data_group = np.zeros(
        (16, 16, 10, image_group.shape[3], bond_data), dtype=np.float64)

    for i in range(bond_data):
        data_group[:, :, :, :, i] = (len([c for c in combinations(range(bond_data - 1), i)]) ** 0.5) * \
            np.cos((image_group) * (np.pi / 2)) ** (bond_data - (i + 1)) * np.sin(
            (image_group) * (np.pi / 2)) ** i

    train_data = np.zeros((16, 16, bond_data, n_train), dtype=np.float64)
    label_data = np.zeros((n_train, bond_label), dtype=np.float64)

    for k, m in product(range(bond_data), range(n_train_each)):
        train_data[:, :, k, m] = data_group[:, :, current_class, m, k]

    cc = set([current_class])
    rest = set(range(0, 10)) - cc

    for k, l, m in product(range(bond_data), range(9), range(int(n_train_each / 9))):
        train_data[:, :, k, n_train_each + l *
                   int(n_train_each / 9) + m] = data_group[:, :, list(rest)[l], m, k]

    label_data[0:n_train_each] = [1, 0]
    label_data[n_train_each:n_train] = [0, 1]

    data_tensor = [[0 for col in range(16)] for row in range(16)]
    for i, j in product(range(16), range(16)):
        data_tensor[i][j] = tn.Tensor(
            train_data[i, j, :, :], labels=["up", "down"])

    label_tensor = tn.Tensor(label_data, labels=["up", "down"])
    return data_tensor, label_tensor


def load_test_data(filename, n_test, n_test_each, bond_data):
    image = sio.loadmat(filename)
    image_group = image['Data_test_group']
    data_group = np.zeros(
        (16, 16, 10, image_group.shape[3], bond_data), dtype=np.float64)

    for i in range(bond_data):
        data_group[:, :, :, :, i] = (len([c for c in combinations(range(bond_data - 1), i)]) ** 0.5) * \
            np.cos((image_group) * (np.pi / 2)) ** (bond_data - (i + 1)) * np.sin(
            (image_group) * (np.pi / 2)) ** i

    test_data = np.zeros((16, 16, bond_data, n_test), dtype=np.float64)

    for k, l, m in product(range(bond_data), range(10), range(n_test_each)):
        test_data[:, :, k, l * n_test_each + m] = data_group[:, :, l, m, k]

    test_tensor = [[0 for col in range(16)] for row in range(16)]
    for i, j in product(range(16), range(16)):
        test_tensor[i][j] = tn.Tensor(
            test_data[i, j, :, :], labels=["up", "down"])

    return test_tensor
