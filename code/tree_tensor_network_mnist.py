#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Mon May 15 11:12:43 2017"""
import numpy as np
import tncontract as tn
from itertools import product
from numpy import linalg as la
from sklearn import preprocessing


class TreeTensorNetwork(object):

    def __init__(self, data, labels, bond_data, bond_inner, bond_label, layer_units):
        self.flag_contract = np.zeros((5, 8, 8), dtype=np.float64)
        # initialize contraction results
        self.layer_units = layer_units

        # self.contracted = [[]] * len(self.layer_units)

        self.contracted = []

        self.contracted.append(data)
        for i in range(1, 5):
            self.contracted.append(
                [[0 for _ in range(self.layer_units[i])] for _ in range(self.layer_units[i])])
        self.tn_layers = [0]

        self.bond_inner = bond_inner
        self.bond_label = bond_label
        self.bond_data = bond_data
        self.labels = labels
        self.n_train = data[0][0].shape[1]

        for i in range(1, 5):
            self.tn_layers.append([])
            for j in range(self.layer_units[i]):
                self.tn_layers[i].append([])
                for k in range(self.layer_units[i]):
                    if i == 1:
                        temp = np.random.random(
                            (self.bond_data, self.bond_data, self.bond_data, self.bond_data, self.bond_inner))
                    elif i == 4:
                        temp = np.random.random(
                            (self.bond_inner, self.bond_inner, self.bond_inner, self.bond_inner, self.bond_label))
                    else:
                        temp = np.random.random(
                            (self.bond_inner, self.bond_inner, self.bond_inner, self.bond_inner, self.bond_inner))
                    self.tn_layers[i][j].append(
                        tn.Tensor(temp, labels=["1", "2", "3", "4", "up"]))
        # return self.tn_layers

    def contract_local(self, tensor1, tensor2, tensor3, tensor4, Num):
        bond = tensor1.shape[0]
        if len(tensor1.shape) == 2 and len(tensor2.shape) == 2 and len(tensor3.shape) == 2 and len(tensor4.shape) == 2:
            tensor_result = tn.random_tensor(bond, bond, bond, bond, Num, labels=[
                                             'a', 'b', 'c', 'd', 'down'])

            for i, j, k, l in product(range(bond), range(bond), range(bond), range(bond)):
                tensor_result.data[i, j, k, l, :] = tensor1.data[i, :] * tensor2.data[j, :] * tensor3.data[k,
                                                                                                           :] * tensor4.data[l, :]

        else:
            tensor_result = tn.random_tensor(bond, bond, bond, bond, self.bond_inner, Num, labels=[
                                             'a', 'b', 'c', 'd', 'e', 'down'])

            if len(tensor1.shape) == 3:
                for i, j, k, l, n in product(range(bond), range(bond), range(bond), range(bond),
                                             range(self.bond_inner)):
                    tensor_result.data[i, j, k, l, n, :] = tensor1.data[i, n, :] * tensor2.data[j, :] * tensor3.data[k,
                                                                                                                     :] * \
                        tensor4.data[l, :]
            if len(tensor2.shape) == 3:
                for i, j, k, l, n in product(range(bond), range(bond), range(bond), range(bond),
                                             range(self.bond_inner)):
                    tensor_result.data[i, j, k, l, n, :] = tensor1.data[i, :] * tensor2.data[j, n, :] * tensor3.data[k,
                                                                                                                     :] * \
                        tensor4.data[l, :]
            if len(tensor3.shape) == 3:
                for i, j, k, l, n in product(range(bond), range(bond), range(bond), range(bond),
                                             range(self.bond_inner)):
                    tensor_result.data[i, j, k, l, n, :] = tensor1.data[i, :] * tensor2.data[j, :] * tensor3.data[k, n,
                                                                                                                  :] * \
                        tensor4.data[l, :]
            if len(tensor4.shape) == 3:
                for i, j, k, l, n in product(range(bond), range(bond), range(bond), range(bond),
                                             range(self.bond_inner)):
                    tensor_result.data[i, j, k, l, n, :] = tensor1.data[i, :] * tensor2.data[j, :] * tensor3.data[k,
                                                                                                                  :] * \
                        tensor4.data[l, n, :]
        return tensor_result

    def contract_local3(self, tensor1, tensor2, tensor3, Num):
        bond = tensor1.shape[0]
        tensor_result = tn.random_tensor(
            bond, bond, bond, Num, labels=['a', 'b', 'c', 'down'])

        for i, j, k in product(range(bond), range(bond), range(bond)):
            tensor_result.data[i, j, k, :] = tensor1.data[i,
                                                          :] * tensor2.data[j, :] * tensor3.data[k, :]
        return tensor_result

    def contract_unit(self, tensor0, tensor1, tensor2, tensor3, tensor4, Num):
        temp = self.contract_local(tensor1, tensor2, tensor3, tensor4, Num)
        tensor_result = tn.contract(
            tensor0, temp, ["1", "2", "3", "4"], ["a", "b", "c", "d"])

        if len(tensor_result.shape) == 2:
            tensor_result.data = preprocessing.normalize(
                tensor_result.data, axis=0, norm='l2')  # normalization
        else:
            for i in range(tensor_result.shape[1]):     # normalization
                tensor_result.data[:, i, :] = preprocessing.normalize(
                    tensor_result.data[:, i, :], axis=0, norm='l2')
        return tensor_result

    def contract_special(self, tensor0, tensor1, lab1, tensor2, lab2, tensor3, lab3, Num):
        temp = self.contract_local3(tensor1, tensor2, tensor3, Num)
        tensor_result = tn.contract(
            tensor0, temp, [lab1, lab2, lab3], ["a", "b", "c"])
        tensor_result.data = tensor_result.data.transpose(1, 0, 2)
        tensor_result.labels[0], tensor_result.labels[1] = tensor_result.labels[1], tensor_result.labels[0]

        for i in range(tensor_result.shape[1]):  # normalization
            tensor_result.data[:, i, :] = preprocessing.normalize(
                tensor_result.data[:, i, :], axis=0, norm='l2')
        return tensor_result

    def update_singletensor(self, c_i, c_j, c_k):

        path_len = 5 - c_i
        path = [[c_i, c_j, c_k]]
        tem_c_j = c_j
        tem_c_k = c_k
        for i in range(1, path_len):
            tem_c_j = tem_c_j // 2
            tem_c_k = tem_c_k // 2
            path.append([c_i + i, tem_c_j, tem_c_k])

        for i in range(1, 5):
            if i == c_i:
                for j, k in product(range(self.layer_units[i]), range(self.layer_units[i])):
                    if (self.flag_contract[i, j, k] == 0) and ((j != c_j) or (k != c_k)):
                        self.contracted[i][j][k] = self.contract_unit(self.tn_layers[i][j][k], self.contracted[i - 1][2 * j][2 * k], self.contracted[(
                            i - 1)][2 * j][2 * k + 1], self.contracted[i - 1][2 * j + 1][2 * k], self.contracted[i - 1][2 * j + 1][2 * k + 1], self.n_train)
                        self.flag_contract[i, j, k] = 1
                        if i < 4:
                            self.flag_contract[i + 1, j // 2, k // 2] = 0
                self.contracted[c_i][c_j][c_k] = self.contract_local(self.contracted[c_i - 1][2 * c_j][2 * c_k], self.contracted[(
                    c_i - 1)][2 * c_j][2 * c_k + 1], self.contracted[c_i - 1][2 * c_j + 1][2 * c_k], self.contracted[c_i - 1][2 * c_j + 1][2 * c_k + 1], self.n_train)
                self.flag_contract[c_i, c_j, c_k] = 0
                if i < 4:
                    self.flag_contract[c_i + 1, c_j // 2, c_k // 2] = 0
            else:
                for j, k in product(range(self.layer_units[i]), range(self.layer_units[i])):
                    if self.flag_contract[i, j, k] == 0:
                        if ([i, j, k] in path) and ((i - 1) == c_i):
                            if (c_j % 2 == 0) and (c_k % 2 == 0):
                                [lab1, lab2, lab3] = ["2", "3", "4"]
                                tensor1 = self.contracted[c_i][c_j][c_k + 1]
                                tensor2 = self.contracted[c_i][c_j + 1][c_k]
                                tensor3 = self.contracted[c_i][c_j + 1][c_k + 1]

                            if (c_j % 2 == 0) and (c_k % 2 == 1):
                                [lab1, lab2, lab3] = ["1", "3", "4"]
                                tensor1 = self.contracted[c_i][c_j][c_k - 1]
                                tensor2 = self.contracted[c_i][c_j + 1][c_k - 1]
                                tensor3 = self.contracted[c_i][c_j + 1][c_k]

                            if (c_j % 2 == 1) and (c_k % 2 == 0):
                                [lab1, lab2, lab3] = ["1", "2", "4"]
                                tensor1 = self.contracted[c_i][c_j - 1][c_k]
                                tensor2 = self.contracted[c_i][c_j - 1][c_k + 1]
                                tensor3 = self.contracted[c_i][c_j][c_k + 1]

                            if (c_j % 2 == 1) and (c_k % 2 == 1):
                                [lab1, lab2, lab3] = ["1", "2", "3"]
                                tensor1 = self.contracted[c_i][c_j - 1][c_k - 1]
                                tensor2 = self.contracted[c_i][c_j - 1][c_k]
                                tensor3 = self.contracted[c_i][c_j][c_k - 1]

                            self.contracted[i][j][k] = self.contract_special(
                                self.tn_layers[i][j][k], tensor1, lab1, tensor2, lab2, tensor3, lab3, self.n_train)
                            self.flag_contract[i, j, k] = 0
                            if i < 4:
                                self.flag_contract[i + 1, j // 2, k // 2] = 0

                        else:
                            # print(i,j,k)
                            self.contracted[i][j][k] = self.contract_unit(self.tn_layers[i][j][k], self.contracted[i - 1][2 * j][2 * k], self.contracted[
                                i - 1][2 * j][2 * k + 1], self.contracted[i - 1][2 * j + 1][2 * k], self.contracted[i - 1][2 * j + 1][2 * k + 1], self.n_train)
                            if ([i, j, k] in path):
                                self.flag_contract[i, j, k] = 0
                            else:
                                self.flag_contract[i, j, k] = 1
                            if i < 4:
                                self.flag_contract[i + 1, j // 2, k // 2] = 0
        if c_i != 4:

            bond = self.contracted[c_i][c_j][c_k].shape[0]
            tensor_environment = tn.random_tensor(
                bond, bond, bond, bond, self.bond_inner, labels=['e1', 'e2', 'e3', 'e4', 'eup'])
            for i, j, k, l, m in product(range(bond), range(bond), range(bond), range(bond), range(self.bond_inner)):
                sum1 = sum(self.contracted[c_i][c_j][c_k].data[i, j, k, l, g] * self.contracted[4][0][0].data[f, m, g] * self.labels.data[g, f]
                           for f in range(self.bond_label) for g in range(self.n_train))
                tensor_environment.data[i, j, k, l, m] = sum1

        else:
            tensor_environment = tn.contract(
                self.contracted[4][0][0], self.labels, "down", "up")

        if c_i == 1:
            matrix = np.reshape(tensor_environment.data, (self.bond_data *
                                                          self.bond_data * self.bond_data * self.bond_data, self.bond_inner))
            u, sigma, vt = la.svd(matrix, 0)
            self.tn_layers[c_i][c_j][c_k].data = np.reshape(
                np.dot(u, vt), (self.bond_data, self.bond_data, self.bond_data, self.bond_data, self.bond_inner))
        else:
            if c_i == 4:
                matrix = np.reshape(tensor_environment.data, (self.bond_inner *
                                                              self.bond_inner * self.bond_inner * self.bond_inner, self.bond_label))
                u, sigma, vt = la.svd(matrix, 0)
                self.tn_layers[c_i][c_j][c_k].data = np.reshape(
                    np.dot(u, vt), (self.bond_inner, self.bond_inner, self.bond_inner, self.bond_inner, self.bond_label))
            else:
                matrix = np.reshape(tensor_environment.data, (self.bond_inner *
                                                              self.bond_inner * self.bond_inner * self.bond_inner, self.bond_inner))
                u, sigma, vt = la.svd(matrix, 0)
                self.tn_layers[c_i][c_j][c_k].data = np.reshape(
                    np.dot(u, vt), (self.bond_inner, self.bond_inner, self.bond_inner, self.bond_inner, self.bond_inner))

        # compute the training accuracy-------------------------------------------
        j = c_j
        k = c_k
        for i in range(c_i, 5):
            self.contracted[i][j][k] = self.contract_unit(self.tn_layers[i][j][k],
                                                          self.contracted[i -
                                                                          1][2 * j][2 * k],
                                                          self.contracted[i -
                                                                          1][2 * j][2 * k + 1],
                                                          self.contracted[i -
                                                                          1][2 * j + 1][2 * k],
                                                          self.contracted[i - 1][2 * j + 1][2 * k + 1], self.n_train)
            j = j // 2
            k = k // 2

        temp = tn.contract(self.contracted[4][0][0], self.labels, "up", "down")
        temp.trace("up", "down")
        acc = temp.data / self.n_train
        return acc

    def train(self, n_epochs):
        acc_train = 0
        for t in range(1, n_epochs + 1):
            print("epochs:", t)
            for i in range(1, 5):
                #print("layer:", i)
                for j in range(self.layer_units[i]):
                    # print([j])
                    for k in range(self.layer_units[i]):
                        # print(i,j,k)
                        acc_train = self.update_singletensor(i, j, k)
                        acc_train = round(acc_train, 3)
            print("training average inner product:", acc_train)
        return acc_train

    def test(self, test_tensor, label_test_tensor):
        Num = test_tensor[0][0].shape[1]
        for j, k in product(range(8), range(8)):
            self.contracted[1][j][k] = self.contract_unit(self.tn_layers[1][j][k],
                                                          test_tensor[2 *
                                                                      j][2 * k],
                                                          test_tensor[2 *
                                                                      j][2 * k + 1],
                                                          test_tensor[2 *
                                                                      j + 1][2 * k],
                                                          test_tensor[2 * j + 1][2 * k + 1], Num)
        for i in range(2, 5):
            for j in (range(self.layer_units[i])):
                for k in (range(self.layer_units[i])):
                    self.contracted[i][j][k] = self.contract_unit(self.tn_layers[i][j][k],
                                                                  self.contracted[i -
                                                                                  1][2 * j][2 * k],
                                                                  self.contracted[i - 1][2 * j][
                        2 * k + 1],
                        self.contracted[i - 1][2 * j + 1][
                        2 * k],
                        self.contracted[i - 1][2 * j + 1][
                        2 * k + 1], Num)

        # option 1
        temp = tn.contract(
            self.contracted[4][0][0], label_test_tensor, "up", "down")
        temp.trace("up", "down")
        acc1 = temp.data / Num

        # option 2
        count = 0
        for i in range(Num):
            x = np.argmax(self.contracted[4][0][0].data[:, i])
            for j in range(2):
                if j == x:
                    self.contracted[4][0][0].data[j, i] = 1
                else:
                    self.contracted[4][0][0].data[j, i] = 0

            if (self.contracted[4][0][0].data[:, i] == label_test_tensor.data[i, :]).all():
                count = count + 1

        acc2 = count / Num

        return acc1, acc2

    def outputvalue(self, test_tensor, Num):
        for j, k in product(range(8), range(8)):
            self.contracted[1][j][k] = self.contract_unit(self.tn_layers[1][j][k],
                                                          test_tensor[2 *
                                                                      j][2 * k],
                                                          test_tensor[2 *
                                                                      j][2 * k + 1],
                                                          test_tensor[2 *
                                                                      j + 1][2 * k],
                                                          test_tensor[2 * j + 1][2 * k + 1], Num)
        for i in range(2, 5):
            for j in (range(self.layer_units[i])):
                for k in (range(self.layer_units[i])):
                    self.contracted[i][j][k] = self.contract_unit(self.tn_layers[i][j][k],
                                                                  self.contracted[i -
                                                                                  1][2 * j][2 * k],
                                                                  self.contracted[i - 1][2 * j][
                                                                      2 * k + 1],
                                                                  self.contracted[i - 1][2 * j + 1][
                                                                      2 * k],
                                                                  self.contracted[i - 1][2 * j + 1][
                                                                      2 * k + 1], Num)

        return self.contracted[4][0][0].data[0, :]
