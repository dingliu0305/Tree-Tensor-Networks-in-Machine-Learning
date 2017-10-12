#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
data_folder = "./data/mnist/"

n_epochs = 3

bond_data = 2
bond_inner = 3
bond_label = 2

n_class = 2
n_train_each = 900
n_train = n_train_each * n_class

n_test_each = 1000
n_test = n_test_each * 10

layer_units = [16, 8, 4, 2, 1]
train_tn = False

LAB1 = [0] * n_train_each
LAB2 = [1] * n_train_each
LAB = np.concatenate((LAB1, LAB2), axis=0)

# -------------------------------------------------------------------------
# load training data
print("loading data")
input = open(data_folder + 'tsne.pkl', 'rb')
ttn = pickle.load(input)


#%%
LAB1 = [0] * n_train_each
LAB2 = [1] * n_train_each
LAB = np.concatenate((LAB1, LAB2), axis=0)


def squash_layer(contraction, which_layer):
    layer = []
    for row in contraction[which_layer]:
        layer += [element.data for element in row]
    return np.vstack(layer).T


def tsne(contraction, which_layer, per, lear, tags=LAB):
    selection = np.random.choice(n_train, size=n_train, replace=False)
    mf = TSNE(n_components=2, perplexity=per,
              learning_rate=lear, init='pca', n_iter=1200)
    M = squash_layer(contraction, which_layer)
    x = M[selection]
    x_embed = mf.fit_transform(x)
    TAGS = []
    TAGS = tags[selection]
    return x_embed, TAGS


def sort(contraction, which_layer, per, lear, tags=LAB):
    x_embed, TAGS = tsne(contraction, which_layer, per, lear, tags=LAB)
    CATS = []
    DOGS = []

    for i in range(len(TAGS)):
        if TAGS[i] == 0:
            CATS.append(x_embed[i])
        if TAGS[i] == 1:
            DOGS.append(x_embed[i])

    result = np.concatenate((CATS, DOGS), axis=0)

    return result


#%%
def plot(contraction,  which_layer, per, lear, tags=LAB):

    result = sort(contraction, which_layer, per, lear, tags=LAB)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x = result[:, 0]
    y = result[:, 1]
    ax1.scatter(x[0:n_train_each], y[0:n_train_each], s=11,
                c='b', marker="o", label='Planes', alpha=0.5)
    ax1.scatter(x[n_train_each + 1:n_train], y[n_train_each + 1:n_train],
                s=11, c='r', marker="o", label='Horses', alpha=0.5)
    plt.legend(loc='upper right')
    plt.axis('off')
    # plt.show()
    pp = PdfPages('%s_P%s_L%s.pdf' % (which_layer, per, lear))
    pp.savefig(fig)
    pp.close()

    return fig


#%%
def sweep(contraction, which_layer, per, tags=LAB):

    L = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
    f = []
    for i in range(0, len(L)):
        f = plot(contraction, i, per, L[i], tags=LAB)

    return f
#%%


def sweep2(contraction, which_layer, tags=LAB):
    G = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    q = []
    for i in range(0, len(G)):
        q = sweep(contraction, which_layer, G[i], tags=LAB)

    return q
#%%


def sweep3(contraction, per, lear, tags=LAB):

    m = []
    for i in range(1, 5):
        m = sweep2(contraction, i, tags=LAB)

    return m


for i in range(5):
    plot(ttn.contracted, i, 60, 400)

plt.show()
