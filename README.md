This is the computational appendix for the following paper:

Ding Liu, Shi-Ju Ran, Peter Wittek, Cheng Peng, Raul Blázquez García, Gang Su, Maciej Lewenstein. Machine Learning by Two-Dimensional Hierarchical Tensor Networks: A Quantum Information Theoretic Perspective on Deep Architectures. *arXiv:1710.xxxxx*, 2017.

The code uses [tncontract](https://github.com/andrewdarmawan/tncontract) for tensor contractions. Other dependencies are SciPy, Matplotlib, and Scikit-learn.

The data files can be downloaded from [here]((https://cloud.icfo.es/owncloud/index.php/s/Ks0QhCYTwiqmpSC?path=%2FMNIST%20Data).).

Files
=====

- `tree_tensor_network_mnist.py`: The implementation of the tree tensor network for the MNIST dataset.

- `tsne_mnist.py`: Plotting the t-SNE embedding.

- `utilities_mnist.py`: Helper functions.

- `TTN_mnist.py`: The main file to train and test the tree tensor network on MNIST.


