import numpy as np
from sklearn.linear_model import orthogonal_mp_gram


class KSVD:

    def __init__(self, k_atoms, num_iterations, tolerance):
        self.k_atoms = k_atoms
        self.num_iter = num_iterations if num_iterations > 0 else 1
        self.tolerance = tolerance

    #Y (n_features, n_targets)
    def run(self, Y):
        n_features = Y.shape[0]

        D = self.__initialize_D(n_features)
        for i in range(self.num_iter):
            X = self.sparse_coding(D, Y)

            if np.linalg.norm(Y - D.dot(X)) < self.tolerance:
                return D, X

            D = self.update_dictionary(Y, D, X)


        return D, X


    def sparse_coding(self, D, Y):
        return orthogonal_mp_gram(D.T.dot(D), D.T.dot(Y))

    def update_dictionary(self, Y, D, X):
        for k in range(self.k_atoms):
            idx = X[k, :] > 0
            if sum(idx) == 0:
                continue
            D[:, k] = 0
            E = Y[:, idx] - D.dot(X[:, idx])
            U, S, Vh = np.linalg.svd(E)
            D[:, k] = U[0]
            X[k, idx] = S[0] * Vh.T[0]

        return D

    def __initialize_D(self, n_features):
        D = np.random.randn(n_features, self.k_atoms)
        return D
