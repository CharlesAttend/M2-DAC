import numpy as np

# X(s,f) C(k,f) W(s,k)
import numpy as np


class PossibilisticCMeans:
    def __init__(
        self, n_clusters, max_iter=100, m=1.5, p=0.5, m2=2, epsilon=0.05, debug=False
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.p = p
        self.m2 = m2
        self.epsilon = epsilon
        self.debug = debug
        self.centroids_ = None
        self.labels_ = None

    def fit(self, X):
        # Initial random centroids
        initial_centroids = X[
            np.random.choice(X.shape[0], self.n_clusters, replace=False), :
        ]

        # Run the kmeans algorithm
        # self.labels_, self.centroids_ = cmeans(
        #     initial_centroids,
        #     X,
        #     i=self.max_iter,
        #     m=self.m,
        #     p=self.p,
        #     m2=self.m2,
        #     e=self.epsilon,
        #     debug=self.debug,
        # )
        self.labels_, self.centroids_ = cmeans(
            X,
            self.n_clusters,
            nb_iter=self.max_iter,
            m=self.m2,
        )
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_, self.centroids_


# def cmeans(c, x, i=10000, m=1.5, p=0.5, m2=2, e=0.05, debug=False):
#     flag = False
#     if debug == True:
#         print("Kmeans started with iterations=", i)
#         flag = True
#     n = compute_n(c, x, p, m2, debug=flag)
#     s = x.shape[0]
#     f = x.shape[1]
#     k = c.shape[0]
#     w = compute_weights(c.copy(), x.copy(), m, n, debug=flag)
#     pc = c
#     while i > 0:
#         # n=compute_n(c,x,p,m2,debug=flag)
#         c = compute_centroids(w.copy(), x.copy(), m, debug=flag)
#         w = compute_weights(c.copy(), x.copy(), m, n, debug=flag)
#         i = i - 1
#         if np.linalg.norm(pc - c) < e:
#             i = 0
#         pc = c
#         if debug == True:
#             print("Iterations left ", i)
#             print("Centroids")
#             print(c)
#             print("Weights")
#             print(w)
#     return w, c


def cmeans(X, nb_clusters, W=None, eta=None, nb_iter=100, m=2):
    """
    Possibilist c-means clustering algorithm.

    :param X: Data points, numpy array of shape (n_samples, n_features).
    :param nb_clusters: The number of clusters to form.
    :param eta: Typicality parameters for each cluster, numpy array of shape (nb_clusters,).
    :param nb_iter: Number of iterations to run.
    :param m: Fuzziness parameter.
    :return: Tuple (W, U) where W is an array of centroids and U is the matrix of typicalities.
    """
    nb_points, nb_dim = X.shape
    if not eta:
        eta = np.array([1.5 for _ in range(nb_clusters)])
    # Initialize cluster centers randomly from data points
    indices = np.random.choice(nb_points, nb_clusters, replace=False)
    if W is None:
        W = X[indices, :]
    # W = X[indices, :]

    # Initialize the matrix of typicalities
    U = np.random.rand(nb_points, nb_clusters)
    U = U / np.sum(U, axis=1, keepdims=True)

    for iteration in range(nb_iter):
        # Update typicalities U
        for i in range(nb_points):
            for j in range(nb_clusters):
                dist = np.linalg.norm(X[i] - W[j])
                U[i, j] = 1 / (1 + (dist / eta[j]) ** (2 / (m - 1)))

        # Update cluster centers W
        for j in range(nb_clusters):
            numerator = np.sum(U[:, j] ** m * X.T, axis=1)
            denominator = np.sum(U[:, j] ** m)
            W[j] = numerator / denominator

        # Optional: Implement convergence check to break the loop if clusters do not change significantly

    return W, U
