import numpy as np


class FuzzyCMeans:
    def __init__(self, n_clusters=2, max_iter=200, m=2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.centroids_ = None
        self.membership_weights_ = None

    def fit(self, df):
        # Initialize variables
        n, d = df.shape
        self.membership_weights_ = self._initializeMembershipWeights(n, self.n_clusters)

        for _ in range(self.max_iter):
            self.centroids_ = self._computeCentroids(df, self.membership_weights_, d)
            self.membership_weights_ = self._updateWeights(
                df, self.membership_weights_, self.centroids_
            )

    def fit_predict_proba(self, df):
        self.fit(df)
        return self.predict_proba(df)

    def fit_predict(self, df):
        self.fit(df)
        return self.predict(df)

    def predict(self, df):
        return self.predict_proba(df).argmax(axis=1)

    def predict_proba(self, df):
        # Check if the model is fitted
        if self.centroids_ is None:
            raise Exception("Model is not fitted yet.")

        # Initialize variables
        n, _ = df.shape
        membership_weights_ = self._initializeMembershipWeights(n, self.n_clusters)

        return self._updateWeights(df, membership_weights_, self.centroids_)

    def _initializeMembershipWeights(self, n, k):
        weight = np.random.dirichlet(np.ones(k), n)
        return np.array(weight)

    def _computeCentroids(self, df, weight_arr, d):
        C = []
        for i in range(self.n_clusters):
            weight_sum = np.power(weight_arr[:, i], self.m).sum()
            Cj = [
                (df.iloc[:, x].values * np.power(weight_arr[:, i], self.m)).sum()
                / weight_sum
                for x in range(d)
            ]
            C.append(Cj)
        return np.array(C)

    def _updateWeights(self, df, weight_arr, C):
        n = df.shape[0]
        denom = np.zeros(n)
        for i in range(self.n_clusters):
            dist = np.sqrt(((df - C[i]) ** 2).sum(axis=1))
            denom += np.power(1 / dist, 1 / (self.m - 1))

        for i in range(self.n_clusters):
            dist = np.sqrt(((df - C[i]) ** 2).sum(axis=1))
            weight_arr[:, i] = np.power(1 / dist, 1 / (self.m - 1)) / denom
        return weight_arr


class PossibilisticCMeans(FuzzyCMeans):
    def __init__(self, n_clusters=2, max_iter=200, m=2, eta=None):
        super().__init__(n_clusters, max_iter, m)
        if not eta:
            eta = np.ones((n_clusters,)) - 0.5
        self.eta = eta

    def _updateWeights(self, df, weight_arr, C):
        X = df.values
        for i in range(self.n_clusters):
            dist = np.linalg.norm(X - C[i], axis=1)
            weight_arr[:, i] = 1 + np.power(dist / self.eta[i], 2 / (self.m - 1))
        return weight_arr
