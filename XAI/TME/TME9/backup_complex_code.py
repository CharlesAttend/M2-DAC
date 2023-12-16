from sklearn import metrics


class FCM:
    def __init__(
        self,
        n_clusters=2,
        init=None,
        metric_name="euclidean",
        n_init=1,
        max_iter=100,
        tol=0.0001,
        random_state=0,
        fuzzifier=2,
    ):
        """
        https://fda.readthedocs.io/en05:40 PM05:40 PM/latest/modules/ml/autosummary/skfda.ml.clustering.FuzzyCMeans.html
        """
        if random_state is not None:
            self.rng = np.random.RandomState(seed=random_state)
        self.init = init
        self.metric_name = metric_name
        self.n_init = n_init
        self.n_clusters = n_clusters  # c, number of clusters
        self.fuzzifier = fuzzifier  # m, desired fuzzy degree
        self.max_iter = max_iter
        self.tol = tol  # tolerance

    def _distance(self, data, centers):
        """Calcule la distance euclidienne."""
        return metrics.pairwise_distances(data, centers, metric=self.metric_name)

    def predict(self, X):
        ...

    def epoch(self):
        ...

    def _update_attribution(self, distances):
        distances_to_centers_raised = distances ** (2 / (self.fuzzifier - 1))

        membership_matrix[:, :] = distances_to_centers_raised / np.sum(
            distances_to_centers_raised,
            axis=1,
            keepdims=True,
        )

        # inf / inf divisions should be 1 in this context
        membership_matrix[np.isnan(membership_matrix)] = 1

        return
        ...

    def _update_centroids(self, X, membership, centroids):
        r"""

        X : La matrice des données, $x_i, i = 1...n$.
        centroids : Les centres des clusters, $w_{r}, r = 1...k$.
        membership : La matrice d'affectation, $u_{ir}$.


        \[
            w_r = \frac{\sum_{i=1}^{n} u_{ir}^{m}x_i}{\sum_{i=1}^{n}u_{ir}^{m}}
        \]
        """
        # i la donnée
        # r le cluster auquel on fait référence
        if self.metric_name == "euclidean":
            membership_raised = np.power(membership, self.fuzzifier)

            centroids = np.einsum(
                "ir,i...->r...",
                membership_raised,
                X,
            ) / membership_raised.sum(axis=0)

            return centroids
        else:
            raise ValueError(f"metric {self.metric_name} is not implemented.")

    def fit(self, X):
        # Setup u0
        n = X.shape[1]
        centers = self.rng.rand(self.n_clusters, n)
        # u0 /= np.ones(
        #     (self.n_clusters, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        # u = np.fmax(u0, np.finfo(np.float64).eps)

        # Initialize loop parameters
        jm = np.zeros(0)
        p = 0

        # Main cmeans loop
        for _ in range(self.max_iter):
            u_new = u.copy()
            # Attributions update
            self._update_attribution()
            # Center update
            [cntr, u, Jjm, d] = epoch(X, u2, c, m)
            jm = np.hstack((jm, Jjm))
            p += 1

            # Stopping rule
            if np.linalg.norm(u - u2) < self.tol:
                break

        # Final calculations
        error = np.linalg.norm(u - u2)
        fpc = _fp_coeff(u)

    return cntr, u, u0, d, jm, p, fpc


membership_matrix[:, :] = distances_to_centers_raised / np.sum(
    distances_to_centers_raised,
    axis=1,
    keepdims=True,
)

membership = np.random.randn(5, 3)
membership[:, :]
