import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skfda
import pandas as pd
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_blobs
from matplotlib.colors import LogNorm


def generate_blobs(
    n_samples=500,
    n_clusters=3,
    cluster_std=1.0,
    n_outliers=50,
    random_seed=42,
):
    """Génère des données artificielles de type "blobs".

    :param n_samples: Nombre d'échantillons, 500 par défaut.
    :param n_clusters: Nombre de clusters, 3 par défaut.
    :param cluster_std: Ecart-type des clusters, 1.0 par défaut.
    :param n_outliers: Nombre d'outliers, 50 par défaut.
    :param random_seed: Graine aléatoire, 42 par défaut.
    :return: Un ensemble de train, de test et d'outliers.
    """
    X, _, centers = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_seed,
        return_centers=True,
    )

    X_outliers = generate_outliers(
        X, centers, n_outliers=n_outliers, random_seed=random_seed
    )

    return X, X_outliers


def generate_outliers(X, centers, n_outliers=50, min_distance=4, random_seed=42):
    """Génère des outliers, c'est-à-dire des points qui sont en dehors des clusters
    générés. Le plus simple serait de générer aléatoirement des outliers sur l'ensemble
    de la grille, mais cela résulterait en des outliers faisant potentiellement partie
    de clusters, biaisant alors notre détection.

    :param X: Clusters.
    :param centers: Centres des clusters.
    :param n_outliers: Nombre d'outliers, defaults to 50
    :param min_distance: Distance euclidienneminimale entre les clusters et les outliers,
    4 est une bonne valeur minimisant le nombre d'outliers pouvant apparaître dans les
    clusters tout en leur laissant la possibilité d'être proche.
    :param random_seed: Graine aléatoire pour la reproductibilité.
    :return: Un ensemble d'outliers.
    """
    rng = np.random.RandomState(random_seed)
    # Générer des outliers
    X_outliers = []
    while len(X_outliers) < n_outliers:
        outlier_candidate = rng.uniform(
            X.min(axis=0) - 10.0, X.max(axis=0) + 10.0, X.shape[1]
        )
        # Est-ce que le candidat est un outlier, c.-à-d. ne fait-il pas partie d'un cluster ?
        if np.all(
            np.min(np.linalg.norm(centers - outlier_candidate, axis=1)) >= min_distance
        ):
            X_outliers.append(outlier_candidate)
    return np.array(X_outliers)


def plot_data(X_train, X_test, X_outliers):
    """Affiche les données artificiellement générées."""
    plt.figure(figsize=(10, 6))
    # Plot les données d'entraînement
    plt.scatter(
        X_train[:, 0], X_train[:, 1], c="blue", label="Training Data", edgecolors="k"
    )
    # Plot les données de test
    plt.scatter(
        X_test[:, 0], X_test[:, 1], c="green", label="Test Data", edgecolors="k"
    )
    # Plot les outliers
    plt.scatter(
        X_outliers[:, 0], X_outliers[:, 1], c="red", label="Outliers", edgecolors="k"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_elongated(
    n_samples=1000,
    n_clusters=2,
    centers=[(-10, -10), (10, 10)],
    cov=[np.array([[5, 0], [0, 1]]), np.array([[1, 0], [0, 5]])],
    n_outliers=50,
    random_seed=42,
):
    """Génère des données artificielles "allongées", c.à-d. des distributions
    normales multidimensionnelles.

    :param n_samples: Nombre d'échantillons, 1000 par défaut.
    :param n_clusters: Nombre de clusters, 2 par défaut.
    :param centers: Les centres pour chaque cluster.
    :param cov: Les matrices de covariance pour chaque cluster.
    :param n_outliers: Nombre d'outliers, 50 par défaut.
    :param random_seed: Graine aléatoire, 42 par défaut.
    :return: Un ensemble de train, de test et d'outliers.
    """
    assert len(centers) == n_clusters, ValueError(
        f"Number of centers {len(centers)} must match number of clusters {n_clusters}"
    )
    assert len(cov) == n_clusters, ValueError(
        f"Number of covariance matrixes {len(centers)} must match number of clusters {n_clusters}"
    )

    rng = np.random.RandomState(random_seed)

    # Génération de distributions normales multidimensionnelles
    X = np.empty((0, 2))

    for cluster in range(n_clusters):
        X_cluster = rng.multivariate_normal(
            centers[cluster], cov[cluster], n_samples // n_clusters
        )
        X = np.vstack([X, X_cluster])

    # Générer des outliers
    X_outliers = generate_outliers(
        X, centers, n_outliers=n_outliers, random_seed=random_seed
    )

    return X, X_outliers


def plot_data_with_label(X, labels_true=None, cluster_centers=None, *args, **kwargs):
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1]})
    ax = kwargs.get("ax", None)
    if not ax:
        ax = plt.gca()
    ax.set_xticks(())
    ax.set_yticks(())
    if kwargs.get("title", None):
        ax.set_title(kwargs["title"])
        kwargs.pop("title")
    if labels_true is not None:
        kwargs["hue"] = "Attribued Cluster"
        df["Attribued Cluster"] = labels_true
    sns.scatterplot(data=df, x="x", y="y", palette="deep", marker=".", **kwargs)

    if cluster_centers is not None:
        kwargs["legend"] = False
        kwargs.pop("hue")
        d = {
            "x": cluster_centers[:, 0],
            "y": cluster_centers[:, 1],
            "Clusters": list(range(len(cluster_centers))),
        }
        sns.scatterplot(
            data=d,
            x="x",
            y="y",
            hue="Clusters",
            palette="deep",
            marker="o",
            s=75,
            **kwargs,
        )


def experim(X, n_clusters, labels_true=None, centers=None, nrow=4, ncol=2, scale=1):
    """
    J'aurai bien fait un truc générique mais chaque algo à sa petite particularité
    j'suis assez triste
    """
    fig = plt.figure(figsize=(3 * nrow * scale, 8 * ncol * scale))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = sns.color_palette("deep", n_colors=n_clusters)

    # Fit estimators
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    gmm.fit(X)

    ac = AgglomerativeClustering(n_clusters=n_clusters)
    sc = SpectralClustering(n_clusters=n_clusters)
    dbscan = DBSCAN().fit(X)

    grid_points = [0, 1]
    fd = skfda.FDataGrid(X, grid_points)
    fuzzy_cmeans = skfda.ml.clustering.FuzzyCMeans(
        n_clusters=n_clusters, random_state=0
    )
    fuzzy_cmeans.fit(fd)

    k_means = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
    k_means.fit(X)

    # Plotting
    ## Y_true
    ax = fig.add_subplot(nrow, ncol, 1)
    plot_data_with_label(
        X, labels_true, centers, title="Ground truth", ax=ax, legend=True
    )

    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 100)
    X_mesh, Y_mesh = np.meshgrid(x, y)
    XX = np.array([X_mesh.ravel(), Y_mesh.ravel()]).T

    ## KMeans
    ax = fig.add_subplot(nrow, ncol, 2)
    plot_data_with_label(
        X,
        k_means.predict(X),
        k_means.cluster_centers_,
        title="KMeans",
        ax=ax,
        legend=False,
    )
    Z = k_means.predict(XX)
    Z = Z.reshape(X_mesh.shape)
    ax.contour(X_mesh, Y_mesh, Z)

    ## Fuzzy C-Mean
    ax = fig.add_subplot(nrow, ncol, 3)
    fcm_cluster_centers = fuzzy_cmeans.cluster_centers_.data_matrix.squeeze()
    fcm_membership_degree = fuzzy_cmeans.membership_degree_  ## comme un predict proba
    fcm_labels = fcm_membership_degree.argmax(axis=1)
    plot_data_with_label(
        X,
        fcm_labels,
        fcm_cluster_centers,
        fcm_membership_degree,
        title="Fuzzy C-Means",
        ax=ax,
        legend=False,
    )

    fd = skfda.FDataGrid(XX, grid_points)
    Z = fuzzy_cmeans.predict_proba(fd)
    for i in range(n_clusters):
        colors_list = [(*colors[i], 0.1), colors[i]]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors_list)
        ax.contour(X_mesh, Y_mesh, Z[:, i].reshape(X_mesh.shape), cmap=cmap)

    ## C-Mean possibiliste
    ax = fig.add_subplot(nrow, ncol, 4)
    ax.set_title("Posibilist C-Mean")

    ## GMM
    ax = fig.add_subplot(nrow, ncol, 5)
    plot_data_with_label(
        X, gmm.predict(X), gmm.means_, title="GMM", ax=ax, legend=False
    )
    Z = -gmm.score_samples(XX)
    Z = Z.reshape(X_mesh.shape)
    ax.contour(
        X_mesh,
        Y_mesh,
        Z,
        norm=LogNorm(vmin=1.0, vmax=1000.0),
        levels=np.logspace(0, 3, 10),
    )

    ## AgglomerativeClustering
    ##### PAS DE PREDICT
    ax = fig.add_subplot(nrow, ncol, 6)
    plot_data_with_label(X, ac.fit_predict(X), title="Agglomerative Clustering", ax=ax)

    # SpectralClustering
    ##### PAS DE PREDICT
    ax = fig.add_subplot(nrow, ncol, 7)
    plot_data_with_label(X, sc.fit_predict(X), title="Spectral Clustering", ax=ax)

    # DBSCAN
    ##### PAS DE PREDICT
    ax = fig.add_subplot(nrow, ncol, 8)
    plot_data_with_label(X, dbscan.fit_predict(X), title="DBSCAN", ax=ax)

    estimators = {
        "gmm": gmm,
        "ac": ac,
        "sc": sc,
        "dbscan": dbscan,
        "fuzzy_cmeans": fuzzy_cmeans,
        "k_means": k_means,
    }
    return estimators, XX, X_mesh, Y_mesh


def plot_3D(XX, X_mesh, Y_mesh, estimator, n_clusters):
    nrow = 1
    ncol = 3
    scale = 5
    colors = sns.color_palette("deep", n_colors=n_clusters)
    fig = plt.figure(figsize=(3 * nrow * scale, 8 * ncol * scale))
    fd = skfda.FDataGrid(XX, [0, 1])
    Z = estimator.predict_proba(fd)
    for i in range(n_clusters):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        colors_list = [(*colors[i], 0.1), colors[i]]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors_list)
        ax.plot_surface(X_mesh, Y_mesh, Z[:, i].reshape(X_mesh.shape), cmap=cmap)
