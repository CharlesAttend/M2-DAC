import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_moons

""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Données                                                                │
    └────────────────────────────────────────────────────────────────────────┘
 """

datasets = []
for noise in np.arange(0, 0.5, 0.1):
    datasets.append(make_moons(noise=noise, random_state=0))

""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Classifiers                                                            │
    └────────────────────────────────────────────────────────────────────────┘
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay


def plot_boundaries(X, y, ax, clf):
    """"Plot the data and the decision boundary resulting from a classifier."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    DecisionBoundaryDisplay.from_estimator(clf, X, ax=ax, eps=0.5)
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())


def plot_obs_and_enemy(obs, enemy, ax, colors=["red", "orange"]):
    """
    Plot the observation to interprete and the enemy returned by the growing sphere
    generation algorithm.
    """
    ax.scatter(*enemy, c=colors[0])
    ax.scatter(*obs, c=colors[1])


names = [
    "Nearest Neighbors",
    "Random Forest",
    "Linear SVM",
]
classifiers = [
    KNeighborsClassifier(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    SVC(),
]
figure = plt.figure(figsize=(27, 9))
i = 1
for ds_cnt, (X, y) in enumerate(datasets):
    i += 1
    for name, clf in zip(names, classifiers):
        clf.fit(X, y)
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title(name)
        plot_boundaries(X, y, ax, clf)
        i += 1
plt.show()

""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Growing Spheres                                                         │
    └────────────────────────────────────────────────────────────────────────┘
 """

class GrowingSpheres:
    """
    obs_to_interprete : x, une observation à interpréter
    clf : classifieur binaire
    eta : hyperparamètre
    n : nombre de points que l'on génère
    """

    def __init__(self, clf, eta, n):
        self.clf = clf
        self.eta = eta
        self.n = n

    def generate_spherical_layer(self, a0, a1) -> np.ndarray:
        """
        Generate a spherical layer with the specified parameters.

        Parameters:
            a0 (float): Inner radius of the spherical layer.
            a1 (float): Outer radius of the spherical layer.

        Returns:
            np.ndarray: A numpy array representing the generated spherical layer.
        """

        def norm(v):
            return np.linalg.norm(v, ord=2, axis=1)

        z = np.random.normal(0, 1, (self.n, self.d))
        u = np.random.uniform(a0**self.d, a1**self.d, size=self.n)
        u = u ** (1 / self.d)
        z = np.array([a * b / c for a, b, c in zip(z, u, norm(z))]) # z = z * u / norm(z)
        return self.obs_to_interprete + z

    # ce qui serait bien c'est avoir une animation de l'algo qui fit au fur et à mesure
    # (les sphères)

    def find_enemy(self, spherical_layer):
        """
        Find and update enemy information in a spherical layer.

        Parameters:
            spherical_layer (numpy.ndarray): A 2D numpy array representing the spherical layer data.

        Returns:
            bool: True if enemies are found in the spherical layer, False otherwise.
        """
        pred = self.clf.predict(spherical_layer)
        self.enemies = spherical_layer[pred != self.obs_predict]
        return (pred != self.obs_predict).any()

    def predict(self, obs_to_interprete):
        self.obs_to_interprete = obs_to_interprete.reshape(1, -1)
        self.obs_predict = clf.predict(self.obs_to_interprete)
        self.d = self.obs_to_interprete.shape[1]
        
        enemy = self.generation()
        return enemy, self.feature_selection(enemy)
    
    def generation(self):
        spherical_layer = self.generate_spherical_layer(0, 1)
        while self.find_enemy(spherical_layer):
            self.eta /= 2
            spherical_layer = self.generate_spherical_layer(0, self.eta)
        a0 = self.eta
        a1 = 2 * self.eta
        while not self.find_enemy(spherical_layer):
            spherical_layer = self.generate_spherical_layer(a0, a1)
            a0 = a1
            a1 = a1 + self.eta
        return self.enemies[
            np.linalg.norm(self.enemies - self.obs_to_interprete).argmin()
        ]


    def feature_selection(self, enemy):
        e_prime = enemy.copy()
        while self.obs_predict != self.clf.predict(e_prime.reshape(1,-1)):
            e_star = e_prime.copy()
            i = np.abs(e_prime - self.obs_to_interprete[0])
            i = i[i != 0].argmin()
            e_prime[i] = self.obs_to_interprete[0][i]
        return e_star
