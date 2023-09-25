
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
    │ Données                                                                │
    └────────────────────────────────────────────────────────────────────────┘
 """

datasets = []
for noise in np.arange(0, 0.5, 0.1):
    datasets.append(make_moons(noise=noise, random_state=0))
