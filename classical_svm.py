import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import decomposition
from sklearn import svm


dataset = datasets.load_breast_cancer(as_frame=True)
X = np.array(dataset.data)
y = np.array(dataset.target)

print(f"{X.shape = }, {y.shape = }")
print(f"{np.unique(y) = }")

model = svm.SVC()

model.fit(X, y)
accuracy = model.score(X, y)
print(f"{accuracy = }")


pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)

n_pts = 10_000
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, int(np.sqrt(n_pts))),
    np.linspace(y_min, y_max, int(np.sqrt(n_pts)))
)

X_reduced_mesh = np.c_[xx.ravel(), yy.ravel()]
X_mesh = pca.inverse_transform(X_reduced_mesh)
Y_pred = model.predict(X_mesh)
Y_mesh = Y_pred.reshape(xx.shape)
print(f"{X_reduced_mesh.shape = }, {Y_mesh.shape = }")

fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(14, 10))

cm = plt.cm.RdBu
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(dataset.target_names))]
print(f"{colors = }")
legend_labels = dataset.target_names

# make the contourf plot with the colors associated with each class
# ax.contourf(xx, yy, Y_mesh, cmap=cmap, alpha=0.5)

# plot the meshgrid points with the colors associated with each class
ax.contourf(xx, yy, Y_mesh, cmap=cmap, alpha=0.8)

scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    # c=np.array(colors)[y].tolist(),
    c=[cmap(i) for i in y],
    edgecolor='k',
    linewidths=1.5,
    s=100
)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

patches = []
for color, legend in zip(colors, legend_labels):
    patch = matplotlib.lines.Line2D(
        [0], [0], marker='o', linestyle='None', markerfacecolor=color,
        markersize=10, markeredgecolor='k', label=legend
    )
    patches.append(patch)

ax.set_title(
    f"Frontières de décision dans l'espace réduit." \
    f"\nExactitude du modèle: {accuracy*100:.2f}%.",
    fontsize=24
)

ax.legend(handles=patches, fontsize=18)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.set_xlabel("PC 1", fontsize=18)
ax.set_ylabel("PC 2", fontsize=18)
ax.minorticks_on()

plt.show()





