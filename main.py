import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from kernels import ClassicalKernel, QuantumKernel
from visualization import Visualizer

if __name__ == '__main__':
    n_qubits = 4
    # dataset = datasets.load_breast_cancer(as_frame=True)
    # dataset = datasets.load_iris(as_frame=True)
    dataset = datasets.make_classification(
        n_samples=100,
        n_features=4,
        n_classes=2,
        n_clusters_per_class=1,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    if isinstance(dataset, tuple):
        X, y = dataset
    elif isinstance(dataset, dict):
        X = dataset["data"]
        y = dataset["target"]
    elif isinstance(dataset, pd.DataFrame):
        X = dataset.data
        y = dataset.target
    else:
        raise ValueError(f"Unknown dataset type: {type(dataset)}")

    X = StandardScaler().fit_transform(X)
    # y = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y.reshape(-1, 1)).reshape(-1).astype(int)
    print(f"{X.shape = }, {y.shape = }")
    print(f"{np.unique(y) = }")

    rn_state = np.random.RandomState(seed=0)
    rn_embed_matrix = rn_state.randn(X.shape[-1], n_qubits)

    clas_kernel = ClassicalKernel(
        embedding_dim=n_qubits,
        metric="rbf",
        encoder_matrix=rn_embed_matrix,
        seed=0
    ).fit(X, y)
    q_kernel = QuantumKernel(
        embedding_dim=n_qubits,
        seed=0,
        encoder_matrix=rn_embed_matrix,
    ).fit(X, y)

    clas_model = svm.SVC(kernel=clas_kernel.kernel, random_state=0)
    qml_model = svm.SVC(kernel=q_kernel.kernel, random_state=0)

    fig, axes = plt.subplots(2, 1, tight_layout=True, figsize=(14, 10), sharex="all", sharey="all")
    for i, (m_name, model) in enumerate(zip(["classical", "qml"], [clas_model, qml_model])):
        fit_start_time = time.time()
        model.fit(X, y)
        fit_end_time = time.time()
        fit_time = fit_end_time - fit_start_time
        accuracy = model.score(X, y)
        print(f"{m_name} accuracy: {accuracy * 100 :.4f}%, {fit_time = }[s]")

        fig, ax = Visualizer.plot_2d_decision_boundaries(
            model=model,
            X=X, y=y,
            # reducer=decomposition.PCA(n_components=2, random_state=0),
            # reducer=umap.UMAP(n_components=2, transform_seed=0, n_jobs=max(0, psutil.cpu_count() - 2)),
            check_estimators=False,
            n_pts=1_000,
            title=f"Decision boundaries in the reduced space.",
            legend_labels=getattr(dataset, "target_names", None),
            # axis_name="RN",
            fig=fig, ax=axes[i],
        )
        ax.set_title(f"{m_name} accuracy: {accuracy * 100:.2f}%")

    plt.show()
