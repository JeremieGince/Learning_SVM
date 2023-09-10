import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from kernels import ClassicalKernel, QuantumKernel
from scratch import SVC
from visualization import Visualizer

if __name__ == '__main__':
    # dataset = datasets.load_breast_cancer(as_frame=True)
    # dataset = datasets.load_iris(as_frame=True)
    dataset = datasets.make_classification(
        n_samples=100,
        n_features=2,
        n_classes=2,
        n_clusters_per_class=1,
        n_informative=2,
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

    # X = StandardScaler().fit_transform(X)
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    # y = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y.reshape(-1, 1)).reshape(-1).astype(int)
    print(f"{X.shape = }, {y.shape = }")
    print(f"{np.unique(y) = }")

    embedding_size = X.shape[-1]

    rn_state = np.random.RandomState(seed=0)
    rn_embed_matrix = rn_state.randn(X.shape[-1], embedding_size)

    clas_kernel = ClassicalKernel(
        embedding_dim=embedding_size,
        metric="rbf",
        # encoder_matrix=rn_embed_matrix,
        seed=0
    ).fit(X, y)
    q_kernel = QuantumKernel(
        embedding_dim=embedding_size,
        seed=0,
        # encoder_matrix=rn_embed_matrix,
        shots=32,
    ).fit(X, y)

    clas_model = svm.SVC(kernel=clas_kernel.kernel, random_state=0)
    qml_model = svm.SVC(kernel=q_kernel.kernel, random_state=0)
    scratch_model = SVC(kernel=clas_kernel.kernel, max_iter=1_000)
    q_scratch_model = SVC(kernel=q_kernel.kernel, max_iter=1_000)
    
    models = {
        "classical": clas_model,
        "scratch": scratch_model,
        "qml": qml_model,
        "q_scratch": q_scratch_model,
    }
    n_plots = len(models)
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(14, 10), sharex="all", sharey="all")
    axes = np.ravel(np.asarray([axes]))
    for i, (m_name, model) in enumerate(models.items()):
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
            n_pts=(100 if m_name.startswith('q') else 100_000),
            title=f"Decision boundaries in the reduced space.",
            legend_labels=getattr(dataset, "target_names", None),
            # axis_name="RN",
            fig=fig, ax=axes[i],
            interpolation="nearest",
        )
        ax.set_title(f"{m_name} accuracy: {accuracy * 100:.2f}%")

    plt.show()

    # models["scratch"].visualize(X, y)
    # models["q_scratch"].visualize(X, y)

