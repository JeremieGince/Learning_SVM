from copy import deepcopy
from typing import Optional, Union, Callable

import numpy as np
from pennylane import AngleEmbedding
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
import pennylane as qml
import pythonbasictools as pbt


class SVMFromScratch(BaseEstimator):
    def __init__(
            self,
            kernel: Union[str, Callable] = "linear",
            lmbda: float = 1e-5,
            eta: float = 0.1,
            max_iter: int = 100,
            tol: float = 1e-3,
    ):
        self.lmbda = lmbda
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.multipliers = None
        self.support_indexes = None
        self.bias = None
        self.training_kernel = None

    @property
    def dual_coef_(self):
        if self.multipliers is None:
            return None
        return self.multipliers[self.support_indexes] * self.y_[self.support_indexes]

    @property
    def intercept_(self):
        return self.bias

    @property
    def support_vectors_(self):
        if self.support_indexes is None:
            return None
        return self.X_[self.support_indexes]
    
    def apply_kernel(self, x0, x1):
        if isinstance(self.kernel, str):
            return pairwise_kernels(x0, x1, metric=self.kernel)
        else:
            return self.kernel(x0, x1)
        
    def _get_support_kernel(self, x):
        x = np.asarray(x)
        is_training_set = (self.X_.shape == x.shape) and np.allclose(self.X_, x)
        if is_training_set:
            if self.training_kernel is None:
                self.training_kernel = self.apply_kernel(self.X_, self.X_)
            _support_kernel = self.training_kernel[self.support_indexes, :]
        else:
            _support_kernel = self.apply_kernel(self.support_vectors_, x)
        return _support_kernel  # (n_supports, n_samples)
    
    def compute_h(self, x):
        x = np.asarray(x)
        h = np.dot(
            self.dual_coef_.reshape(1, -1),  # (1, n_supports)
            self._get_support_kernel(x)  # (n_supports, n_samples)
        ) + self.bias  # (1, n_supports) @ (n_supports, n_samples) + (1, n_samples) -> (1, n_samples)
        h = h.reshape(-1)  # (n_samples, )
        return h
    
    def compute_hinge_loss(self, x):
        loss = np.sum(np.maximum(0, 1 - self.y_ * self.compute_h(x)))
        reg = (self.lmbda / 2) * np.sum(self.multipliers[self.support_indexes] ** 2)
        return loss + reg
    
    def compute_derivative_hinge_by_multipliers(self, x):
        condition = ((1 - self.y_ * self.compute_h(x)) > 0.0).astype(int)  # (n_samples, )
        kernel = self._get_support_kernel(x)  # (n_supports, n_samples)
        ys = self.y_[self.support_indexes]
        d_loss = np.dot(-self.y_.reshape(1, -1), np.dot(ys, kernel)).reshape(-1)  # (n_samples, )
        d_reg = self.lmbda * np.sum(np.abs(self.multipliers))  # (n_samples, )
        return d_loss * condition + d_reg
    
    def compute_derivative_hinge_by_bias(self, x):
        condition = ((1 - self.y_ * self.compute_h(x)) > 0.0).astype(int)  # (n_samples, )
        return np.sum(-self.y_ * condition)

    def compute_bias(self, x, y):
        h = self.compute_h(x)
        bias = np.mean(y - h)
        return bias
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise NotImplementedError("Only binary classification is supported")
        self.X_ = deepcopy(X)
        self.y_ = deepcopy(y)
        # make sure that y is in {-1, 1}
        self.y_[self.y_ == self.classes_[0]] = -1
        self.y_[self.y_ == self.classes_[1]] = 1
        
        self.multipliers = np.ones(X.shape[0])
        self.support_indexes = np.arange(X.shape[0])
        self.bias = 0.0
        
        self.training_kernel = self.apply_kernel(self.X_, self.X_)

        iteration = 0
        self.history = []
        while iteration != self.max_iter:
            d_hl_d_m = self.compute_derivative_hinge_by_multipliers(self.X_)
            d_hl_d_b = self.compute_derivative_hinge_by_bias(self.X_)

            self.multipliers -= self.eta * d_hl_d_m
            self.multipliers[self.multipliers < 0.0] = 0.0
            # self.multipliers[self.multipliers > self._C] = 0.0
            # n_supports = max(1, int(0.5 * self.multipliers.shape[0]))
            # put the top n_supports multipliers to 0
            # self.multipliers[np.argsort(self.multipliers)[:-n_supports]] = 0.0

            self.support_indexes = np.argwhere(self.multipliers > 0.0).reshape(-1)
            self.bias -= self.eta * d_hl_d_b

            loss = self.compute_hinge_loss(self.X_)
            self.history.append(loss)
            if loss <= self.tol:
                break
            iteration += 1
        return self
    
    def predict(self, x):
        check_is_fitted(self)
        x = check_array(x)
        y_hat = np.sign(self.compute_h(x)).astype(int)
        y_hat[y_hat == -1] = self.classes_[0]
        y_hat[y_hat == 1] = self.classes_[1]
        return y_hat
    
    def predict_proba(self, x):
        check_is_fitted(self)
        x = check_array(x)
        return self.compute_h(x)
    
    def score(self, X, y):
        check_is_fitted(self)
        X, y = check_X_y(X, y)
        y_hat = self.predict(X)
        return np.mean(y_hat == y)

    def get_hyperplane(self):
        check_is_fitted(self)
        w = np.dot(self.dual_coef_, self.support_vectors_)
        b = self.bias
        return w, b
    
    def decision_function(self, x):
        check_is_fitted(self)
        x = check_array(x)
        return self.compute_h(x)

    def visualize(self, x, y, **kwargs):
        import matplotlib.pyplot as plt

        check_is_fitted(self)
        x, y = check_X_y(x, y)
        if x.shape[-1] != 2:
            raise ValueError(f"x.shape[-1] = {x.shape[-1]} != 2. Only 2D data is supported.")
        if y.ndim != 1:
            raise ValueError(f"y.ndim = {y.ndim} != 1. Only 1D labels are supported.")

        w, b = self.get_hyperplane()
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, num=100),
            np.linspace(y_min, y_max, num=100)
        )
        x_mesh = np.c_[xx.ravel(), yy.ravel()]
        y_mesh = self.predict(x_mesh).reshape(xx.shape)
        h_mesh = self.compute_h(x_mesh).reshape(xx.shape)

        fig, ax = kwargs.get("fig", None), kwargs.get("ax", None)
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(14, 10))

        ax.contourf(xx, yy, h_mesh, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")

        # plot the decision function
        ax.contour(xx, yy, h_mesh, levels=[0], linewidths=2, colors="k")
        # plot the positive distance margin
        ax.contour(xx, yy, h_mesh, levels=[1], linewidths=2, colors="k", linestyles="--")
        # plot the negative distance margin
        ax.contour(xx, yy, h_mesh, levels=[-1], linewidths=2, colors="k", linestyles="--")
        # plot the support vectors
        ax.scatter(
            self.support_vectors_[:, 0],
            self.support_vectors_[:, 1],
            s=100,
            facecolors="none",
            edgecolors="k"
        )

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("SVM from scratch")
        plt.show()
        return fig, ax


class SVC:
    def __new__(cls, *args, **kwargs):
        decision_function_shape = kwargs.get("decision_function_shape", "ovr").lower()
        if decision_function_shape == "ovr":
            return OneVsRestClassifier(SVMFromScratch(*args, **kwargs))
        elif decision_function_shape == "ovo":
            return OneVsOneClassifier(SVMFromScratch(*args, **kwargs))
        else:
            raise ValueError(f"Unknown decision_function_shape: {decision_function_shape}")

