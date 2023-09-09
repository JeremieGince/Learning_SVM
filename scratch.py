from typing import Optional, Union, Callable

import numpy as np
from pennylane import AngleEmbedding
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
            C: float = 1.0,
            lmbda: float = 0.1,
            eta: float = 0.1,
            n_iter: int = 100,
            
    ):
        self._C = C
        self._lmbda = lmbda
        self._eta = eta
        self._n_iter = n_iter
        self._kernel = kernel
        self.multipliers = None
        self.support_indexes = None
        self.bias = None
    
    def apply_kernel(self, x0, x1):
        if isinstance(self._kernel, str):
            return pairwise_kernels(x0, x1, metric=self._kernel)
        else:
            return self._kernel(x0, x1)
    
    def compute_h(self, x):
        supports = self.X_[self.support_indexes]
        kernel = self.apply_kernel(supports, x)
        alphas = self.multipliers[self.support_indexes]
        ys = self.y_[self.support_indexes]
        alpha_y = alphas * ys
        h = np.dot(alpha_y, kernel) + self.bias
        return h
    
    def compute_hinge_loss(self, x):
        loss = np.sum(1 - self.y_ * self.compute_h(x))
        reg = (self._lmbda / 2) * np.sum(self.multipliers**2)
        return loss + reg
    
    def compute_derivative_hinge_by_multipliers(self, x):
        supports = self.X_[self.support_indexes]
        kernel = self.apply_kernel(supports, x)
        ys = self.y_[self.support_indexes]
        return -self.y_ * np.dot(ys, kernel) + self._lmbda * self.multipliers
    
    def compute_derivative_hinge_by_bias(self, x):
        return -self.y_
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise NotImplementedError("Only binary classification is supported")
        self.X_ = X
        self.y_ = y
        # make sure that y is in {-1, 1}
        self.y_[self.y_ == self.classes_[0]] = -1
        self.y_[self.y_ == self.classes_[1]] = 1
        
        self.multipliers = np.ones(X.shape[0])
        self.support_indexes = np.arange(X.shape[0])
        self.bias = 0.0
        
        for _ in range(self._n_iter):
            d_hl_d_m = self.compute_derivative_hinge_by_multipliers(self.X_)
            d_hl_d_b = self.compute_derivative_hinge_by_bias(self.X_)
            self.multipliers -= self._eta * d_hl_d_m
            self.multipliers = np.clip(self.multipliers, 0.0, self._C)
            self.support_indexes = np.where(self.multipliers > 0.0)[0]
            self.bias -= self._eta * d_hl_d_b
            
            if np.isclose(self.compute_hinge_loss(self.X_), 0.0):
                break
            
        return self
    
    def predict(self, x):
        check_is_fitted(self)
        x = check_array(x)
        y_hat = np.sign(self.compute_h(x))
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
        return np.mean(self.predict(X) == y)
