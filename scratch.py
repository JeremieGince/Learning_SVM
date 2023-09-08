from typing import Optional

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
    raise NotImplementedError()
