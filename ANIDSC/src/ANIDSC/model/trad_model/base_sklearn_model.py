import numpy as np
import pickle
from abc import ABC, abstractmethod
from typing import Any, Optional
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class BaseSklearnModel(ABC):
    """
    Base wrapper for scikit-learn estimators that mimics the preprocess/train/predict
    interface of the provided BaseTorchModel.

    Subclasses must implement `init_model(self)` to create/assign self.model
    (an sklearn-like estimator).
    """

    def __init__(self, input_dims: int = None, **kwargs):
        self.preprocessors = [self.ensure_2d]
        self.input_dims = input_dims
        self.model = self.init_model(**kwargs)

    @abstractmethod
    def init_model(self, **kwargs) -> None:
        """Create and assign self.model (an sklearn estimator)."""
        raise NotImplementedError

    # ----------------------
    # Preprocessing helpers
    # ----------------------
    def preprocess(self, X: Any) -> np.ndarray:
        """Run the configured preprocessors in order and return an ndarray."""
        for p in self.preprocessors:
            X = p(X)
        return X

    def ensure_2d(self, X: np.ndarray) -> np.ndarray:
        """Ensure X is 2D: shape (n_samples, n_features)."""
        X = np.asarray(X)
        if X.ndim == 0:
            return X.reshape(1, 1)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X

    # ----------------------
    # Prediction / Scoring
    # ----------------------
    def predict_step(self, X: np.ndarray):
        """
        Compute a model-based score/prediction for X.
        Order of preference:
            decision_function -> score_samples -> predict_proba -> predict -> transform -> score
        Returns numpy array, or None if the model has no suitable method or model is None.
        """
        
                
        Xp = self.preprocess(X)

        try:
            return self.model.decision_function(Xp)
        except NotFittedError as e:
            return np.full(X.shape[0],-1)
            
    # ----------------------
    # Training / updating
    # ----------------------
    def train_step(self, X: np.ndarray):
        """
        Update the model with a single batch.
        - If estimator supports partial_fit, calls partial_fit(X, y, **fit_kwargs)
        - Otherwise calls fit(X, y, **fit_kwargs) (for unsupervised, y can be None)
        Returns:
            - If y provided and estimator has `score`, returns model.score(X, y)
            - Otherwise returns None
        Note: calling fit repeatedly replaces the model fit (not incremental). Use partial_fit
        when streaming is required.
        """
        
        Xp = self.preprocess(X)

        self.model=self.model.partial_fit(Xp)
        return self.model.decision_function(Xp) # training loss in some sense
        
