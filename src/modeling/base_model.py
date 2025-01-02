from abc import abstractmethod
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class BaseModel:
    """Base class for all models."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.best_params: Optional[Dict] = None

    @abstractmethod
    def _create_model(self) -> None:
        """Create the underlying model."""
        pass

    @abstractmethod
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Get feature importance for the model."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Fit the model to the training data."""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        self._create_model()
        self.model.fit(X_scaled_df, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        return self.model.predict(X_scaled_df)