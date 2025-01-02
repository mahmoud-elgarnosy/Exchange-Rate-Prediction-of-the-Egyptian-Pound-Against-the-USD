import joblib
import pandas as pd
from typing import Union
import numpy as np


class ModelPredictor:
    def __init__(self, model_path: str):
        """
        Initialize predictor with saved model.

        Args:
            model_path: Path to the saved model file
        """
        self.model = joblib.load(model_path)

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using loaded model.

        Args:
            data: Input features for prediction

        Returns:
            Array of predictions
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        return self.model.predict(data)

    def get_feature_importance(self) -> Union[pd.Series, None]:
        """Get feature importance if model supports it."""
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(self.model.feature_importances_)
        return None