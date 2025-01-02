from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class ModelMetrics:
    """Data class to hold model evaluation metrics."""
    rmse: float
    mae: float
    r2: float
    feature_importance: pd.DataFrame
    predictions: np.ndarray
    actual_values: np.ndarray
    model_name: str
    best_params: Optional[Dict] = None
