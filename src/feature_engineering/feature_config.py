from dataclasses import dataclass
from typing import List

@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    lag_days: List[int] = None
    rolling_windows: List[int] = None
    ema_spans: List[int] = None
    roc_periods: List[int] = None
    inflation_lags: List[int] = None
    gold_windows: List[int] = None

    def __post_init__(self):
        """Set default values if None."""
        self.lag_days = self.lag_days or [1, 2, 3, 4]
        self.rolling_windows = self.rolling_windows or [2, 3, 5]
        self.ema_spans = self.ema_spans or [3, 5, 7, 14, 21]
        self.roc_periods = self.roc_periods or [2, 5, 10]
        self.inflation_lags = self.inflation_lags or [1, 2]
        self.gold_windows = self.gold_windows or [2, 3, 4, 7, 14]