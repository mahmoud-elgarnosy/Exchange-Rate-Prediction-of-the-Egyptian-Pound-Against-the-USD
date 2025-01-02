from typing import Tuple, Optional
import pandas as pd
from src.feature_engineering.feature_config import FeatureConfig
from src.feature_engineering.base_features import BaseFeatureGenerator
from src.feature_engineering.specialized_features import SpecializedFeatureGenerator


class FeatureEngineer:
    """Main class for creating financial time series features."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize with feature configuration."""
        self.config = config or FeatureConfig()
        self.base_generator = BaseFeatureGenerator()
        self.specialized_generator = SpecializedFeatureGenerator()

    def prepare_forecast_data(
            self,
            df_exchange: pd.DataFrame,
            df_gold: pd.DataFrame,
            df_inflation: pd.DataFrame,
            target_days: int = 7,
            exchange_col: str = 'EGP=X',
            gold_col: str = '21K - Local Price/Buy',
            inflation_col: str = 'Core (m/m)'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare complete feature set for forecasting."""
        # Create target
        df_target = df_exchange.copy()
        df_target['target'] = df_target[exchange_col].shift(-target_days)

        # Generate base features
        df_features = self.base_generator.create_temporal_features(df_target)
        df_features = self.base_generator.create_lagged_features(
            df_features, exchange_col, self.config.lag_days)
        df_features = self.base_generator.create_rolling_features(
            df_features, exchange_col, self.config.rolling_windows)
        df_features = self.base_generator.create_technical_features(
            df_features, exchange_col, self.config.ema_spans, self.config.roc_periods)

        # Generate specialized features
        df_features = self.specialized_generator.create_gold_features(
            df_features, df_gold, exchange_col, gold_col, self.config.gold_windows)
        df_features = self.specialized_generator.create_inflation_features(
            df_features, df_inflation, exchange_col, inflation_col, self.config.inflation_lags)

        # Clean and separate features
        df_features = df_features.dropna()
        y = df_features['target']
        X = df_features.drop(['target'], axis=1)

        return X, y
