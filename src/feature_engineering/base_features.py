import pandas as pd


class BaseFeatureGenerator:
    """Base class for generating common financial features."""

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features for seasonal patterns."""
        df_features = df.copy()
        df_features['day_of_week'] = df.index.dayofweek
        df_features['day_of_month'] = df.index.day
        return df_features

    def create_lagged_features(self, df: pd.DataFrame, column: str, lag_days) -> pd.DataFrame:
        """Create lagged features for autocorrelation patterns."""
        df_features = df.copy()
        for lag in lag_days:
            df_features[f'lag_{lag}d'] = df[column].shift(lag)
        return df_features

    def create_rolling_features(self, df: pd.DataFrame, column: str, windows) -> pd.DataFrame:
        """Create rolling statistics for trend and volatility."""
        df_features = df.copy()
        for window in windows:
            rolling = df[column].rolling(window=window)
            df_features.update({
                f'rolling_mean_{window}d': rolling.mean(),
                f'rolling_std_{window}d': rolling.std(),
                f'rolling_min_{window}d': rolling.min(),
                f'rolling_max_{window}d': rolling.max(),
                f'momentum_{window}d': df[column] - df[column].shift(window),
                f'return_{window}d': df[column].pct_change(window)
            })
        return df_features

    def create_technical_features(self, df: pd.DataFrame, column: str,
                                  ema_spans, roc_periods) -> pd.DataFrame:
        """Create technical analysis features."""
        df_features = df.copy()

        # Exponential moving averages
        for span in ema_spans:
            df_features[f'ema_{span}d'] = df[column].ewm(span=span, adjust=False).mean()

        # Rate of change
        for period in roc_periods:
            df_features[f'roc_{period}d'] = df[column].pct_change(period) * 100

        return df_features