import pandas as pd
import numpy as np


class SpecializedFeatureGenerator:
    """Class for generating specialized financial features."""

    def _prepare_daily_inflation(self, df_inflation: pd.DataFrame,
                                 target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Prepare daily inflation data from monthly data."""
        df_inflation = df_inflation.copy()
        df_inflation.index = pd.to_datetime(df_inflation.index).to_period('M').to_timestamp()

        return df_inflation.reindex(
            pd.date_range(df_inflation.index.min(), df_inflation.index.max(), freq='D')
        ).ffill().reindex(target_index)

    def create_inflation_features(self, df_daily: pd.DataFrame, df_inflation: pd.DataFrame,
                                  exchange_col: str, inflation_col: str,
                                  inflation_lags) -> pd.DataFrame:
        """Create inflation-related features."""
        df_features = df_daily.copy()

        # Prepare daily inflation data
        df_inflation_daily = self._prepare_daily_inflation(df_inflation, df_features.index)
        df_features['inflation_current'] = df_inflation_daily[inflation_col]

        # Create lagged features
        for i in inflation_lags:
            df_features[f'inflation_lag_{i}m'] = df_inflation_daily[inflation_col].shift(30 * i)

        # Create rolling features
        for window in [2, 3]:
            days = window * 30
            df_features[f'inflation_mean_{window}m'] = (
                df_inflation_daily[inflation_col]
                .rolling(window=days, min_periods=1)
                .mean()
            )
            df_features[f'inflation_std_{window}m'] = (
                df_inflation_daily[inflation_col]
                .rolling(window=days, min_periods=1)
                .std()
            )

        # Create interaction features
        df_features['inflation_exchange_ratio'] = (
                df_features[exchange_col] / df_features['inflation_current']
        )

        return df_features

    def _align_gold_data(self, df_exchange: pd.DataFrame,
                         df_gold: pd.DataFrame) -> pd.DataFrame:
        """Safely align gold data with exchange rate data."""
        df_exchange_clean = df_exchange.copy()
        df_gold_clean = df_gold.copy()

        # Handle duplicates
        if df_exchange_clean.index.duplicated().any():
            print("Warning: Found duplicate indices in exchange rate data. Taking last value.")
            df_exchange_clean = df_exchange_clean.loc[~df_exchange_clean.index.duplicated(keep='last')]

        if df_gold_clean.index.duplicated().any():
            print("Warning: Found duplicate indices in gold data. Taking last value.")
            df_gold_clean = df_gold_clean.loc[~df_gold_clean.index.duplicated(keep='last')]

        # Create complete date range
        date_range = pd.date_range(
            start=min(df_exchange_clean.index.min(), df_gold_clean.index.min()),
            end=max(df_exchange_clean.index.max(), df_gold_clean.index.max()),
            freq='D'
        )

        # Align and forward fill
        df_gold_aligned = df_gold_clean.reindex(date_range).ffill(limit=5)
        return df_gold_aligned.reindex(df_exchange_clean.index)

    def create_gold_features(self, df_exchange: pd.DataFrame, df_gold: pd.DataFrame,
                             exchange_col: str, gold_col: str,
                             gold_windows) -> pd.DataFrame:
        """Create gold-related features."""
        df_features = df_exchange.copy()

        # Align gold data
        df_gold_aligned = self._align_gold_data(df_exchange, df_gold)
        df_features['gold_price'] = df_gold_aligned[gold_col]

        valid_data = df_features['gold_price'].notna()

        if valid_data.any():
            # Returns and volatility
            for window in gold_windows:
                df_features[f'gold_return_{window}d'] = (
                    df_features.loc[valid_data, 'gold_price'].pct_change(window)
                )
                df_features[f'gold_volatility_{window}d'] = (
                    df_features.loc[valid_data, 'gold_price']
                    .pct_change()
                    .rolling(window)
                    .std()
                )

            # Correlation features
            exchange_returns = df_features.loc[valid_data, exchange_col].pct_change()
            gold_returns = df_features.loc[valid_data, 'gold_price'].pct_change()

            for window in gold_windows:
                df_features.loc[valid_data, f'gold_correlation_{window}d'] = (
                    exchange_returns.rolling(window)
                    .corr(gold_returns)
                )

            # Technical indicators
            df_features.loc[valid_data, 'gold_ema_short'] = (
                df_features.loc[valid_data, 'gold_price']
                .ewm(span=7, adjust=False)
                .mean()
            )
            df_features.loc[valid_data, 'gold_ema_long'] = (
                df_features.loc[valid_data, 'gold_price']
                .ewm(span=30, adjust=False)
                .mean()
            )

            df_features['gold_ema_signal'] = (
                    df_features['gold_ema_short'] > df_features['gold_ema_long']
            ).astype(int)
        else:
            print("Warning: No valid gold data found after alignment")

        return df_features