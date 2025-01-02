from dataclasses import dataclass, asdict
import pandas as pd
from scipy import stats


@dataclass
class TimeSeriesStats:
    """Data class to hold time series statistics."""
    mean: float
    median: float
    std_dev: float
    skewness: float
    kurtosis: float
    daily_returns_mean: float
    daily_returns_volatility: float
    monthly_returns_mean: float
    monthly_returns_volatility: float

    @classmethod
    def from_series(cls, series: pd.Series) -> 'TimeSeriesStats':
        """Create TimeSeriesStats from a pandas Series."""
        return cls(
            mean=series.mean(),
            median=series.median(),
            std_dev=series.std(),
            skewness=stats.skew(series),
            kurtosis=stats.kurtosis(series),
            daily_returns_mean=series.pct_change().mean() * 100,
            daily_returns_volatility=series.pct_change().std() * 100,
            monthly_returns_mean = series.pct_change(30).mean() * 100,
            monthly_returns_volatility = series.pct_change(30).std() * 100
        )

    def to_series(self) -> pd.Series:
        """Convert TimeSeriesStats to pandas Series."""
        return pd.Series(asdict(self))


@dataclass
class FinancialStats:
    """Data class to hold financial time series statistics."""
    mean: list
    median: list
    std_dev: list
    cv: list  # Coefficient of Variation
    skewness: list
    kurtosis: list
    daily_returns_mean: list
    daily_returns_volatility: list
    monthly_returns_mean: list
    monthly_returns_volatility: list

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> 'FinancialStats':
        """Create FinancialStats from a pandas Series."""
        return cls(
            mean=df.mean(),
            median=df.median(),
            std_dev=df.std(),
            cv=df.std() / df.mean(),
            skewness=df.skew(),
            kurtosis=df.kurtosis(),
            daily_returns_mean=df.pct_change().mean(),
            daily_returns_volatility=df.pct_change().std(),
            monthly_returns_mean=df.pct_change(30).mean(),
            monthly_returns_volatility=df.pct_change(30).std()
        )

    def to_df(self) -> pd.DataFrame:
        """Convert TimeSeriesStats to pandas Series."""
        return pd.DataFrame(asdict(self)).T
