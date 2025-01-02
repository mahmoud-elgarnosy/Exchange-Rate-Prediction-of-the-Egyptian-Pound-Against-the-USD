from typing import Tuple
import numpy as np
import pandas as pd
from arch import arch_model
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from src.utils.data_classes import TimeSeriesStats


class TimeSeriesAnalyzer:
    """Class for analyzing time series data."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame."""
        self._validate_dataframe(df)
        self.df = df.copy()

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")

    def calculate_statistics(self, column_name: str) -> pd.Series:
        """Calculate essential time series statistics."""
        self._validate_column(column_name)
        return TimeSeriesStats.from_series(self.df[column_name]).to_series()

    def create_acf_pacf_plot(
            self,
            column_name: str,
            nlags: int = 40
    ) -> Tuple[go.Figure, np.ndarray, np.ndarray]:
        """Create ACF and PACF plots with confidence intervals."""
        self._validate_column(column_name)

        # Calculate ACF and PACF
        acf_values = acf(self.df[column_name], nlags=nlags)
        pacf_values = pacf(self.df[column_name], nlags=nlags)

        fig = self._create_acf_pacf_figure(
            acf_values=acf_values,
            pacf_values=pacf_values,
            nlags=nlags,
            confidence_interval=1.96 / np.sqrt(len(self.df))
        )

        return fig, acf_values, pacf_values

    def analyze_volatility(
            self,
            column_name: str,
            window: int = 30
    ) -> Tuple[go.Figure, pd.Series]:
        """Analyze and plot volatility using GARCH model."""
        self._validate_column(column_name)

        # Calculate returns and fit GARCH model
        returns = self._calculate_returns(column_name, window)
        volatility = self._fit_garch_model(returns) / 10

        # Create visualization
        fig = self._create_volatility_figure(
            returns=returns,
            volatility=volatility,
            column_name=column_name,
            window=window
        )

        return fig, volatility

    def _validate_column(self, column_name: str) -> None:
        """Validate if column exists in DataFrame."""
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

    @staticmethod
    def _create_acf_pacf_figure(
            acf_values: np.ndarray,
            pacf_values: np.ndarray,
            nlags: int,
            confidence_interval: float
    ) -> go.Figure:
        """Create ACF and PACF figure."""
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=('Autocorrelation Function (ACF)',
                            'Partial Autocorrelation Function (PACF)')
        )

        # Add ACF and PACF traces
        for i, (values, name) in enumerate([(acf_values, 'ACF'),
                                            (pacf_values, 'PACF')], 1):
            fig.add_trace(
                go.Scatter(
                    x=list(range(nlags + 1)),
                    y=values,
                    mode='lines+markers',
                    name=name
                ),
                row=i,
                col=1
            )

            # Add confidence intervals
            for sign in [1, -1]:
                fig.add_hline(
                    y=sign * confidence_interval,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="95% Confidence Interval" if sign > 0 else None,
                    row=i,
                    col=1
                )

        fig.update_layout(
            height=800,
            title_text="ACF and PACF Analysis"
        )

        return fig

    def _calculate_returns(self, column_name: str, window: int) -> pd.Series:
        """Calculate returns for the time series."""
        returns = self.df[column_name].pct_change(window)
        return returns.iloc[window:]

    @staticmethod
    def _fit_garch_model(returns: pd.Series) -> pd.Series:
        """Fit GARCH model to returns data."""
        model = arch_model(returns * 10, vol="GARCH", p=1, q=1)
        fitted_model = model.fit(disp="off")
        return fitted_model.conditional_volatility

    @staticmethod
    def _create_volatility_figure(
            returns: pd.Series,
            volatility: pd.Series,
            column_name: str,
            window: int
    ) -> go.Figure:
        """Create volatility analysis figure."""
        fig = go.Figure()

        # Add returns trace
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=returns,
            mode="lines",
            name="returns",
            line=dict(color="blue")
        ))

        # Add volatility trace
        fig.add_trace(go.Scatter(
            x=volatility.index,
            y=volatility,
            mode="lines",
            name="Volatility",
            line=dict(color="red")
        ))

        fig.update_layout(
            title=f"{window} rolling Returns and GARCH Volatility - {column_name}",
            xaxis_title="Date",
            yaxis_title="Values",
            legend_title="Legend",
            template="plotly_white"
        )

        return fig
