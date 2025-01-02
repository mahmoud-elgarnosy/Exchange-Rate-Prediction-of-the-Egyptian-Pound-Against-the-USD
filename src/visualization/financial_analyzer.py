from typing import Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils.data_classes import FinancialStats


class FinancialAnalyzer:
    """Class for analyzing financial time series data."""

    def __init__(
            self,
            exchange_rates: pd.DataFrame,
            gold_prices: pd.DataFrame,
            inflation_rates: pd.DataFrame
    ):
        """Initialize with financial data."""
        self._validate_inputs(exchange_rates, gold_prices, inflation_rates)
        self.exchange_rates = exchange_rates
        self.gold_prices = gold_prices
        self.inflation_rates = inflation_rates
        self.exchange_rate_col = 'EGP=X'
        self.gold_price_col = '21K - Local Price/Buy'
        self.inflation_col = 'Core (m/m)'

    @staticmethod
    def _validate_inputs(*dataframes: pd.DataFrame) -> None:
        """Validate input DataFrames."""
        for df in dataframes:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("All inputs must be pandas DataFrames")
            if df.empty:
                raise ValueError("DataFrames cannot be empty")

    def calculate_statistics(self) -> pd.DataFrame:
        """Calculate comprehensive statistics for a financial time series."""
        combined_df = self.exchange_rates.merge(
            self.gold_prices, left_index=True, right_index=True)
        return FinancialStats.from_df(combined_df).to_df()

    def analyze_correlations(self) -> Tuple[go.Figure, pd.DataFrame]:
        """Analyze correlations between financial indicators."""
        # Resample and merge data
        monthly_data = self.prepare_monthly_data()
        correlation_matrix = monthly_data.corr()

        # Create correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            text=np.round(correlation_matrix, 2),
            texttemplate='%{text}',
            textfont={'size': 12},
            hoverongaps=False,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))

        fig.update_layout(
            title='Correlation Matrix of Financial Indicators',
            height=600,
            width=600
        )

        return fig, correlation_matrix

    def plot_rolling_corr(self, window=30) -> go.Figure:
        """Create interactive plots of rolling statistics."""

        exchange_returns = self.exchange_rates['EGP=X'].pct_change()
        gold_returns = self.gold_prices['21K - Local Price/Buy'].pct_change()

        # Calculate rolling correlations
        rolling_corr = exchange_returns.rolling(window=window).corr(gold_returns)

        fig = go.Figure()

        # Add returns trace
        fig.add_trace(go.Scatter(x=self.exchange_rates.index, y=rolling_corr,
                                 name='',
                                 line=dict(color='red')))
        fig.update_layout(
            title_text="'30-day Rolling Exchange Rate (EGP/USD) vs Gold Price Correlation'",
            hovermode='x unified'
        )

        return fig

    def plot_time_series_overview(self) -> go.Figure:
        """Create interactive overview plot of all time series."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Exchange Rate and Gold Price',
                            'Monthly Inflation Rate'),
            vertical_spacing=0.2,
            specs=[[{"secondary_y": True}],
                   [{"secondary_y": False}]]
        )

        # Add traces
        self._add_overview_traces(fig)

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Financial Time Series Overview",
            hovermode='x unified'
        )

        # Update axis labels
        self._update_overview_axes(fig)

        return fig

    def prepare_monthly_data(self) -> pd.DataFrame:
        """Prepare monthly data for correlation analysis."""
        monthly_exchange = self.exchange_rates.resample('M').mean()
        monthly_gold = self.gold_prices.resample('M').mean()

        combined_df = monthly_exchange.merge(
            monthly_gold, left_index=True, right_index=True)
        return combined_df.merge(
            self.inflation_rates, left_index=True, right_index=True)

    @staticmethod
    def _validate_column(df: pd.DataFrame, column: str) -> None:
        """Validate if column exists in DataFrame."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

    def _add_overview_traces(self, fig: go.Figure) -> None:
        """Add overview traces to figure."""
        # Exchange Rate
        fig.add_trace(
            go.Scatter(x=self.exchange_rates.index,
                       y=self.exchange_rates[self.exchange_rate_col],
                       name="EGP/USD Rate",
                       line=dict(color='#1f77b4')),
            row=1, col=1, secondary_y=False
        )

        # Gold Price
        fig.add_trace(
            go.Scatter(x=self.gold_prices.index,
                       y=self.gold_prices[self.gold_price_col],
                       name="Gold Price",
                       line=dict(color='#ff7f0e')),
            row=1, col=1, secondary_y=True
        )

        # Inflation
        fig.add_trace(
            go.Scatter(x=self.inflation_rates.index,
                       y=self.inflation_rates[self.inflation_col],
                       name="Inflation Rate",
                       mode='lines+markers',
                       line=dict(color='green')),
            row=2, col=1
        )

    @staticmethod
    def _update_overview_axes(fig: go.Figure) -> None:
        """Update axes labels for overview plot."""
        fig.update_yaxes(title_text="Exchange Rate (EGP/USD)",
                         row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Gold Price",
                         row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Inflation Rate (%)",
                         row=2, col=1)
