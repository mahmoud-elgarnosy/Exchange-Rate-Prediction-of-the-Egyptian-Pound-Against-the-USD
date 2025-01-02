import plotly.graph_objects as go
import numpy as np
from src.utils.detect_outlier import detect_global_outliers, detect_local_outliers, interpolate_outliers


def visualize_outliers(data, timestamps=None):
    """
    Create an interactive plot showing time series data with local and global outliers.

    Parameters:
    data (array-like): Time series data
    timestamps (array-like, optional): Timestamps for x-axis. If None, will use indices

    Returns:
    plotly.graph_objects.Figure
    """
    # If no timestamps provided, use indices
    if timestamps is None:
        timestamps = np.arange(len(data))

    # Detect outliers
    global_mask, global_indices = detect_global_outliers(data)
    local_mask, local_indices = detect_local_outliers(data)

    # Create figure
    fig = go.Figure()

    # Add main line trace
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=data,
        mode='lines',
        name='Time Series',
        line=dict(color='rgb(31, 119, 180)'),
    ))

    # Add global outliers
    if len(global_indices) > 0:
        fig.add_trace(go.Scatter(
            x=timestamps[global_indices],
            y=data[global_indices],
            mode='markers',
            name='Global Outliers',
            marker=dict(
                size=12,
                color='red',
                symbol='circle-open',
                line=dict(width=2)
            )
        ))

    # Add local outliers
    local_only_indices = [i for i in local_indices if i not in global_indices]
    if len(local_only_indices) > 0:
        fig.add_trace(go.Scatter(
            x=timestamps[local_only_indices],
            y=data[local_only_indices],
            mode='markers',
            name='Local Outliers',
            marker=dict(
                size=12,
                color='orange',
                symbol='circle-open',
                line=dict(width=2)
            )
        ))

    # Update layout
    fig.update_layout(
        title='Time Series with Outlier Detection',
        xaxis_title='Time',
        yaxis_title='Value',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        width=1000,
        height=600
    )

    return fig


def visualize_interpolation(data, method='linear'):
    """
    Create an interactive plot showing original data, outliers, and interpolation.

    Parameters:
    data (array-like): Time series data
    method (str): Interpolation method

    Returns:
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    # Get outliers and interpolated data
    interpolated_data, outlier_indices = interpolate_outliers(data, method=method)

    # Create figure
    fig = go.Figure()

    # Add original data
    fig.add_trace(go.Scatter(
        x=np.arange(len(data)),
        y=data,
        mode='lines',
        name='Original Data',
        line=dict(color='rgb(31, 119, 180)')
    ))

    # Add interpolated line
    fig.add_trace(go.Scatter(
        x=np.arange(len(data)),
        y=interpolated_data,
        mode='lines',
        name='Interpolated Data',
        line=dict(color='rgb(255, 127, 14)', dash='dash')
    ))

    # Add outlier points
    if len(outlier_indices) > 0:
        fig.add_trace(go.Scatter(
            x=outlier_indices,
            y=data[outlier_indices],
            mode='markers',
            name='Outliers',
            marker=dict(
                size=10,
                color='red',
                symbol='circle-open',
                line=dict(width=2)
            )
        ))

    # Update layout
    fig.update_layout(
        title=f'Time Series with {method.capitalize()} Interpolation',
        xaxis_title='Time',
        yaxis_title='Value',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        width=1000,
        height=600
    )

    return fig
