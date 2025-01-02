from scipy import stats
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def detect_global_outliers(data, threshold=3):
    """
    Detect global outliers using z-score method.

    Parameters:
    data (array-like): Time series data
    threshold (float): Number of standard deviations to use as threshold (default: 3)

    Returns:
    tuple: (boolean mask of outliers, indices of outliers)
    """
    # Convert to numpy array if needed
    data = np.array(data)

    # Calculate z-scores
    z_scores = np.abs(stats.zscore(data))

    # Create boolean mask of outliers
    outliers_mask = z_scores > threshold

    # Get indices of outliers
    outlier_indices = np.where(outliers_mask)[0]

    return outliers_mask, outlier_indices


def detect_local_outliers(data, window_size=10, threshold=2):
    """
    Detect local outliers using rolling statistics.

    Parameters:
    data (array-like): Time series data
    window_size (int): Size of the rolling window
    threshold (float): Number of standard deviations to use as threshold

    Returns:
    tuple: (boolean mask of outliers, indices of outliers)
    """
    # Convert to pandas Series if needed
    data = pd.Series(data)

    # Calculate rolling statistics
    rolling_mean = data.rolling(window=window_size, center=True).mean()
    rolling_std = data.rolling(window=window_size, center=True).std()

    # Calculate z-scores within the rolling window
    z_scores = np.abs((data - rolling_mean) / rolling_std)

    # Create boolean mask of outliers
    outliers_mask = z_scores > threshold

    # Get indices of outliers
    outlier_indices = np.where(outliers_mask)[0]

    return outliers_mask.values, outlier_indices


def detect_combined_outliers(data, global_threshold=3, local_window_size=10, local_threshold=2):
    """
    Detect both global and local outliers and combine results.

    Parameters:
    data (array-like): Time series data
    global_threshold (float): Threshold for global outlier detection
    local_window_size (int): Window size for local outlier detection
    local_threshold (float): Threshold for local outlier detection

    Returns:
    tuple: (boolean mask of combined outliers, indices of combined outliers)
    """
    # Detect global outliers
    global_mask, _ = detect_global_outliers(data, global_threshold)

    # Detect local outliers
    local_mask, _ = detect_local_outliers(data, local_window_size, local_threshold)

    # Combine masks (point is outlier if detected by either method)
    combined_mask = global_mask | local_mask

    # Get indices of combined outliers
    combined_indices = np.where(combined_mask)[0]

    return combined_mask, combined_indices


# Example usage function
def analyze_outliers(data):
    """
    Analyze and print outlier detection results.

    Parameters:
    data (array-like): Time series data
    """
    # Detect outliers using all methods
    global_mask, global_indices = detect_global_outliers(data)
    local_mask, local_indices = detect_local_outliers(data)
    combined_mask, combined_indices = detect_combined_outliers(data)

    print(f"Global outliers found: {len(global_indices)}")
    print(f"Local outliers found: {len(local_indices)}")
    print(f"Combined outliers found: {len(combined_indices)}")

    return global_indices, local_indices, combined_indices


def interpolate_outliers(data, method='linear', window_size=5, polynomial_order=2):
    """
    Interpolate combined outliers using various methods.

    Parameters:
    data (array-like): Original time series data
    method (str): Interpolation method ('linear', 'cubic', 'nearest', 'spline', 'savgol')
    window_size (int): Window size for Savitzky-Golay filter
    polynomial_order (int): Polynomial order for Savitzky-Golay filter

    Returns:
    tuple: (interpolated data, interpolated indices)
    """
    # Convert to numpy array if needed
    data = np.array(data)

    # Detect combined outliers
    outlier_mask, outlier_indices = detect_combined_outliers(data)

    # Create copy of data for interpolation
    interpolated_data = data.copy()

    # Generate indices for the entire dataset
    all_indices = np.arange(len(data))

    # Get non-outlier indices and values
    clean_indices = all_indices[~outlier_mask]
    clean_values = data[~outlier_mask]

    if len(outlier_indices) == 0:
        return interpolated_data, []

    if method == 'savgol':
        # Use Savitzky-Golay filter for smoothing
        interpolated_data = savgol_filter(data, window_size, polynomial_order)
        # Only replace outlier points
        mask = ~outlier_mask
        interpolated_data[mask] = data[mask]

    else:
        # Create interpolation function using non-outlier points
        if method == 'linear':
            f = interp1d(clean_indices, clean_values, kind='linear',
                         fill_value='extrapolate')
        elif method == 'cubic':
            f = interp1d(clean_indices, clean_values, kind='cubic',
                         fill_value='extrapolate')
        elif method == 'nearest':
            f = interp1d(clean_indices, clean_values, kind='nearest',
                         fill_value='extrapolate')
        elif method == 'spline':
            if len(clean_indices) > 3:  # Cubic spline requires at least 4 points
                f = interp1d(clean_indices, clean_values, kind='cubic',
                             fill_value='extrapolate')
            else:
                f = interp1d(clean_indices, clean_values, kind='linear',
                             fill_value='extrapolate')
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        # Interpolate only the outlier points
        interpolated_data[outlier_indices] = f(outlier_indices)

    return interpolated_data, outlier_indices


