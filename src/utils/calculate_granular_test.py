import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller


def check_stationarity(series, verbose=True):
    """
    Perform Augmented Dickey-Fuller test to check for stationarity

    Parameters:
    -----------
    series : pandas.Series
        Time series to test
    verbose : bool
        Whether to print detailed results

    Returns:
    --------
    bool
        True if stationary, False otherwise
    float
        p-value of the test
    """
    result = adfuller(series.dropna())

    if verbose:
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')

    return result[1] < 0.05, result[1]


def make_stationary(series):
    """
    Convert series to stationary by differencing

    Parameters:
    -----------
    series : pandas.Series
        Time series to convert

    Returns:
    --------
    pandas.Series
        Differenced series
    """
    return series.diff().dropna()


def perform_granger_causality(data, target_col, cause_col, max_lags=7):
    """
    Perform Granger Causality test

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the variables
    target_col : str
        Name of the target variable column
    cause_col : str
        Name of the potential causal variable column
    max_lags : int
        Maximum number of lags to test

    Returns:
    --------
    dict
        Dictionary containing test results for each lag
    """
    # Prepare data
    d = pd.DataFrame({
        target_col: data[target_col],
        cause_col: data[cause_col]
    })

    # Run Granger Causality test
    results = grangercausalitytests(d[1:], maxlag=max_lags, verbose=False)

    # Extract p-values for F-test
    p_values = {lag: round(results[lag][0]['ssr_chi2test'][1], 4)
                for lag in range(1, max_lags + 1)}

    return p_values


def analyze_causality(df, target='exchange_rate', features=None,
                      max_lags=7):
    """
    Complete analysis pipeline for Granger Causality

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with all variables
    target : str
        Name of target variable
    features : list
        List of feature names to test for causality
    max_lags : int
        Maximum number of lags to test
    """
    # Process data
    if features is None:
        features = ['gold_price', 'inflation_rate']

    # Check and ensure stationarity for all variables
    variables = [target] + features
    stationary_data = pd.DataFrame(index=df.index)

    for var in variables:
        series = df[var]
        is_stationary, _ = check_stationarity(series, verbose=False)

        if not is_stationary:
            print(f"{var} is not stationary, applying differencing")
            stationary_data[var] = make_stationary(series)
        else:
            stationary_data[var] = series

    # Perform Granger Causality tests
    results = {}
    for feature in features:
        print(f"\nTesting if {feature} Granger-causes {target}:")
        p_values = perform_granger_causality(stationary_data, target, feature, max_lags)

        # Print results
        print("\nResults for different lag periods:")
        for lag, p_value in p_values.items():
            significance = "Significant" if p_value < 0.05 else "Not significant"
            print(f"Lag {lag}: p-value = {p_value} ({significance})")

        results[feature] = p_values

    return results