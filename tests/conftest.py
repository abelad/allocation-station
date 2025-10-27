"""
Shared pytest fixtures and configuration for all tests.

This file is automatically loaded by pytest and provides common fixtures
and test utilities used across the test suite.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock
import json
import yaml


# ==============================================================================
# Performance Fixtures
# ==============================================================================

@pytest.fixture
def benchmark():
    """
    Simple benchmark fixture for measuring execution time.

    Usage:
        def test_performance(benchmark):
            result = benchmark(lambda: expensive_operation())
    """
    def _benchmark(func):
        start = time.time()
        result = func()
        elapsed = time.time() - start
        print(f"\nBenchmark: {elapsed:.4f} seconds")
        return result
    return _benchmark


@pytest.fixture
def timer():
    """Context manager for timing operations."""
    class Timer:
        def __init__(self):
            self.elapsed = 0

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.elapsed = time.time() - self.start

    return Timer


# ==============================================================================
# Data Generation Fixtures
# ==============================================================================

@pytest.fixture
def sample_portfolio():
    """
    Generate a sample portfolio for testing.

    Returns:
        Dict with holdings, prices, and values
    """
    holdings = {
        'SPY': 100,
        'TLT': 150,
        'GLD': 50,
        'VTI': 75,
        'BND': 200
    }

    prices = {
        'SPY': 442.15,
        'TLT': 92.30,
        'GLD': 186.50,
        'VTI': 225.40,
        'BND': 75.20
    }

    values = {k: holdings[k] * prices[k] for k in holdings}

    return {
        'holdings': holdings,
        'prices': prices,
        'values': values,
        'total_value': sum(values.values())
    }


@pytest.fixture
def price_series_data():
    """
    Generate sample price series data.

    Returns:
        pandas DataFrame with OHLCV data
    """
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')

    # Generate realistic price movements
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.02, 252)
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'date': dates,
        'open': prices * np.random.uniform(0.98, 1.02, 252),
        'high': prices * np.random.uniform(1.01, 1.05, 252),
        'low': prices * np.random.uniform(0.95, 0.99, 252),
        'close': prices,
        'volume': np.random.uniform(1e6, 1e7, 252)
    })

    return df


@pytest.fixture
def returns_data():
    """
    Generate sample returns data.

    Returns:
        numpy array of returns
    """
    np.random.seed(42)
    # Generate 252 days of returns (1 year)
    daily_returns = np.random.normal(0.0003, 0.02, 252)
    return daily_returns


@pytest.fixture
def correlation_matrix():
    """
    Generate a valid correlation matrix.

    Returns:
        numpy array representing correlation matrix
    """
    n_assets = 5
    # Generate random correlation values
    np.random.seed(42)

    # Create a random matrix
    A = np.random.randn(n_assets, n_assets)

    # Make it symmetric and positive semi-definite
    corr = np.dot(A, A.T)

    # Normalize to get correlations
    d = np.sqrt(np.diag(corr))
    corr = corr / d[:, None] / d[None, :]

    return corr


# ==============================================================================
# Mock Objects Fixtures
# ==============================================================================

@pytest.fixture
def mock_broker():
    """
    Mock broker connection for testing.

    Returns:
        Mock broker object with common methods
    """
    broker = Mock()

    # Setup common broker methods
    broker.connect = Mock(return_value=True)
    broker.disconnect = Mock(return_value=True)
    broker.is_connected = Mock(return_value=True)

    # Mock position data
    broker.get_positions = Mock(return_value=[
        {'symbol': 'SPY', 'quantity': 100, 'value': 44215},
        {'symbol': 'TLT', 'quantity': 150, 'value': 13845},
        {'symbol': 'GLD', 'quantity': 50, 'value': 9325}
    ])

    # Mock order placement
    broker.place_order = Mock(return_value={
        'order_id': 'ORD123456',
        'status': 'filled',
        'filled_price': 442.15,
        'filled_quantity': 10
    })

    # Mock market data
    broker.get_quote = Mock(return_value={
        'bid': 441.90,
        'ask': 442.40,
        'last': 442.15,
        'volume': 65432100
    })

    return broker


@pytest.fixture
def mock_data_provider():
    """
    Mock market data provider.

    Returns:
        Mock data provider with common methods
    """
    provider = Mock()

    # Historical data
    provider.get_historical_prices = Mock(return_value=pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30),
        'close': np.random.uniform(100, 110, 30)
    }))

    # Real-time data
    provider.get_realtime_quote = Mock(return_value={
        'symbol': 'SPY',
        'price': 442.15,
        'change': 2.35,
        'change_percent': 0.53
    })

    return provider


# ==============================================================================
# File System Fixtures
# ==============================================================================

@pytest.fixture
def temp_directory():
    """
    Create a temporary directory for testing.

    Yields:
        Path to temporary directory (cleaned up after test)
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_csv_file(temp_directory):
    """
    Create a sample CSV file for testing.

    Returns:
        Path to CSV file
    """
    csv_path = temp_directory / "test_data.csv"

    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'value': np.random.uniform(100000, 110000, 10)
    })

    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_json_file(temp_directory):
    """
    Create a sample JSON file for testing.

    Returns:
        Path to JSON file
    """
    json_path = temp_directory / "test_config.json"

    data = {
        'portfolio': {
            'name': 'Test Portfolio',
            'holdings': ['SPY', 'TLT', 'GLD'],
            'target_allocation': {'stocks': 60, 'bonds': 40}
        }
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    return json_path


# ==============================================================================
# Monte Carlo Fixtures
# ==============================================================================

@pytest.fixture
def monte_carlo_params():
    """
    Standard Monte Carlo simulation parameters.

    Returns:
        Dict with simulation parameters
    """
    return {
        'initial_value': 1000000,
        'annual_return': 0.07,
        'annual_volatility': 0.15,
        'years': 30,
        'simulations': 1000,
        'withdrawal_rate': 0.04
    }


@pytest.fixture
def monte_carlo_results():
    """
    Generate sample Monte Carlo results for testing.

    Returns:
        Dict with simulation results
    """
    np.random.seed(42)
    simulations = 100
    years = 10

    # Generate random walk paths
    paths = []
    for _ in range(simulations):
        returns = np.random.normal(0.07, 0.15, years)
        values = 1000000 * np.cumprod(1 + returns)
        paths.append(values)

    paths = np.array(paths)

    return {
        'paths': paths,
        'mean_final': np.mean(paths[:, -1]),
        'median_final': np.median(paths[:, -1]),
        'percentile_10': np.percentile(paths[:, -1], 10),
        'percentile_90': np.percentile(paths[:, -1], 90),
        'success_rate': np.mean(paths[:, -1] > 0)
    }


# ==============================================================================
# Database Fixtures
# ==============================================================================

@pytest.fixture
def mock_database():
    """
    Mock database connection for testing.

    Returns:
        Mock database with common methods
    """
    db = Mock()

    # Connection methods
    db.connect = Mock(return_value=True)
    db.disconnect = Mock(return_value=True)
    db.is_connected = Mock(return_value=True)

    # CRUD operations
    db.save_portfolio = Mock(return_value={'id': 'PORT123', 'success': True})
    db.load_portfolio = Mock(return_value={
        'id': 'PORT123',
        'name': 'Test Portfolio',
        'total_value': 1000000,
        'holdings': {'SPY': 100, 'TLT': 150}
    })
    db.delete_portfolio = Mock(return_value={'success': True})

    # Query methods
    db.query_portfolios = Mock(return_value=[
        {'id': 'PORT123', 'name': 'Portfolio 1'},
        {'id': 'PORT456', 'name': 'Portfolio 2'}
    ])

    return db


# ==============================================================================
# Configuration Fixtures
# ==============================================================================

@pytest.fixture
def test_config():
    """
    Test configuration settings.

    Returns:
        Dict with test configuration
    """
    return {
        'portfolio': {
            'rebalance_threshold': 0.05,
            'min_position_size': 0.01,
            'max_position_size': 0.40,
            'target_cash': 0.02
        },
        'risk': {
            'max_drawdown': 0.20,
            'var_confidence': 0.95,
            'stress_test_scenarios': ['2008_crisis', 'covid_2020', 'dotcom_burst']
        },
        'optimization': {
            'method': 'mean_variance',
            'constraints': ['long_only', 'sum_to_one'],
            'risk_free_rate': 0.03
        }
    }


# ==============================================================================
# Test Markers Configuration
# ==============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for workflows"
    )
    config.addinivalue_line(
        "markers", "performance: Performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", "stress: Stress tests for large datasets"
    )
    config.addinivalue_line(
        "markers", "property: Property-based tests"
    )
    config.addinivalue_line(
        "markers", "regression: Regression tests for bug fixes"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests"
    )
    config.addinivalue_line(
        "markers", "requires_network: Tests requiring network access"
    )


# ==============================================================================
# Pytest Hooks
# ==============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "test_performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        if "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
    # Cleanup if needed


# ==============================================================================
# Assertion Helpers
# ==============================================================================

@pytest.fixture
def assert_dataframe_equal():
    """Helper to assert DataFrames are equal."""
    def _assert_equal(df1, df2, **kwargs):
        pd.testing.assert_frame_equal(df1, df2, **kwargs)
    return _assert_equal


@pytest.fixture
def assert_array_equal():
    """Helper to assert arrays are equal."""
    def _assert_equal(arr1, arr2, **kwargs):
        np.testing.assert_array_equal(arr1, arr2, **kwargs)
    return _assert_equal


@pytest.fixture
def assert_close():
    """Helper for approximate equality."""
    def _assert_close(actual, expected, rel_tol=1e-5, abs_tol=1e-9):
        assert abs(actual - expected) <= max(rel_tol * abs(expected), abs_tol)
    return _assert_close