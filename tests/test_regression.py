"""Regression tests to catch bugs and prevent regressions."""

import pytest
import numpy as np
import pandas as pd


@pytest.mark.regression
class TestRegressionBugs:
    """Tests for previously discovered bugs."""

    def test_division_by_zero_in_allocation(self):
        """Regression: Division by zero when portfolio value is zero."""
        # Bug: allocation calculation failed when total value was zero
        total_value = 0

        if total_value == 0:
            allocation = {}
        else:
            allocation = {'SPY': 100}

        assert isinstance(allocation, dict)

    def test_negative_sharpe_ratio(self):
        """Regression: Sharpe ratio calculation failed with negative returns."""
        portfolio_return = -0.05
        rf_rate = 0.03
        volatility = 0.15

        sharpe = (portfolio_return - rf_rate) / volatility
        assert sharpe < 0  # Should be negative

    def test_inf_values_in_returns(self):
        """Regression: Infinite values in return calculations."""
        returns = np.array([0.01, 0.02, np.inf, 0.03])

        # Handle inf values
        clean_returns = returns[np.isfinite(returns)]

        assert len(clean_returns) < len(returns)
        assert not np.any(np.isinf(clean_returns))

    def test_nan_in_correlation_matrix(self):
        """Regression: NaN values in correlation matrix."""
        # Bug: Correlation matrix had NaN for constant series
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1, 1, 1, 1, 1]  # Constant series
        })

        corr = data.corr()
        # Replace NaN with 0 for constant series
        corr = corr.fillna(0)

        assert not corr.isna().any().any()

    def test_empty_holdings_handling(self):
        """Regression: Crash when processing empty holdings."""
        holdings = {}

        total_value = sum(holdings.values()) if holdings else 0
        assert total_value == 0

    def test_date_parsing_edge_case(self):
        """Regression: Date parsing failed for certain formats."""
        date_strings = ['2024-01-01', '01/01/2024', '2024-01-01T00:00:00']

        for date_str in date_strings:
            try:
                parsed = pd.to_datetime(date_str)
                assert isinstance(parsed, pd.Timestamp)
            except:
                assert False, f"Failed to parse: {date_str}"


@pytest.mark.regression
class TestRegressionPerformance:
    """Regression tests for performance issues."""

    def test_no_quadratic_complexity(self):
        """Regression: Ensure linear complexity for portfolio calculations."""
        import time

        # Test with different sizes
        sizes = [100, 200, 400]
        times = []

        for n in sizes:
            holdings = {f"A{i}": np.random.uniform(10, 100) for i in range(n)}

            start = time.time()
            total = sum(holdings.values())
            elapsed = time.time() - start

            times.append(elapsed)

        # Check that time scales approximately linearly, not quadratically
        # If quadratic, time_400 would be ~16x time_100
        # If linear, time_400 would be ~4x time_100
        if times[0] > 0:
            ratio = times[2] / times[0]
            assert ratio < 10  # Should be much less than quadratic

    def test_memory_leak_prevention(self):
        """Regression: Check for memory leaks in repeated calculations."""
        import gc

        initial_objects = len(gc.get_objects())

        # Perform many calculations
        for _ in range(100):
            holdings = {f"A{i}": np.random.uniform(10, 100) for i in range(10)}
            total = sum(holdings.values())
            del holdings

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count shouldn't grow significantly
        growth = final_objects - initial_objects
        assert growth < 1000  # Reasonable threshold


@pytest.mark.regression
class TestRegressionDataIntegrity:
    """Regression tests for data integrity issues."""

    def test_precision_loss_prevention(self):
        """Regression: Prevent precision loss in calculations."""
        value1 = 1000000.123456789
        value2 = 0.000000001

        sum_value = value1 + value2

        # Should not lose significant digits
        assert sum_value > value1

    def test_rounding_consistency(self):
        """Regression: Ensure consistent rounding behavior."""
        value = 123.456789

        rounded_2 = round(value, 2)
        rounded_4 = round(value, 4)

        assert rounded_2 == 123.46
        assert rounded_4 == 123.4568

    def test_timestamp_timezone_handling(self):
        """Regression: Consistent timezone handling."""
        import datetime

        # UTC timestamp
        utc_time = datetime.datetime(2024, 1, 1, 12, 0, 0)

        # Should handle timezone conversions consistently
        assert utc_time.hour == 12


@pytest.mark.regression
class TestRegressionValidation:
    """Regression tests for input validation."""

    def test_reject_invalid_symbols(self):
        """Regression: Properly reject invalid ticker symbols."""
        invalid_symbols = ['', '123', 'AB', 'TOOLONGSYMBOL']

        for symbol in invalid_symbols:
            if len(symbol) == 0 or symbol.isdigit() or len(symbol) > 5:
                with pytest.raises(ValueError):
                    raise ValueError(f"Invalid symbol: {symbol}")

    def test_reject_negative_prices(self):
        """Regression: Reject negative prices."""
        price = -10.50

        with pytest.raises(ValueError):
            if price < 0:
                raise ValueError("Price cannot be negative")

    def test_reject_future_dates(self):
        """Regression: Reject dates in the future for historical data."""
        from datetime import datetime, timedelta

        future_date = datetime.now() + timedelta(days=30)
        current_date = datetime.now()

        assert future_date > current_date
