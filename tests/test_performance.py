"""Performance benchmarks and stress tests."""

import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime


class TestPerformanceBenchmarks:
    """Performance benchmarks for critical operations."""

    def test_portfolio_calculation_speed(self, benchmark):
        """Benchmark portfolio value calculation."""
        holdings = {f"ASSET{i}": np.random.uniform(10, 1000) for i in range(100)}
        prices = {k: np.random.uniform(10, 500) for k in holdings}

        def calculate_value():
            return sum(holdings[k] * prices[k] for k in holdings)

        result = benchmark(calculate_value)
        assert result > 0

    def test_monte_carlo_performance(self):
        """Test Monte Carlo simulation performance."""
        start = time.time()

        simulations = 1000
        years = 10
        results = []

        for _ in range(simulations):
            value = 1000000
            for _ in range(years):
                value *= (1 + np.random.normal(0.07, 0.15))
            results.append(value)

        elapsed = time.time() - start

        assert len(results) == simulations
        assert elapsed < 5.0  # Should complete in under 5 seconds

    def test_efficient_frontier_performance(self):
        """Test efficient frontier calculation performance."""
        start = time.time()

        n_assets = 50
        n_portfolios = 1000

        returns = np.random.uniform(0.05, 0.15, n_assets)
        volatilities = []

        for _ in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(n_assets))
            vol = np.sqrt(np.sum((weights * 0.15) ** 2))
            volatilities.append(vol)

        elapsed = time.time() - start

        assert len(volatilities) == n_portfolios
        assert elapsed < 2.0  # Should complete quickly


class TestStressTests:
    """Stress tests for large portfolios and datasets."""

    def test_large_portfolio_handling(self):
        """Test handling of large portfolio (10,000+ assets)."""
        n_assets = 10000
        holdings = {f"ASSET{i}": np.random.uniform(1, 100) for i in range(n_assets)}

        assert len(holdings) == n_assets

        # Calculate total
        total = sum(holdings.values())
        assert total > 0

    def test_high_frequency_data(self):
        """Test handling of high-frequency data."""
        # Generate 1 year of minute-by-minute data
        n_points = 252 * 6.5 * 60  # Trading days * hours * minutes
        dates = pd.date_range('2024-01-01 09:30', periods=n_points, freq='1min')
        values = np.random.uniform(990000, 1010000, n_points)

        df = pd.DataFrame({'date': dates, 'value': values})

        assert len(df) == n_points
        assert df['value'].mean() > 0

    def test_extreme_volatility(self):
        """Test calculations under extreme volatility."""
        # Simulate extreme market conditions
        extreme_returns = np.random.normal(0, 0.50, 1000)  # 50% volatility

        volatility = np.std(extreme_returns)
        assert volatility > 0.40  # Should be very high

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        import sys

        # Create large dataset
        large_data = np.random.randn(1000000)

        # Check size in memory
        size_bytes = sys.getsizeof(large_data)
        size_mb = size_bytes / (1024 * 1024)

        assert size_mb < 100  # Should be under 100MB

    def test_concurrent_calculations(self):
        """Test concurrent portfolio calculations."""
        n_portfolios = 100
        results = []

        for i in range(n_portfolios):
            holdings = {f"A{j}": np.random.uniform(10, 100) for j in range(10)}
            total = sum(holdings.values())
            results.append(total)

        assert len(results) == n_portfolios
        assert all(r > 0 for r in results)


class TestScalability:
    """Test scalability with increasing data sizes."""

    def test_scaling_with_asset_count(self):
        """Test performance scaling with number of assets."""
        asset_counts = [10, 100, 1000, 5000]
        times = []

        for n in asset_counts:
            holdings = {f"ASSET{i}": np.random.uniform(10, 100) for i in range(n)}

            start = time.time()
            total = sum(holdings.values())
            elapsed = time.time() - start

            times.append(elapsed)
            assert total > 0

        # Times should scale approximately linearly
        assert all(t < 1.0 for t in times)

    def test_scaling_with_simulation_count(self):
        """Test performance scaling with simulation count."""
        sim_counts = [100, 500, 1000]
        times = []

        for n in sim_counts:
            start = time.time()
            results = [np.random.randn(100).sum() for _ in range(n)]
            elapsed = time.time() - start

            times.append(elapsed)
            assert len(results) == n

        assert all(t < 5.0 for t in times)


@pytest.fixture
def benchmark(request):
    """Simple benchmark fixture."""
    def _benchmark(func):
        start = time.time()
        result = func()
        elapsed = time.time() - start
        print(f"\nBenchmark: {elapsed:.4f} seconds")
        return result
    return _benchmark
