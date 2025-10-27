"""Automated test data generation utilities."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import random
import string


class TestDataGenerator:
    """Generate realistic test data for portfolio testing."""

    @staticmethod
    def generate_portfolio(n_assets: int = 10, total_value: float = 1000000) -> Dict:
        """Generate random portfolio with realistic holdings."""
        symbols = [TestDataGenerator.generate_symbol() for _ in range(n_assets)]
        weights = np.random.dirichlet(np.ones(n_assets))

        holdings = {}
        prices = {}
        values = {}

        for symbol, weight in zip(symbols, weights):
            price = np.random.uniform(10, 500)
            value = total_value * weight
            quantity = value / price

            holdings[symbol] = quantity
            prices[symbol] = price
            values[symbol] = value

        return {
            'holdings': holdings,
            'prices': prices,
            'values': values,
            'total_value': total_value
        }

    @staticmethod
    def generate_symbol() -> str:
        """Generate realistic stock ticker symbol."""
        length = random.randint(3, 4)
        return ''.join(random.choices(string.ascii_uppercase, k=length))

    @staticmethod
    def generate_price_series(start_price: float = 100, days: int = 252,
                             mu: float = 0.0003, sigma: float = 0.02) -> pd.DataFrame:
        """Generate realistic price series using geometric Brownian motion."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days),
                             end=datetime.now(), freq='D')

        returns = np.random.normal(mu, sigma, days)
        prices = start_price * np.cumprod(1 + returns)

        return pd.DataFrame({
            'date': dates[:len(prices)],
            'close': prices,
            'open': prices * np.random.uniform(0.98, 1.02, len(prices)),
            'high': prices * np.random.uniform(1.00, 1.03, len(prices)),
            'low': prices * np.random.uniform(0.97, 1.00, len(prices)),
            'volume': np.random.randint(1000000, 10000000, len(prices))
        })

    @staticmethod
    def generate_returns(n_days: int = 252, mu: float = 0.0003,
                        sigma: float = 0.02) -> np.ndarray:
        """Generate realistic return series."""
        return np.random.normal(mu, sigma, n_days)

    @staticmethod
    def generate_correlation_matrix(n_assets: int = 10,
                                   avg_correlation: float = 0.3) -> np.ndarray:
        """Generate realistic correlation matrix."""
        # Generate random correlation matrix
        A = np.random.randn(n_assets, n_assets)
        corr = np.corrcoef(A)

        # Adjust to target average correlation
        corr = corr * avg_correlation / corr[np.triu_indices_from(corr, k=1)].mean()
        np.fill_diagonal(corr, 1.0)

        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        eigenvalues = np.maximum(eigenvalues, 0.01)
        corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Normalize
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

        return corr

    @staticmethod
    def generate_options_chain(spot: float = 100, days_to_expiry: int = 30) -> pd.DataFrame:
        """Generate realistic options chain data."""
        strikes = np.arange(spot * 0.8, spot * 1.2, 5)
        chain_data = []

        for strike in strikes:
            intrinsic_call = max(spot - strike, 0)
            intrinsic_put = max(strike - spot, 0)

            time_value = np.random.uniform(2, 10) * np.exp(-days_to_expiry / 365)

            chain_data.append({
                'strike': strike,
                'call_bid': max(0, intrinsic_call + time_value - 0.5),
                'call_ask': intrinsic_call + time_value + 0.5,
                'put_bid': max(0, intrinsic_put + time_value - 0.5),
                'put_ask': intrinsic_put + time_value + 0.5,
                'call_volume': np.random.randint(0, 10000),
                'put_volume': np.random.randint(0, 10000),
                'call_oi': np.random.randint(0, 50000),
                'put_oi': np.random.randint(0, 50000)
            })

        return pd.DataFrame(chain_data)

    @staticmethod
    def generate_benchmark_data(portfolio_returns: np.ndarray,
                               beta: float = 1.0) -> np.ndarray:
        """Generate benchmark returns correlated with portfolio."""
        epsilon = np.random.normal(0, 0.01, len(portfolio_returns))
        benchmark_returns = (portfolio_returns / beta) + epsilon
        return benchmark_returns


# Test the generators
class TestDataGenerators:
    """Tests for data generation utilities."""

    def test_portfolio_generation(self):
        """Test portfolio data generation."""
        portfolio = TestDataGenerator.generate_portfolio(n_assets=5)

        assert len(portfolio['holdings']) == 5
        assert portfolio['total_value'] > 0
        assert abs(sum(portfolio['values'].values()) - portfolio['total_value']) < 0.01

    def test_price_series_generation(self):
        """Test price series generation."""
        prices = TestDataGenerator.generate_price_series(days=100)

        assert len(prices) <= 100
        assert 'close' in prices.columns
        assert all(prices['close'] > 0)
        assert all(prices['high'] >= prices['low'])

    def test_correlation_matrix_generation(self):
        """Test correlation matrix generation."""
        corr = TestDataGenerator.generate_correlation_matrix(n_assets=5)

        assert corr.shape == (5, 5)
        assert np.all(np.diag(corr) == 1.0)
        assert np.all(np.abs(corr) <= 1.0)

        # Check symmetry
        assert np.allclose(corr, corr.T)

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvals(corr)
        assert np.all(eigenvalues >= -1e-10)

    def test_options_chain_generation(self):
        """Test options chain generation."""
        chain = TestDataGenerator.generate_options_chain(spot=100)

        assert len(chain) > 0
        assert 'strike' in chain.columns
        assert all(chain['call_bid'] <= chain['call_ask'])
        assert all(chain['put_bid'] <= chain['put_ask'])

    def test_returns_generation(self):
        """Test returns generation."""
        returns = TestDataGenerator.generate_returns(n_days=252)

        assert len(returns) == 252
        assert isinstance(returns, np.ndarray)

        # Check that mean and std are roughly as expected
        assert abs(np.mean(returns) - 0.0003) < 0.01
        assert abs(np.std(returns) - 0.02) < 0.01
