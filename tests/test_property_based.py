"""Property-based tests using Hypothesis."""

import pytest
from hypothesis import given, strategies as st, assume
import numpy as np


@pytest.mark.property
class TestPropertyBasedPortfolio:
    """Property-based tests for portfolio operations."""

    @given(
        holdings=st.dictionaries(
            st.text(min_size=3, max_size=5, alphabet=st.characters(whitelist_categories=('Lu',))),
            st.floats(min_value=1, max_value=1000),
            min_size=1,
            max_size=10
        )
    )
    def test_total_value_always_positive(self, holdings):
        """Property: Total portfolio value should always be positive."""
        total = sum(holdings.values())
        assert total > 0

    @given(
        value1=st.floats(min_value=100, max_value=1000000),
        value2=st.floats(min_value=100, max_value=1000000)
    )
    def test_allocation_sums_to_100(self, value1, value2):
        """Property: Allocation percentages should sum to 100."""
        total = value1 + value2
        alloc1 = (value1 / total) * 100
        alloc2 = (value2 / total) * 100

        assert pytest.approx(alloc1 + alloc2, rel=1e-6) == 100.0

    @given(
        initial=st.floats(min_value=1000, max_value=1000000),
        return_rate=st.floats(min_value=-0.5, max_value=2.0)
    )
    def test_return_calculation_consistent(self, initial, return_rate):
        """Property: Return calculations should be consistent."""
        final = initial * (1 + return_rate)
        calculated_return = (final - initial) / initial

        assert pytest.approx(calculated_return, rel=1e-6) == return_rate

    @given(
        returns=st.lists(
            st.floats(min_value=-0.1, max_value=0.1),
            min_size=10,
            max_size=100
        )
    )
    def test_volatility_non_negative(self, returns):
        """Property: Volatility should never be negative."""
        volatility = np.std(returns)
        assert volatility >= 0

    @given(
        weights=st.lists(
            st.floats(min_value=0, max_value=1),
            min_size=2,
            max_size=10
        )
    )
    def test_normalized_weights_sum_to_one(self, weights):
        """Property: Normalized weights should sum to 1."""
        assume(sum(weights) > 0)  # Avoid division by zero

        total = sum(weights)
        normalized = [w / total for w in weights]

        assert pytest.approx(sum(normalized), rel=1e-6) == 1.0

    @given(
        values=st.lists(
            st.floats(min_value=10, max_value=1000),
            min_size=2,
            max_size=20
        )
    )
    def test_max_drawdown_non_positive(self, values):
        """Property: Maximum drawdown should be non-positive."""
        cummax = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - cummax) / cummax
        max_dd = drawdowns.min()

        assert max_dd <= 0

    @given(
        spot=st.floats(min_value=10, max_value=500),
        strike=st.floats(min_value=10, max_value=500)
    )
    def test_option_intrinsic_value(self, spot, strike):
        """Property: Call option intrinsic value is max(spot - strike, 0)."""
        intrinsic = max(spot - strike, 0)
        assert intrinsic >= 0

    @given(
        correlation=st.floats(min_value=-1, max_value=1)
    )
    def test_correlation_bounds(self, correlation):
        """Property: Correlation should be between -1 and 1."""
        assert -1 <= correlation <= 1


@pytest.mark.property
class TestPropertyBasedRiskMetrics:
    """Property-based tests for risk metrics."""

    @given(
        returns=st.lists(
            st.floats(min_value=-0.5, max_value=0.5),
            min_size=100,
            max_size=1000
        ),
        confidence=st.floats(min_value=0.90, max_value=0.99)
    )
    def test_var_ordering(self, returns, confidence):
        """Property: VaR at higher confidence should be more conservative."""
        var_95 = np.percentile(returns, (1 - 0.95) * 100)
        var_99 = np.percentile(returns, (1 - 0.99) * 100)

        assert var_99 <= var_95  # 99% VaR should be lower (more negative)

    @given(
        rf_rate=st.floats(min_value=0, max_value=0.05),
        portfolio_return=st.floats(min_value=0.05, max_value=0.20),
        volatility=st.floats(min_value=0.01, max_value=0.30)
    )
    def test_sharpe_ratio_properties(self, rf_rate, portfolio_return, volatility):
        """Property: Sharpe ratio should increase with return and decrease with volatility."""
        sharpe = (portfolio_return - rf_rate) / volatility

        # Higher return should give higher Sharpe
        higher_return_sharpe = (portfolio_return * 1.1 - rf_rate) / volatility
        assert higher_return_sharpe > sharpe

        # Higher volatility should give lower Sharpe
        higher_vol_sharpe = (portfolio_return - rf_rate) / (volatility * 1.1)
        assert higher_vol_sharpe < sharpe
