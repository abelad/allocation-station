"""
Performance Attribution Analysis

This module provides comprehensive performance attribution capabilities for analyzing
portfolio returns relative to benchmarks and understanding the sources of performance.

Key Features:
- Brinson attribution (allocation, selection, interaction effects)
- Factor-based attribution (Fama-French, custom factors)
- Contribution analysis (asset-level performance contribution)
- Risk-adjusted attribution (Sharpe, information ratio decomposition)
- Benchmark-relative attribution
- Time-weighted vs money-weighted return calculations
- Custom attribution models

Classes:
    BrinsonAttribution: Multi-period Brinson-Fachler attribution analysis
    FactorAttribution: Factor-based performance attribution
    ContributionAnalyzer: Asset-level contribution analysis
    RiskAdjustedAttribution: Risk-adjusted performance metrics and attribution
    BenchmarkAttribution: Benchmark-relative performance analysis
    ReturnCalculator: Time-weighted and money-weighted return calculations
    CustomAttributionModel: Framework for user-defined attribution models
"""

from typing import Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class AttributionMethod(str, Enum):
    """Attribution calculation methods."""
    BRINSON_HOOD_BEEBOWER = "brinson_hood_beebower"  # Original Brinson (geometric)
    BRINSON_FACHLER = "brinson_fachler"  # Arithmetic attribution
    GEOMETRIC = "geometric"  # Geometric linking
    ARITHMETIC = "arithmetic"  # Arithmetic attribution


class FactorModel(str, Enum):
    """Factor model types."""
    FAMA_FRENCH_3 = "fama_french_3"  # Market, Size, Value
    FAMA_FRENCH_5 = "fama_french_5"  # + Profitability, Investment
    CARHART_4 = "carhart_4"  # FF3 + Momentum
    CUSTOM = "custom"  # User-defined factors


class ReturnMethod(str, Enum):
    """Return calculation methods."""
    TIME_WEIGHTED = "time_weighted"  # TWR - geometric linking
    MONEY_WEIGHTED = "money_weighted"  # MWR - IRR
    SIMPLE = "simple"  # Simple return
    LOG = "log"  # Logarithmic return


class AttributionResult(BaseModel):
    """Results from attribution analysis."""
    total_return: float = Field(description="Total portfolio return")
    benchmark_return: float = Field(description="Benchmark return")
    active_return: float = Field(description="Active return (portfolio - benchmark)")
    components: Dict[str, float] = Field(description="Attribution components")
    by_sector: Optional[pd.DataFrame] = Field(None, description="Sector-level attribution")

    class Config:
        arbitrary_types_allowed = True


class BrinsonAttribution:
    """
    Brinson attribution analysis for understanding sources of active return.

    Decomposes active return into:
    - Allocation Effect: Impact of sector weight decisions
    - Selection Effect: Impact of security selection within sectors
    - Interaction Effect: Combined impact of allocation and selection

    Supports both single-period and multi-period attribution with geometric linking.
    """

    def __init__(self, method: AttributionMethod = AttributionMethod.BRINSON_FACHLER):
        """
        Initialize Brinson attribution analyzer.

        Args:
            method: Attribution calculation method (Brinson-Fachler or Hood-Beebower)
        """
        self.method = method

    def calculate_single_period(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> AttributionResult:
        """
        Calculate single-period Brinson attribution.

        Args:
            portfolio_weights: Portfolio weights by sector/asset
            benchmark_weights: Benchmark weights by sector/asset
            portfolio_returns: Returns of portfolio holdings
            benchmark_returns: Returns of benchmark components

        Returns:
            AttributionResult with allocation, selection, and interaction effects
        """
        # Ensure indices match
        sectors = portfolio_weights.index.union(benchmark_weights.index)
        pw = portfolio_weights.reindex(sectors, fill_value=0.0)
        bw = benchmark_weights.reindex(sectors, fill_value=0.0)
        pr = portfolio_returns.reindex(sectors, fill_value=0.0)
        br = benchmark_returns.reindex(sectors, fill_value=0.0)

        # Calculate total returns
        portfolio_return = (pw * pr).sum()
        benchmark_return = (bw * br).sum()
        active_return = portfolio_return - benchmark_return

        if self.method == AttributionMethod.BRINSON_FACHLER:
            # Brinson-Fachler (arithmetic)
            # Allocation = (wp - wb) * (rb - R_b)
            allocation = (pw - bw) * (br - benchmark_return)

            # Selection = wb * (rp - rb)
            selection = bw * (pr - br)

            # Interaction = (wp - wb) * (rp - rb)
            interaction = (pw - bw) * (pr - br)

        else:  # BRINSON_HOOD_BEEBOWER
            # Hood-Beebower (geometric approximation)
            allocation = (pw - bw) * br
            selection = bw * (pr - br)
            interaction = (pw - bw) * (pr - br)

        # Create detailed sector-level DataFrame
        sector_attribution = pd.DataFrame({
            'portfolio_weight': pw,
            'benchmark_weight': bw,
            'portfolio_return': pr,
            'benchmark_return': br,
            'allocation_effect': allocation,
            'selection_effect': selection,
            'interaction_effect': interaction,
            'total_effect': allocation + selection + interaction,
        })

        components = {
            'allocation': allocation.sum(),
            'selection': selection.sum(),
            'interaction': interaction.sum(),
        }

        return AttributionResult(
            total_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            components=components,
            by_sector=sector_attribution,
        )

    def calculate_multi_period(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
    ) -> Dict[str, Union[AttributionResult, pd.DataFrame]]:
        """
        Calculate multi-period Brinson attribution with geometric linking.

        Args:
            portfolio_weights: DataFrame with dates as index, sectors as columns
            benchmark_weights: DataFrame with dates as index, sectors as columns
            portfolio_returns: DataFrame with dates as index, sectors as columns
            benchmark_returns: DataFrame with dates as index, sectors as columns

        Returns:
            Dictionary with period-by-period results and linked total attribution
        """
        periods = portfolio_weights.index
        period_results = {}

        # Calculate attribution for each period
        for date in periods:
            result = self.calculate_single_period(
                portfolio_weights.loc[date],
                benchmark_weights.loc[date],
                portfolio_returns.loc[date],
                benchmark_returns.loc[date],
            )
            period_results[date] = result

        # Geometric linking of attribution effects
        linked_result = self._geometric_linking(period_results)

        # Create summary DataFrame
        summary = pd.DataFrame([
            {
                'date': date,
                'portfolio_return': result.total_return,
                'benchmark_return': result.benchmark_return,
                'active_return': result.active_return,
                **result.components,
            }
            for date, result in period_results.items()
        ]).set_index('date')

        return {
            'period_results': period_results,
            'linked_result': linked_result,
            'summary': summary,
        }

    def _geometric_linking(self, period_results: Dict) -> AttributionResult:
        """
        Geometrically link multi-period attribution effects.

        Uses the Menchero (2000) method for geometric attribution linking.
        """
        # Calculate cumulative returns
        portfolio_returns = [r.total_return for r in period_results.values()]
        benchmark_returns = [r.benchmark_return for r in period_results.values()]

        cumulative_portfolio = np.prod([1 + r for r in portfolio_returns]) - 1
        cumulative_benchmark = np.prod([1 + r for r in benchmark_returns]) - 1
        cumulative_active = cumulative_portfolio - cumulative_benchmark

        # Approximate geometric linking of components
        # (Simplified - full Menchero method is more complex)
        allocation_effects = [r.components['allocation'] for r in period_results.values()]
        selection_effects = [r.components['selection'] for r in period_results.values()]
        interaction_effects = [r.components['interaction'] for r in period_results.values()]

        # Geometric compounding approximation
        linked_allocation = sum(allocation_effects)
        linked_selection = sum(selection_effects)
        linked_interaction = sum(interaction_effects)

        components = {
            'allocation': linked_allocation,
            'selection': linked_selection,
            'interaction': linked_interaction,
        }

        return AttributionResult(
            total_return=cumulative_portfolio,
            benchmark_return=cumulative_benchmark,
            active_return=cumulative_active,
            components=components,
        )


class FactorAttribution:
    """
    Factor-based performance attribution using multi-factor models.

    Attributes returns to systematic factor exposures (beta) and
    security-specific alpha.
    """

    def __init__(
        self,
        factor_model: FactorModel = FactorModel.FAMA_FRENCH_3,
        custom_factors: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize factor attribution analyzer.

        Args:
            factor_model: Pre-defined factor model to use
            custom_factors: DataFrame of custom factor returns (if CUSTOM model)
        """
        self.factor_model = factor_model
        self.custom_factors = custom_factors
        self.factor_loadings = None
        self.alpha = None

    def analyze(
        self,
        portfolio_returns: pd.Series,
        factor_returns: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, Union[float, pd.Series, pd.DataFrame]]:
        """
        Perform factor attribution analysis.

        Args:
            portfolio_returns: Time series of portfolio returns
            factor_returns: DataFrame with factor returns (dates x factors)
            risk_free_rate: Risk-free rate for excess returns

        Returns:
            Dictionary with alpha, factor loadings, and attribution results
        """
        if factor_returns is None:
            if self.custom_factors is None:
                raise ValueError("Must provide factor_returns or set custom_factors")
            factor_returns = self.custom_factors

        # Align dates
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        port_returns = portfolio_returns.loc[common_dates]
        fact_returns = factor_returns.loc[common_dates]

        # Calculate excess returns
        excess_portfolio = port_returns - risk_free_rate

        # Regression: R_p - R_f = alpha + beta_1*F_1 + ... + beta_k*F_k + epsilon
        X = fact_returns.values
        y = excess_portfolio.values

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # OLS regression
        try:
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            raise ValueError("Factor regression failed - check for multicollinearity")

        alpha = coefficients[0]
        betas = pd.Series(coefficients[1:], index=fact_returns.columns)

        # Calculate factor contributions
        factor_contributions = fact_returns.multiply(betas, axis=1)

        # Store results
        self.alpha = alpha
        self.factor_loadings = betas

        # Attribution by factor
        total_factor_contribution = factor_contributions.sum(axis=1)
        residual = excess_portfolio - alpha - total_factor_contribution

        # Summary statistics
        avg_factor_contrib = factor_contributions.mean()

        # R-squared
        ss_total = np.sum((y - np.mean(y)) ** 2)
        predictions = X_with_intercept @ coefficients
        ss_residual = np.sum((y - predictions) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

        # Decompose average return
        avg_return = port_returns.mean()
        avg_alpha = alpha
        avg_factor_returns = avg_factor_contrib.sum()
        avg_residual = residual.mean()

        return {
            'alpha': alpha,
            'factor_loadings': betas,
            'r_squared': r_squared,
            'factor_contributions': factor_contributions,
            'factor_contribution_summary': avg_factor_contrib,
            'residuals': residual,
            'return_decomposition': {
                'average_return': avg_return,
                'alpha': avg_alpha,
                'factor_contribution': avg_factor_returns,
                'residual': avg_residual,
            },
        }

    def decompose_active_return(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Decompose active return into factor tilts and selection skill.

        Active return = (beta_p - beta_b) * factor_returns + (alpha_p - alpha_b)
        """
        # Analyze portfolio
        port_analysis = self.analyze(portfolio_returns, factor_returns)

        # Analyze benchmark
        bench_analysis = self.analyze(benchmark_returns, factor_returns)

        # Calculate active exposures
        active_betas = port_analysis['factor_loadings'] - bench_analysis['factor_loadings']

        # Active alpha
        active_alpha = port_analysis['alpha'] - bench_analysis['alpha']

        # Factor timing contribution
        factor_timing = (active_betas * factor_returns.mean()).sum()

        # Total active return
        avg_active = (portfolio_returns - benchmark_returns).mean()

        return {
            'active_return': avg_active,
            'active_alpha': active_alpha,
            'active_betas': active_betas,
            'factor_timing_contribution': factor_timing,
            'unexplained': avg_active - active_alpha - factor_timing,
        }


class ContributionAnalyzer:
    """
    Analyzes the contribution of individual assets to portfolio performance.

    Calculates both absolute and relative contributions, accounting for
    weights and returns.
    """

    def calculate_contributions(
        self,
        weights: pd.Series,
        returns: pd.Series,
        beginning_weights: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Calculate asset-level performance contributions.

        Args:
            weights: Current/ending weights
            returns: Period returns for each asset
            beginning_weights: Beginning-of-period weights (if different)

        Returns:
            DataFrame with contribution analysis
        """
        if beginning_weights is None:
            beginning_weights = weights

        # Average weight for contribution calculation
        avg_weights = (beginning_weights + weights) / 2

        # Contribution to return
        contribution = avg_weights * returns

        # Portfolio return
        portfolio_return = contribution.sum()

        # Percentage contribution
        if portfolio_return != 0:
            pct_contribution = contribution / portfolio_return * 100
        else:
            pct_contribution = pd.Series(0, index=contribution.index)

        # Create results DataFrame
        results = pd.DataFrame({
            'beginning_weight': beginning_weights,
            'ending_weight': weights,
            'average_weight': avg_weights,
            'return': returns,
            'contribution': contribution,
            'pct_contribution': pct_contribution,
        })

        # Sort by absolute contribution
        results = results.sort_values('contribution', ascending=False, key=abs)

        return results

    def calculate_risk_contribution(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate asset-level risk (volatility) contributions.

        Risk contribution = (weight * marginal risk) / portfolio risk
        where marginal risk = (Σ * w)_i
        """
        w = weights.values
        cov = covariance_matrix.values

        # Portfolio variance
        portfolio_variance = w @ cov @ w
        portfolio_vol = np.sqrt(portfolio_variance)

        # Marginal contribution to risk
        marginal_risk = cov @ w

        # Risk contribution
        risk_contribution = weights.values * marginal_risk

        # Percentage risk contribution
        pct_risk_contribution = risk_contribution / portfolio_vol if portfolio_vol > 0 else np.zeros_like(risk_contribution)

        results = pd.DataFrame({
            'weight': weights,
            'marginal_risk': marginal_risk,
            'risk_contribution': risk_contribution,
            'pct_risk_contribution': pct_risk_contribution * 100,
        }, index=weights.index)

        results = results.sort_values('risk_contribution', ascending=False)

        return results

    def analyze_top_contributors(
        self,
        weights: pd.Series,
        returns: pd.Series,
        top_n: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        """
        Identify top positive and negative contributors.

        Args:
            weights: Asset weights
            returns: Asset returns
            top_n: Number of top contributors to show

        Returns:
            Dictionary with top winners and losers
        """
        contributions = self.calculate_contributions(weights, returns)

        top_winners = contributions.nlargest(top_n, 'contribution')
        top_losers = contributions.nsmallest(top_n, 'contribution')

        return {
            'top_winners': top_winners,
            'top_losers': top_losers,
            'all_contributions': contributions,
        }


class RiskAdjustedAttribution:
    """
    Risk-adjusted performance attribution.

    Decomposes risk-adjusted metrics (Sharpe ratio, Information ratio)
    into components to understand sources of risk-adjusted returns.
    """

    def sharpe_decomposition(
        self,
        portfolio_returns: pd.Series,
        sector_weights: pd.Series,
        sector_returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Decompose portfolio Sharpe ratio by sector contributions.

        Args:
            portfolio_returns: Portfolio return time series
            sector_weights: Weights allocated to each sector
            sector_returns: Return time series for each sector
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary with Sharpe decomposition
        """
        # Portfolio Sharpe ratio
        excess_returns = portfolio_returns - risk_free_rate
        sharpe = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # Sector Sharpe ratios
        sector_sharpes = {}
        for sector in sector_returns.columns:
            sector_excess = sector_returns[sector] - risk_free_rate
            sector_sharpe = sector_excess.mean() / sector_excess.std() if sector_excess.std() > 0 else 0
            sector_sharpes[sector] = sector_sharpe

        sector_sharpes = pd.Series(sector_sharpes)

        # Approximate Sharpe contribution (weighted average approximation)
        sharpe_contribution = sector_weights * sector_sharpes

        return {
            'portfolio_sharpe': sharpe,
            'sector_sharpes': sector_sharpes,
            'sector_weights': sector_weights,
            'sharpe_contribution': sharpe_contribution,
            'weighted_avg_sharpe': sharpe_contribution.sum(),
        }

    def information_ratio_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        asset_weights: pd.Series,
        asset_returns: pd.DataFrame,
        benchmark_weights: pd.Series,
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Attribute Information Ratio to individual positions.

        IR = Active Return / Tracking Error
        """
        # Active returns
        active_returns = portfolio_returns - benchmark_returns

        # Information ratio
        tracking_error = active_returns.std()
        information_ratio = active_returns.mean() / tracking_error if tracking_error > 0 else 0

        # Asset-level active returns
        active_weights = asset_weights - benchmark_weights.reindex(asset_weights.index, fill_value=0)

        # Contribution to active return
        asset_active_returns = asset_returns.sub(benchmark_returns, axis=0)
        avg_contribution = (active_weights * asset_active_returns.mean()).sort_values(ascending=False)

        # Contribution to tracking error (approximate)
        asset_te_contribution = (active_weights.abs() * asset_returns.std()).sort_values(ascending=False)

        return {
            'information_ratio': information_ratio,
            'active_return': active_returns.mean(),
            'tracking_error': tracking_error,
            'asset_active_contributions': avg_contribution,
            'asset_te_contributions': asset_te_contribution,
        }


class BenchmarkAttribution:
    """
    Benchmark-relative performance attribution analysis.

    Comprehensive analysis of performance vs. benchmark including
    tracking error decomposition and relative risk metrics.
    """

    def analyze(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_weights: Optional[pd.Series] = None,
        benchmark_weights: Optional[pd.Series] = None,
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Comprehensive benchmark-relative analysis.

        Args:
            portfolio_returns: Portfolio return time series
            benchmark_returns: Benchmark return time series
            portfolio_weights: Current portfolio weights (optional)
            benchmark_weights: Benchmark weights (optional)

        Returns:
            Dictionary with relative performance metrics
        """
        # Align returns
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        port_ret = portfolio_returns.loc[common_dates]
        bench_ret = benchmark_returns.loc[common_dates]

        # Active returns
        active_returns = port_ret - bench_ret

        # Cumulative returns
        cum_portfolio = (1 + port_ret).cumprod() - 1
        cum_benchmark = (1 + bench_ret).cumprod() - 1
        cum_active = cum_portfolio - cum_benchmark

        # Performance metrics
        avg_active_return = active_returns.mean()
        tracking_error = active_returns.std()
        information_ratio = avg_active_return / tracking_error if tracking_error > 0 else 0

        # Up/Down capture
        up_periods = bench_ret > 0
        down_periods = bench_ret < 0

        up_capture = port_ret[up_periods].mean() / bench_ret[up_periods].mean() if up_periods.sum() > 0 else np.nan
        down_capture = port_ret[down_periods].mean() / bench_ret[down_periods].mean() if down_periods.sum() > 0 else np.nan

        # Beta
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        benchmark_variance = bench_ret.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        # Correlation
        correlation = port_ret.corr(bench_ret)

        # Hit rate (percentage of periods outperforming)
        hit_rate = (active_returns > 0).sum() / len(active_returns)

        results = {
            'total_portfolio_return': cum_portfolio.iloc[-1] if len(cum_portfolio) > 0 else 0,
            'total_benchmark_return': cum_benchmark.iloc[-1] if len(cum_benchmark) > 0 else 0,
            'total_active_return': cum_active.iloc[-1] if len(cum_active) > 0 else 0,
            'avg_active_return': avg_active_return,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'correlation': correlation,
            'up_capture': up_capture,
            'down_capture': down_capture,
            'hit_rate': hit_rate,
            'active_returns_series': active_returns,
            'cumulative_active': cum_active,
        }

        # Add weight-based analysis if weights provided
        if portfolio_weights is not None and benchmark_weights is not None:
            active_weights = portfolio_weights - benchmark_weights.reindex(portfolio_weights.index, fill_value=0)
            active_weight_sum = active_weights.abs().sum() / 2  # Active share
            results['active_share'] = active_weight_sum
            results['active_weights'] = active_weights

        return results

    def tracking_error_decomposition(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Decompose tracking error into systematic and specific components.

        TE² = TE²_systematic + TE²_specific
        """
        # Active returns
        active_returns = portfolio_returns - benchmark_returns
        total_te = active_returns.std()

        # Regress active returns on factors
        common_dates = active_returns.index.intersection(factor_returns.index)
        y = active_returns.loc[common_dates].values
        X = factor_returns.loc[common_dates].values

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # Regression
        coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        predictions = X_with_intercept @ coefficients
        residuals = y - predictions

        # Systematic TE (explained by factors)
        systematic_te = np.std(predictions)

        # Specific TE (unexplained)
        specific_te = np.std(residuals)

        return {
            'total_tracking_error': total_te,
            'systematic_tracking_error': systematic_te,
            'specific_tracking_error': specific_te,
            'systematic_pct': systematic_te / total_te * 100 if total_te > 0 else 0,
            'specific_pct': specific_te / total_te * 100 if total_te > 0 else 0,
        }


class ReturnCalculator:
    """
    Calculate returns using different methodologies.

    Supports time-weighted returns (TWR), money-weighted returns (MWR/IRR),
    and handles cash flows correctly.
    """

    def time_weighted_return(
        self,
        portfolio_values: pd.Series,
        cash_flows: Optional[pd.Series] = None,
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Calculate time-weighted return (geometric linking).

        TWR eliminates the impact of cash flows timing, measuring
        manager skill independent of investor cash flow decisions.

        Args:
            portfolio_values: Portfolio values at each date
            cash_flows: Cash flows (contributions positive, withdrawals negative)

        Returns:
            Dictionary with TWR and period returns
        """
        if cash_flows is None:
            cash_flows = pd.Series(0, index=portfolio_values.index)

        # Align dates
        dates = portfolio_values.index
        values = portfolio_values.values
        flows = cash_flows.reindex(dates, fill_value=0).values

        # Calculate sub-period returns
        period_returns = []
        for i in range(1, len(dates)):
            # Adjust beginning value for cash flows
            # Assumption: cash flows occur at end of period
            beginning_value = values[i-1]
            ending_value = values[i]
            period_flow = flows[i]

            # Return for the period
            if beginning_value > 0:
                period_return = (ending_value - period_flow - beginning_value) / beginning_value
            else:
                period_return = 0.0

            period_returns.append(period_return)

        period_returns = pd.Series(period_returns, index=dates[1:])

        # Geometric linking
        twr = np.prod([1 + r for r in period_returns]) - 1

        # Annualized return
        n_years = (dates[-1] - dates[0]).days / 365.25
        annualized_twr = (1 + twr) ** (1 / n_years) - 1 if n_years > 0 else twr

        return {
            'time_weighted_return': twr,
            'annualized_twr': annualized_twr,
            'period_returns': period_returns,
        }

    def money_weighted_return(
        self,
        portfolio_values: pd.Series,
        cash_flows: pd.Series,
    ) -> Dict[str, float]:
        """
        Calculate money-weighted return (Internal Rate of Return).

        MWR/IRR accounts for the timing and size of cash flows,
        representing the actual investor experience.

        Args:
            portfolio_values: Portfolio values at each date
            cash_flows: Cash flows at each date

        Returns:
            Dictionary with MWR/IRR
        """
        from scipy.optimize import newton

        dates = portfolio_values.index
        values = portfolio_values.values
        flows = cash_flows.reindex(dates, fill_value=0).values

        # Convert dates to time in years from start
        start_date = dates[0]
        times = np.array([(d - start_date).days / 365.25 for d in dates])

        # Initial investment (negative of first value)
        initial_investment = -values[0]

        # IRR equation: sum of PV of cash flows = 0
        # NPV(r) = initial_investment + sum(CF_t / (1+r)^t) + final_value / (1+r)^T = 0

        def npv(rate):
            """Net present value function."""
            pv_flows = initial_investment
            for i in range(1, len(times)):
                # Cash flow at time i
                cf = flows[i]
                pv_flows += cf / ((1 + rate) ** times[i])
            # Final value
            pv_flows += values[-1] / ((1 + rate) ** times[-1])
            return pv_flows

        def npv_derivative(rate):
            """Derivative of NPV for Newton's method."""
            deriv = 0
            for i in range(1, len(times)):
                cf = flows[i]
                deriv -= cf * times[i] / ((1 + rate) ** (times[i] + 1))
            deriv -= values[-1] * times[-1] / ((1 + rate) ** (times[-1] + 1))
            return deriv

        # Solve for IRR using Newton's method
        try:
            irr = newton(npv, x0=0.1, fprime=npv_derivative, maxiter=100, tol=1e-6)
        except (RuntimeError, ValueError):
            # Fallback to simple approximation if Newton fails
            irr = (values[-1] / values[0]) ** (1 / times[-1]) - 1 if times[-1] > 0 else 0

        return {
            'money_weighted_return': irr,
            'irr': irr,
        }

    def compare_returns(
        self,
        portfolio_values: pd.Series,
        cash_flows: pd.Series,
    ) -> pd.DataFrame:
        """
        Compare TWR and MWR to understand impact of cash flow timing.

        Args:
            portfolio_values: Portfolio values over time
            cash_flows: Cash flows over time

        Returns:
            DataFrame comparing different return calculations
        """
        twr_result = self.time_weighted_return(portfolio_values, cash_flows)
        mwr_result = self.money_weighted_return(portfolio_values, cash_flows)

        # Simple return (no cash flow adjustment)
        simple_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]

        comparison = pd.DataFrame({
            'Return Method': ['Time-Weighted (TWR)', 'Money-Weighted (MWR/IRR)', 'Simple Return'],
            'Value': [
                twr_result['time_weighted_return'],
                mwr_result['money_weighted_return'],
                simple_return,
            ],
            'Annualized': [
                twr_result['annualized_twr'],
                mwr_result['money_weighted_return'],  # Already annualized
                np.nan,
            ],
            'Description': [
                'Manager skill (timing-independent)',
                'Investor experience (timing-dependent)',
                'Naive calculation (ignores flows)',
            ],
        })

        return comparison


class CustomAttributionModel:
    """
    Framework for creating custom attribution models.

    Allows users to define their own attribution logic and calculations
    for specialized use cases.
    """

    def __init__(
        self,
        attribution_function: Callable,
        name: str = "Custom Attribution",
    ):
        """
        Initialize custom attribution model.

        Args:
            attribution_function: Function that takes portfolio data and returns attribution
            name: Name for this attribution model
        """
        self.attribution_function = attribution_function
        self.name = name

    def calculate(self, *args, **kwargs) -> Dict:
        """
        Calculate attribution using custom function.

        Args:
            *args, **kwargs: Passed to attribution_function

        Returns:
            Dictionary with attribution results
        """
        return self.attribution_function(*args, **kwargs)


# Example custom attribution functions

def style_drift_attribution(
    current_weights: pd.Series,
    target_weights: pd.Series,
    returns: pd.Series,
) -> Dict[str, Union[float, pd.Series]]:
    """
    Attribute performance to style drift from target allocation.

    Returns:
        Attribution showing impact of deviating from target weights
    """
    # Drift from target
    weight_drift = current_weights - target_weights

    # Performance impact of drift
    drift_impact = weight_drift * returns

    # Total drift impact
    total_drift_impact = drift_impact.sum()

    # Hypothetical return at target weights
    target_return = (target_weights * returns).sum()

    # Actual return
    actual_return = (current_weights * returns).sum()

    return {
        'actual_return': actual_return,
        'target_return': target_return,
        'drift_impact': total_drift_impact,
        'drift_by_asset': drift_impact,
        'weight_drift': weight_drift,
    }


def tactical_vs_strategic_attribution(
    strategic_weights: pd.Series,
    tactical_weights: pd.Series,
    returns: pd.Series,
) -> Dict[str, Union[float, pd.Series]]:
    """
    Separate attribution between strategic (long-term) and tactical (short-term) decisions.

    Returns:
        Attribution showing value added from tactical tilts
    """
    # Strategic return (long-term policy)
    strategic_return = (strategic_weights * returns).sum()

    # Tactical return (active tilts)
    tactical_return = (tactical_weights * returns).sum()

    # Tactical value added
    tactical_value_added = tactical_return - strategic_return

    # Contribution by tactical tilt
    tilts = tactical_weights - strategic_weights
    tactical_contributions = tilts * returns

    return {
        'strategic_return': strategic_return,
        'tactical_return': tactical_return,
        'tactical_value_added': tactical_value_added,
        'tactical_tilts': tilts,
        'tactical_contributions': tactical_contributions,
    }
