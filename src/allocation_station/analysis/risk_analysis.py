"""Comprehensive risk analysis including stress testing, tail risk, regime detection, and concentration metrics."""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from pydantic import BaseModel, Field
import warnings


# ============================================================================
# Stress Testing
# ============================================================================

class StressScenario(str, Enum):
    """Pre-defined stress scenarios."""
    MARKET_CRASH_1987 = "market_crash_1987"  # -20% in 1 day
    DOTCOM_BUBBLE_2000 = "dotcom_bubble_2000"  # -50% over months
    FINANCIAL_CRISIS_2008 = "financial_crisis_2008"  # -57% peak to trough
    COVID_CRASH_2020 = "covid_crash_2020"  # -34% in weeks
    CUSTOM = "custom"


class StressTest(BaseModel):
    """Represents a stress test scenario."""

    name: str
    scenario_type: StressScenario
    asset_shocks: Dict[str, float]  # Asset symbol -> return shock
    correlation_multiplier: float = 1.5  # Increase in correlations during stress
    volatility_multiplier: float = 2.0  # Increase in volatility during stress

    class Config:
        use_enum_values = True


class StressTestEngine:
    """
    Engine for performing portfolio stress tests.

    Tests portfolio performance under extreme market conditions.
    """

    def __init__(self):
        """Initialize stress test engine."""
        self.scenarios = self._define_default_scenarios()

    def _define_default_scenarios(self) -> Dict[str, StressTest]:
        """Define standard stress scenarios."""
        return {
            'black_monday': StressTest(
                name="Black Monday (1987)",
                scenario_type=StressScenario.MARKET_CRASH_1987,
                asset_shocks={'equity': -0.20, 'bond': -0.05, 'commodity': -0.15},
                correlation_multiplier=2.0,
                volatility_multiplier=3.0
            ),
            'financial_crisis': StressTest(
                name="Financial Crisis (2008)",
                scenario_type=StressScenario.FINANCIAL_CRISIS_2008,
                asset_shocks={'equity': -0.57, 'bond': 0.05, 'commodity': -0.45, 'real_estate': -0.40},
                correlation_multiplier=1.8,
                volatility_multiplier=2.5
            ),
            'covid_crash': StressTest(
                name="COVID-19 Crash (2020)",
                scenario_type=StressScenario.COVID_CRASH_2020,
                asset_shocks={'equity': -0.34, 'bond': 0.10, 'commodity': -0.30, 'cryptocurrency': -0.50},
                correlation_multiplier=1.6,
                volatility_multiplier=2.2
            ),
            'inflation_shock': StressTest(
                name="High Inflation Shock",
                scenario_type=StressScenario.CUSTOM,
                asset_shocks={'equity': -0.15, 'bond': -0.20, 'commodity': 0.30, 'real_estate': -0.10},
                correlation_multiplier=1.3,
                volatility_multiplier=1.5
            ),
            'interest_rate_spike': StressTest(
                name="Interest Rate Spike",
                scenario_type=StressScenario.CUSTOM,
                asset_shocks={'equity': -0.10, 'bond': -0.15, 'real_estate': -0.20, 'commodity': -0.05},
                correlation_multiplier=1.4,
                volatility_multiplier=1.6
            )
        }

    def run_stress_test(
        self,
        portfolio_weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        scenario: Union[str, StressTest],
        asset_class_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run stress test on portfolio.

        Args:
            portfolio_weights: Current portfolio weights
            asset_returns: Historical asset returns
            scenario: Scenario name or StressTest object
            asset_class_mapping: Mapping of assets to asset classes

        Returns:
            Dictionary with stress test results
        """
        # Get scenario
        if isinstance(scenario, str):
            if scenario not in self.scenarios:
                raise ValueError(f"Unknown scenario: {scenario}")
            scenario_obj = self.scenarios[scenario]
        else:
            scenario_obj = scenario

        # Calculate portfolio returns under stress
        stressed_returns = {}

        for asset, weight in portfolio_weights.items():
            # Map asset to asset class
            asset_class = asset_class_mapping.get(asset, 'equity') if asset_class_mapping else 'equity'

            # Get shock for asset class
            shock = scenario_obj.asset_shocks.get(asset_class, -0.10)  # Default -10%

            stressed_returns[asset] = shock * weight

        # Portfolio impact
        portfolio_shock = sum(stressed_returns.values())

        # Calculate stressed volatility
        base_vol = asset_returns.std().mean() * np.sqrt(252)
        stressed_vol = base_vol * scenario_obj.volatility_multiplier

        # Estimate VaR under stress
        stressed_var_95 = portfolio_shock - (1.645 * stressed_vol)

        return {
            'scenario_name': scenario_obj.name,
            'portfolio_shock': portfolio_shock,
            'portfolio_shock_pct': portfolio_shock * 100,
            'individual_shocks': stressed_returns,
            'stressed_volatility': stressed_vol,
            'stressed_var_95': stressed_var_95,
            'correlation_multiplier': scenario_obj.correlation_multiplier,
            'passed': portfolio_shock > -0.30  # Pass if loss < 30%
        }

    def run_multiple_scenarios(
        self,
        portfolio_weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        scenarios: Optional[List[str]] = None,
        asset_class_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Run multiple stress scenarios.

        Args:
            portfolio_weights: Portfolio weights
            asset_returns: Asset returns
            scenarios: List of scenario names (None = all)
            asset_class_mapping: Asset to class mapping

        Returns:
            DataFrame with results from all scenarios
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())

        results = []

        for scenario_name in scenarios:
            result = self.run_stress_test(
                portfolio_weights,
                asset_returns,
                scenario_name,
                asset_class_mapping
            )
            results.append(result)

        return pd.DataFrame(results)


# ============================================================================
# Tail Risk Analysis
# ============================================================================

class TailRiskMetrics(BaseModel):
    """Tail risk metrics for a portfolio."""

    var_95: float = Field(..., description="Value at Risk (95%)")
    var_99: float = Field(..., description="Value at Risk (99%)")
    cvar_95: float = Field(..., description="Conditional VaR (95%)")
    cvar_99: float = Field(..., description="Conditional VaR (99%)")
    expected_shortfall: float = Field(..., description="Expected Shortfall")
    tail_ratio: float = Field(..., description="Tail ratio (right/left tail)")
    skewness: float = Field(..., description="Return distribution skewness")
    kurtosis: float = Field(..., description="Return distribution kurtosis")
    max_drawdown: float = Field(..., description="Maximum drawdown")


class TailRiskAnalyzer:
    """
    Analyzes tail risk and extreme loss scenarios.

    Focuses on fat-tailed distributions and extreme events.
    """

    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize tail risk analyzer.

        Args:
            confidence_levels: Confidence levels for VaR/CVaR
        """
        self.confidence_levels = confidence_levels

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Return series
            confidence_level: Confidence level
            method: 'historical', 'parametric', or 'cornish_fisher'

        Returns:
            VaR value
        """
        if method == 'historical':
            return returns.quantile(1 - confidence_level)

        elif method == 'parametric':
            # Assume normal distribution
            mu = returns.mean()
            sigma = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return mu + z_score * sigma

        elif method == 'cornish_fisher':
            # Adjust for skewness and kurtosis
            z = stats.norm.ppf(1 - confidence_level)
            s = returns.skew()
            k = returns.kurtosis()

            z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36

            return returns.mean() + z_cf * returns.std()

        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            returns: Return series
            confidence_level: Confidence level

        Returns:
            CVaR value
        """
        var = self.calculate_var(returns, confidence_level, method='historical')
        return returns[returns <= var].mean()

    def calculate_tail_metrics(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> TailRiskMetrics:
        """
        Calculate comprehensive tail risk metrics.

        Args:
            returns: Return series
            annualize: Whether to annualize metrics

        Returns:
            TailRiskMetrics object
        """
        # Annualization factor
        factor = np.sqrt(252) if annualize else 1.0

        # VaR and CVaR at different confidence levels
        var_95 = self.calculate_var(returns, 0.95) * factor
        var_99 = self.calculate_var(returns, 0.99) * factor
        cvar_95 = self.calculate_cvar(returns, 0.95) * factor
        cvar_99 = self.calculate_cvar(returns, 0.99) * factor

        # Expected shortfall
        expected_shortfall = cvar_95

        # Tail ratio (95th percentile / 5th percentile)
        right_tail = returns.quantile(0.95)
        left_tail = returns.quantile(0.05)
        tail_ratio = abs(right_tail / left_tail) if left_tail != 0 else 0

        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        return TailRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            expected_shortfall=expected_shortfall,
            tail_ratio=tail_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            max_drawdown=max_dd
        )

    def estimate_extreme_value_distribution(
        self,
        returns: pd.Series,
        threshold_percentile: float = 0.05
    ) -> Dict[str, float]:
        """
        Fit Generalized Pareto Distribution to tail.

        Args:
            returns: Return series
            threshold_percentile: Percentile for threshold

        Returns:
            Dictionary with GPD parameters
        """
        threshold = returns.quantile(threshold_percentile)
        exceedances = returns[returns < threshold] - threshold

        if len(exceedances) < 10:
            return {'error': 'Insufficient tail observations'}

        # Fit GPD (simplified - in practice use specialized library)
        shape = exceedances.std() / exceedances.mean() - 1
        scale = exceedances.mean() * (1 + shape)

        return {
            'threshold': threshold,
            'shape': shape,
            'scale': scale,
            'n_exceedances': len(exceedances)
        }


# ============================================================================
# Market Regime Detection
# ============================================================================

class MarketRegime(str, Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


class RegimeDetector:
    """
    Detects market regimes using various methods.

    Identifies bull/bear markets, volatility regimes, and crisis periods.
    """

    def __init__(
        self,
        bull_threshold: float = 0.20,
        bear_threshold: float = -0.20,
        vol_window: int = 60
    ):
        """
        Initialize regime detector.

        Args:
            bull_threshold: Threshold for bull market (20% gain)
            bear_threshold: Threshold for bear market (-20% loss)
            vol_window: Window for volatility calculation
        """
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.vol_window = vol_window

    def detect_bull_bear(
        self,
        prices: pd.Series,
        lookback_window: int = 252
    ) -> pd.Series:
        """
        Detect bull and bear markets.

        Args:
            prices: Price series
            lookback_window: Lookback window for regime detection

        Returns:
            Series of regime labels
        """
        # Calculate returns from peak/trough
        cumulative = prices / prices.expanding().max()
        drawdown = cumulative - 1

        # Detect regimes
        regimes = pd.Series(index=prices.index, dtype=str)
        regimes[:] = MarketRegime.SIDEWAYS.value

        # Bull market: sustained uptrend
        bull_mask = (prices / prices.rolling(lookback_window).min() - 1) > self.bull_threshold
        regimes[bull_mask] = MarketRegime.BULL.value

        # Bear market: significant drawdown
        bear_mask = drawdown < self.bear_threshold
        regimes[bear_mask] = MarketRegime.BEAR.value

        return regimes

    def detect_volatility_regime(
        self,
        returns: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Detect volatility regimes.

        Args:
            returns: Return series
            window: Volatility window

        Returns:
            Series of volatility regime labels
        """
        window = window or self.vol_window

        # Calculate rolling volatility
        vol = returns.rolling(window).std() * np.sqrt(252)

        # Define regimes based on percentiles
        low_threshold = vol.quantile(0.33)
        high_threshold = vol.quantile(0.67)

        regimes = pd.Series(index=returns.index, dtype=str)
        regimes[:] = 'normal'
        regimes[vol < low_threshold] = MarketRegime.LOW_VOLATILITY.value
        regimes[vol > high_threshold] = MarketRegime.HIGH_VOLATILITY.value

        # Crisis: extremely high volatility (>95th percentile)
        crisis_threshold = vol.quantile(0.95)
        regimes[vol > crisis_threshold] = MarketRegime.CRISIS.value

        return regimes

    def detect_regime_hidden_markov(
        self,
        returns: pd.Series,
        n_regimes: int = 2
    ) -> pd.Series:
        """
        Detect regimes using Hidden Markov Model.

        Simplified implementation - in practice use hmmlearn library.

        Args:
            returns: Return series
            n_regimes: Number of regimes

        Returns:
            Series of regime labels
        """
        # Simplified: use K-means clustering on returns and volatility
        vol = returns.rolling(20).std()

        features = pd.DataFrame({
            'returns': returns,
            'volatility': vol
        }).dropna()

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regime_labels = kmeans.fit_predict(features)

        regimes = pd.Series(index=features.index, data=regime_labels)

        # Reindex to original
        regimes = regimes.reindex(returns.index, method='ffill')

        return regimes

    def calculate_regime_statistics(
        self,
        returns: pd.Series,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate statistics for each regime.

        Args:
            returns: Return series
            regimes: Regime labels

        Returns:
            DataFrame with regime statistics
        """
        stats_list = []

        for regime in regimes.unique():
            regime_returns = returns[regimes == regime]

            if len(regime_returns) > 0:
                stats_list.append({
                    'regime': regime,
                    'count': len(regime_returns),
                    'pct_of_total': len(regime_returns) / len(returns) * 100,
                    'mean_return': regime_returns.mean() * 252,
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe': (regime_returns.mean() / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else 0,
                    'max_return': regime_returns.max(),
                    'min_return': regime_returns.min()
                })

        return pd.DataFrame(stats_list)


# ============================================================================
# Correlation Breakdown Analysis
# ============================================================================

class CorrelationBreakdownAnalyzer:
    """
    Analyzes correlation breakdowns and regime changes.

    Identifies when correlations diverge from historical norms.
    """

    def __init__(self, baseline_window: int = 252, rolling_window: int = 60):
        """
        Initialize correlation breakdown analyzer.

        Args:
            baseline_window: Window for baseline correlation
            rolling_window: Window for rolling correlation
        """
        self.baseline_window = baseline_window
        self.rolling_window = rolling_window

    def detect_correlation_breakdown(
        self,
        returns: pd.DataFrame,
        threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Detect correlation breakdown periods.

        Args:
            returns: DataFrame of asset returns
            threshold: Threshold for breakdown detection

        Returns:
            DataFrame with breakdown indicators
        """
        # Calculate baseline correlation
        baseline_corr = returns.rolling(self.baseline_window).corr()

        # Calculate rolling correlation
        rolling_corr = returns.rolling(self.rolling_window).corr()

        # Detect breakdowns (large deviation from baseline)
        breakdowns = []

        for date in returns.index[self.baseline_window:]:
            if date in baseline_corr.index and date in rolling_corr.index:
                base_matrix = baseline_corr.loc[date]
                roll_matrix = rolling_corr.loc[date]

                # Calculate average correlation difference
                diff = (roll_matrix - base_matrix).abs()
                avg_diff = diff.values[np.triu_indices_from(diff.values, k=1)].mean()

                breakdowns.append({
                    'date': date,
                    'avg_correlation_diff': avg_diff,
                    'breakdown': avg_diff > threshold
                })

        return pd.DataFrame(breakdowns)

    def calculate_correlation_dispersion(
        self,
        returns: pd.DataFrame,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate dispersion of pairwise correlations.

        High dispersion indicates breakdown in correlation structure.

        Args:
            returns: Asset returns
            window: Rolling window

        Returns:
            Series of correlation dispersion
        """
        rolling_corr = returns.rolling(window).corr()

        dispersion = {}

        for date in returns.index[window:]:
            if date in rolling_corr.index:
                corr_matrix = rolling_corr.loc[date]
                # Get upper triangle correlations
                corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                # Calculate standard deviation of correlations
                dispersion[date] = np.std(corr_values)

        return pd.Series(dispersion)


# ============================================================================
# Liquidity Risk Assessment
# ============================================================================

class LiquidityMetrics(BaseModel):
    """Liquidity risk metrics."""

    bid_ask_spread: float = Field(..., description="Average bid-ask spread")
    turnover_ratio: float = Field(..., description="Trading volume / market cap")
    amihud_illiquidity: float = Field(..., description="Amihud illiquidity measure")
    days_to_liquidate: float = Field(..., description="Days to liquidate position")
    liquidity_score: float = Field(..., ge=0, le=100, description="Overall liquidity score")


class LiquidityRiskAssessor:
    """
    Assesses portfolio liquidity risk.

    Evaluates ability to liquidate positions without significant price impact.
    """

    def __init__(self):
        """Initialize liquidity risk assessor."""
        pass

    def calculate_amihud_illiquidity(
        self,
        returns: pd.Series,
        volume: pd.Series
    ) -> float:
        """
        Calculate Amihud illiquidity measure.

        Amihud = Average(|return| / dollar_volume)

        Args:
            returns: Return series
            volume: Volume series (in dollars)

        Returns:
            Amihud illiquidity measure
        """
        illiquidity = abs(returns) / (volume + 1e-10)  # Avoid division by zero
        return illiquidity.mean() * 1e6  # Scale up for readability

    def estimate_days_to_liquidate(
        self,
        position_size: float,
        avg_daily_volume: float,
        participation_rate: float = 0.10
    ) -> float:
        """
        Estimate days needed to liquidate position.

        Args:
            position_size: Size of position
            avg_daily_volume: Average daily trading volume
            participation_rate: Max % of daily volume to trade

        Returns:
            Number of days to liquidate
        """
        daily_liquidation = avg_daily_volume * participation_rate
        days = position_size / daily_liquidation if daily_liquidation > 0 else np.inf
        return days

    def calculate_liquidity_score(
        self,
        bid_ask_spread: float,
        turnover_ratio: float,
        market_cap: float
    ) -> float:
        """
        Calculate overall liquidity score (0-100).

        Args:
            bid_ask_spread: Bid-ask spread as %
            turnover_ratio: Volume / market cap
            market_cap: Market capitalization

        Returns:
            Liquidity score (100 = most liquid)
        """
        score = 100.0

        # Penalize wide spreads
        if bid_ask_spread > 0.01:  # > 1%
            score -= 30
        elif bid_ask_spread > 0.005:  # > 0.5%
            score -= 15

        # Penalize low turnover
        if turnover_ratio < 0.01:  # < 1% daily
            score -= 30
        elif turnover_ratio < 0.05:  # < 5% daily
            score -= 15

        # Penalize small market cap
        if market_cap < 1e9:  # < $1B
            score -= 20
        elif market_cap < 10e9:  # < $10B
            score -= 10

        return max(0, min(100, score))


# ============================================================================
# Concentration Risk
# ============================================================================

class ConcentrationRiskAnalyzer:
    """
    Analyzes portfolio concentration risk.

    Measures diversification and concentration across assets, sectors, and factors.
    """

    def __init__(self):
        """Initialize concentration risk analyzer."""
        pass

    def calculate_herfindahl_index(
        self,
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate Herfindahl-Hirschman Index.

        HHI = sum of squared weights
        Range: 1/n (perfectly diversified) to 1 (fully concentrated)

        Args:
            weights: Portfolio weights

        Returns:
            HHI value
        """
        return sum(w**2 for w in weights.values())

    def calculate_effective_n(
        self,
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate effective number of assets.

        Effective N = 1 / HHI

        Args:
            weights: Portfolio weights

        Returns:
            Effective number of assets
        """
        hhi = self.calculate_herfindahl_index(weights)
        return 1.0 / hhi if hhi > 0 else 0

    def calculate_concentration_ratio(
        self,
        weights: Dict[str, float],
        top_n: int = 5
    ) -> float:
        """
        Calculate concentration ratio for top N positions.

        Args:
            weights: Portfolio weights
            top_n: Number of top positions

        Returns:
            Sum of top N weights
        """
        sorted_weights = sorted(weights.values(), reverse=True)
        return sum(sorted_weights[:top_n])

    def calculate_diversification_ratio(
        self,
        weights: Dict[str, float],
        asset_volatilities: Dict[str, float],
        correlation_matrix: pd.DataFrame
    ) -> float:
        """
        Calculate diversification ratio.

        DR = (sum of weighted vols) / portfolio vol

        Args:
            weights: Portfolio weights
            asset_volatilities: Individual asset volatilities
            correlation_matrix: Asset correlation matrix

        Returns:
            Diversification ratio
        """
        # Weighted average volatility
        weighted_vol = sum(weights[asset] * asset_volatilities[asset] for asset in weights)

        # Portfolio volatility
        weight_array = np.array([weights.get(asset, 0) for asset in correlation_matrix.columns])
        vol_array = np.array([asset_volatilities.get(asset, 0) for asset in correlation_matrix.columns])

        cov_matrix = np.outer(vol_array, vol_array) * correlation_matrix.values
        portfolio_var = weight_array @ cov_matrix @ weight_array.T
        portfolio_vol = np.sqrt(portfolio_var)

        return weighted_vol / portfolio_vol if portfolio_vol > 0 else 0


# ============================================================================
# Systematic vs Idiosyncratic Risk
# ============================================================================

class RiskDecomposition(BaseModel):
    """Risk decomposition results."""

    total_risk: float
    systematic_risk: float
    idiosyncratic_risk: float
    systematic_pct: float
    idiosyncratic_pct: float
    beta: float
    r_squared: float


class RiskDecomposer:
    """
    Decomposes portfolio risk into systematic and idiosyncratic components.

    Uses factor models to separate market risk from asset-specific risk.
    """

    def __init__(self):
        """Initialize risk decomposer."""
        pass

    def decompose_risk(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series
    ) -> RiskDecomposition:
        """
        Decompose portfolio risk using single-factor model.

        Args:
            portfolio_returns: Portfolio return series
            market_returns: Market return series

        Returns:
            RiskDecomposition object
        """
        # Align series
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'market': market_returns
        }).dropna()

        # Calculate beta
        covariance = aligned['portfolio'].cov(aligned['market'])
        market_variance = aligned['market'].var()
        beta = covariance / market_variance if market_variance > 0 else 0

        # Calculate R-squared
        correlation = aligned['portfolio'].corr(aligned['market'])
        r_squared = correlation ** 2

        # Total risk (variance)
        total_variance = aligned['portfolio'].var()
        total_risk = np.sqrt(total_variance) * np.sqrt(252)  # Annualized

        # Systematic risk (explained by market)
        systematic_variance = (beta ** 2) * market_variance
        systematic_risk = np.sqrt(systematic_variance) * np.sqrt(252)

        # Idiosyncratic risk (residual)
        idiosyncratic_variance = total_variance - systematic_variance
        idiosyncratic_risk = np.sqrt(max(0, idiosyncratic_variance)) * np.sqrt(252)

        # Percentages
        systematic_pct = (systematic_variance / total_variance) * 100 if total_variance > 0 else 0
        idiosyncratic_pct = 100 - systematic_pct

        return RiskDecomposition(
            total_risk=total_risk,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            systematic_pct=systematic_pct,
            idiosyncratic_pct=idiosyncratic_pct,
            beta=beta,
            r_squared=r_squared
        )

    def decompose_multi_factor(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Decompose risk using multi-factor model.

        Args:
            portfolio_returns: Portfolio returns
            factor_returns: DataFrame of factor returns

        Returns:
            Dictionary with factor exposures and risk decomposition
        """
        # Align data
        aligned = factor_returns.copy()
        aligned['portfolio'] = portfolio_returns
        aligned = aligned.dropna()

        # Prepare for regression
        X = aligned[factor_returns.columns]
        y = aligned['portfolio']

        # Run regression
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X, y)

        # Factor exposures (betas)
        factor_betas = dict(zip(factor_returns.columns, model.coef_))

        # Predictions and residuals
        predicted = model.predict(X)
        residuals = y - predicted

        # Risk decomposition
        total_var = y.var()
        explained_var = predicted.var()
        residual_var = residuals.var()

        return {
            'factor_betas': factor_betas,
            'r_squared': model.score(X, y),
            'total_risk': np.sqrt(total_var) * np.sqrt(252),
            'systematic_risk': np.sqrt(explained_var) * np.sqrt(252),
            'idiosyncratic_risk': np.sqrt(residual_var) * np.sqrt(252),
            'systematic_pct': (explained_var / total_var) * 100 if total_var > 0 else 0
        }
