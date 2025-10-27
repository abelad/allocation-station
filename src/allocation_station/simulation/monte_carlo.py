"""Monte Carlo simulation engine for portfolio analysis."""

from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
from ..core import Portfolio, Asset
from ..portfolio import AllocationStrategy, WithdrawalStrategy


class SimulationConfig(BaseModel):
    """Configuration for Monte Carlo simulation."""

    # Simulation parameters
    n_simulations: int = Field(1000, description="Number of simulation runs")
    time_horizon: int = Field(30, description="Time horizon in years")
    time_steps: int = Field(252, description="Time steps per year (252 for daily)")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    # Return parameters
    expected_returns: Optional[Dict[str, float]] = Field(None, description="Expected annual returns")
    volatilities: Optional[Dict[str, float]] = Field(None, description="Annual volatilities")
    correlations: Optional[pd.DataFrame] = Field(None, description="Correlation matrix")

    # Distribution settings
    distribution: str = Field("normal", description="Return distribution (normal, t, historical)")
    confidence_levels: List[float] = Field([0.05, 0.25, 0.50, 0.75, 0.95], description="Percentiles to calculate")

    # Rebalancing
    rebalance_frequency: str = Field("annual", description="Rebalancing frequency")
    rebalance_threshold: float = Field(0.05, description="Threshold for rebalancing")

    # Withdrawal settings
    withdrawal_strategy: Optional[WithdrawalStrategy] = Field(None, description="Withdrawal strategy")
    inflation_rate: float = Field(0.025, description="Annual inflation rate")

    # Performance settings
    use_parallel: bool = Field(True, description="Use parallel processing")
    n_jobs: int = Field(-1, description="Number of parallel jobs (-1 for all cores)")
    show_progress: bool = Field(True, description="Show progress bar")

    class Config:
        arbitrary_types_allowed = True

    @validator("n_simulations")
    def validate_simulations(cls, v):
        """Ensure reasonable number of simulations."""
        if v < 100:
            warnings.warn("Less than 100 simulations may not provide reliable results")
        elif v > 100000:
            warnings.warn("More than 100,000 simulations may take excessive time")
        return v


@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation."""

    # Portfolio paths
    portfolio_values: np.ndarray  # Shape: (n_simulations, n_periods)
    returns: np.ndarray  # Shape: (n_simulations, n_periods)

    # Summary statistics
    final_values: np.ndarray
    total_returns: np.ndarray
    annualized_returns: np.ndarray
    max_drawdowns: np.ndarray

    # Risk metrics
    var_levels: Dict[float, float]
    cvar_levels: Dict[float, float]

    # Percentile paths
    percentile_paths: Dict[float, np.ndarray]

    # Success metrics
    success_rate: float  # Probability of meeting goals
    median_final_value: float
    mean_final_value: float

    # Additional analysis
    best_scenario: Dict[str, Any]
    worst_scenario: Dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        df = pd.DataFrame({
            'final_value': self.final_values,
            'total_return': self.total_returns,
            'annualized_return': self.annualized_returns,
            'max_drawdown': self.max_drawdowns,
        })
        return df


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for portfolio analysis.

    This class implements Monte Carlo methods for simulating portfolio
    performance under various market scenarios and strategies.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize simulator with configuration."""
        self.config = config or SimulationConfig()

        if self.config.random_seed:
            np.random.seed(self.config.random_seed)

    def simulate(
        self,
        portfolio: Portfolio,
        strategy: Optional[AllocationStrategy] = None
    ) -> SimulationResults:
        """
        Run Monte Carlo simulation for portfolio.

        Args:
            portfolio: Portfolio to simulate
            strategy: Allocation strategy to apply

        Returns:
            SimulationResults object with simulation outcomes
        """
        # Prepare simulation parameters
        params = self._prepare_simulation_params(portfolio)

        # Run simulations
        if self.config.use_parallel:
            results = self._run_parallel_simulations(portfolio, strategy, params)
        else:
            results = self._run_sequential_simulations(portfolio, strategy, params)

        # Analyze results
        return self._analyze_results(results)

    def _prepare_simulation_params(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Prepare parameters for simulation."""
        params = {}

        # Get asset symbols
        symbols = list(portfolio.assets.keys())
        params['symbols'] = symbols

        # Calculate or use provided returns and volatilities
        if self.config.expected_returns:
            params['returns'] = np.array([
                self.config.expected_returns.get(s, 0.07) for s in symbols
            ])
        else:
            params['returns'] = self._calculate_historical_returns(portfolio)

        if self.config.volatilities:
            params['volatilities'] = np.array([
                self.config.volatilities.get(s, 0.15) for s in symbols
            ])
        else:
            params['volatilities'] = self._calculate_historical_volatilities(portfolio)

        # Get correlation matrix
        if self.config.correlations is not None:
            params['correlation'] = self.config.correlations.values
        else:
            params['correlation'] = self._calculate_correlation_matrix(portfolio)

        # Calculate covariance matrix
        vol_matrix = np.diag(params['volatilities'])
        params['covariance'] = vol_matrix @ params['correlation'] @ vol_matrix

        # Get initial weights
        if portfolio.allocation:
            weights = portfolio.calculate_weights()
            params['initial_weights'] = np.array([weights.get(s, 0) for s in symbols])
        else:
            # Equal weight if no allocation specified
            params['initial_weights'] = np.ones(len(symbols)) / len(symbols)

        # Time parameters
        params['n_periods'] = self.config.time_horizon * self.config.time_steps
        params['dt'] = 1.0 / self.config.time_steps

        return params

    def _run_sequential_simulations(
        self,
        portfolio: Portfolio,
        strategy: Optional[AllocationStrategy],
        params: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Run simulations sequentially."""
        results = []

        iterator = range(self.config.n_simulations)
        if self.config.show_progress:
            iterator = tqdm(iterator, desc="Running simulations")

        for _ in iterator:
            path = self._simulate_single_path(portfolio, strategy, params)
            results.append(path)

        return results

    def _run_parallel_simulations(
        self,
        portfolio: Portfolio,
        strategy: Optional[AllocationStrategy],
        params: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Run simulations in parallel."""
        with tqdm(total=self.config.n_simulations, desc="Running simulations",
                  disable=not self.config.show_progress) as pbar:

            def run_and_update(i):
                result = self._simulate_single_path(portfolio, strategy, params)
                pbar.update(1)
                return result

            results = Parallel(n_jobs=self.config.n_jobs, backend='threading')(
                delayed(run_and_update)(i) for i in range(self.config.n_simulations)
            )

        return results

    def _simulate_single_path(
        self,
        portfolio: Portfolio,
        strategy: Optional[AllocationStrategy],
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Simulate a single portfolio path."""
        n_assets = len(params['symbols'])
        n_periods = params['n_periods']
        dt = params['dt']

        # Initialize portfolio value path
        portfolio_values = np.zeros(n_periods + 1)
        portfolio_values[0] = portfolio.initial_value

        # Current weights
        weights = params['initial_weights'].copy()

        # Generate random returns based on distribution
        if self.config.distribution == 'normal':
            # Generate correlated normal returns
            random_returns = np.random.multivariate_normal(
                mean=params['returns'] * dt,
                cov=params['covariance'] * dt,
                size=n_periods
            )
        elif self.config.distribution == 't':
            # Student-t distribution for fat tails
            df = 5  # Degrees of freedom
            random_returns = stats.multivariate_t.rvs(
                loc=params['returns'] * dt,
                shape=params['covariance'] * dt,
                df=df,
                size=n_periods
            )
        else:
            # Use historical simulation (bootstrap)
            random_returns = self._bootstrap_returns(portfolio, n_periods)

        # Simulate portfolio evolution
        for t in range(n_periods):
            # Calculate portfolio return for this period
            portfolio_return = np.sum(weights * random_returns[t])

            # Update portfolio value
            portfolio_values[t + 1] = portfolio_values[t] * (1 + portfolio_return)

            # Apply withdrawal if configured
            if self.config.withdrawal_strategy:
                withdrawal = self._calculate_withdrawal(
                    portfolio_values[t + 1],
                    t,
                    self.config.withdrawal_strategy
                )
                portfolio_values[t + 1] -= withdrawal

            # Check for rebalancing
            if self._should_rebalance(t):
                weights = self._rebalance_weights(weights, random_returns[t], params)

        return portfolio_values

    def _should_rebalance(self, period: int) -> bool:
        """Check if portfolio should be rebalanced at this period."""
        if self.config.rebalance_frequency == 'never':
            return False

        freq_map = {
            'daily': 1,
            'weekly': 5,
            'monthly': 21,
            'quarterly': 63,
            'annual': 252,
        }

        frequency = freq_map.get(self.config.rebalance_frequency, 252)
        return period > 0 and period % frequency == 0

    def _rebalance_weights(
        self,
        current_weights: np.ndarray,
        returns: np.ndarray,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Rebalance portfolio weights."""
        # Update weights based on returns
        new_values = current_weights * (1 + returns)
        total_value = np.sum(new_values)

        if total_value > 0:
            new_weights = new_values / total_value

            # Check if rebalancing needed (based on threshold)
            max_drift = np.max(np.abs(new_weights - params['initial_weights']))

            if max_drift > self.config.rebalance_threshold:
                return params['initial_weights']
            else:
                return new_weights

        return current_weights

    def _calculate_withdrawal(
        self,
        portfolio_value: float,
        period: int,
        strategy: WithdrawalStrategy
    ) -> float:
        """Calculate withdrawal amount for period."""
        # Simplified withdrawal calculation
        annual_periods = self.config.time_steps

        if period % annual_periods == 0:  # Annual withdrawal
            withdrawal_rate = strategy.initial_withdrawal_rate

            # Adjust for inflation
            if strategy.inflation_adjustment:
                years = period / annual_periods
                inflation_factor = (1 + self.config.inflation_rate) ** years
                withdrawal_rate *= inflation_factor

            return portfolio_value * withdrawal_rate

        return 0

    def _analyze_results(self, simulation_paths: List[np.ndarray]) -> SimulationResults:
        """Analyze simulation results."""
        # Convert to numpy array
        paths = np.array(simulation_paths)
        n_sims, n_periods = paths.shape

        # Calculate final values
        final_values = paths[:, -1]
        initial_value = paths[0, 0]

        # Calculate returns
        total_returns = (final_values - initial_value) / initial_value
        years = n_periods / self.config.time_steps
        annualized_returns = (1 + total_returns) ** (1/years) - 1

        # Calculate maximum drawdowns
        max_drawdowns = np.zeros(n_sims)
        for i in range(n_sims):
            cummax = np.maximum.accumulate(paths[i])
            drawdowns = (paths[i] - cummax) / cummax
            max_drawdowns[i] = drawdowns.min()

        # Calculate VaR and CVaR
        var_levels = {}
        cvar_levels = {}

        for confidence in self.config.confidence_levels:
            var_levels[confidence] = np.percentile(final_values, confidence * 100)
            # CVaR is the mean of values below VaR
            cvar_levels[confidence] = final_values[final_values <= var_levels[confidence]].mean()

        # Calculate percentile paths
        percentile_paths = {}
        for percentile in self.config.confidence_levels:
            percentile_paths[percentile] = np.percentile(paths, percentile * 100, axis=0)

        # Success rate (e.g., portfolio survives)
        success_rate = np.mean(final_values > 0)

        # Best and worst scenarios
        best_idx = np.argmax(final_values)
        worst_idx = np.argmin(final_values)

        best_scenario = {
            'final_value': final_values[best_idx],
            'total_return': total_returns[best_idx],
            'path': paths[best_idx]
        }

        worst_scenario = {
            'final_value': final_values[worst_idx],
            'total_return': total_returns[worst_idx],
            'path': paths[worst_idx]
        }

        # Calculate returns array
        returns_array = np.diff(paths, axis=1) / paths[:, :-1]

        return SimulationResults(
            portfolio_values=paths,
            returns=returns_array,
            final_values=final_values,
            total_returns=total_returns,
            annualized_returns=annualized_returns,
            max_drawdowns=max_drawdowns,
            var_levels=var_levels,
            cvar_levels=cvar_levels,
            percentile_paths=percentile_paths,
            success_rate=success_rate,
            median_final_value=np.median(final_values),
            mean_final_value=np.mean(final_values),
            best_scenario=best_scenario,
            worst_scenario=worst_scenario
        )

    def _calculate_historical_returns(self, portfolio: Portfolio) -> np.ndarray:
        """Calculate historical returns from portfolio assets."""
        returns = []

        for symbol, asset in portfolio.assets.items():
            if asset.metrics and asset.metrics.expected_return:
                returns.append(asset.metrics.expected_return)
            else:
                returns.append(0.07)  # Default 7% return

        return np.array(returns)

    def _calculate_historical_volatilities(self, portfolio: Portfolio) -> np.ndarray:
        """Calculate historical volatilities from portfolio assets."""
        volatilities = []

        for symbol, asset in portfolio.assets.items():
            if asset.metrics and asset.metrics.volatility:
                volatilities.append(asset.metrics.volatility)
            else:
                volatilities.append(0.15)  # Default 15% volatility

        return np.array(volatilities)

    def _calculate_correlation_matrix(self, portfolio: Portfolio) -> np.ndarray:
        """Calculate correlation matrix from historical data."""
        try:
            return portfolio.calculate_correlation_matrix().values
        except:
            # Default correlation matrix (moderate correlation)
            n = len(portfolio.assets)
            corr = np.full((n, n), 0.3)  # 0.3 correlation between assets
            np.fill_diagonal(corr, 1.0)
            return corr

    def _bootstrap_returns(self, portfolio: Portfolio, n_periods: int) -> np.ndarray:
        """Bootstrap returns from historical data."""
        # This would sample from historical returns
        # For now, use normal distribution as fallback
        params = self._prepare_simulation_params(portfolio)

        return np.random.multivariate_normal(
            mean=params['returns'] / self.config.time_steps,
            cov=params['covariance'] / self.config.time_steps,
            size=n_periods
        )