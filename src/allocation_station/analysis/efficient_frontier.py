"""Efficient frontier analysis using Modern Portfolio Theory."""

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp
from dataclasses import dataclass
from ..core import Portfolio


class OptimizationObjective(str, Enum):
    """Portfolio optimization objectives."""

    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    MEAN_CVaR = "mean_cvar"


@dataclass
class OptimalPortfolio:
    """Represents an optimal portfolio allocation."""

    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    objective_value: float


class EfficientFrontier:
    """
    Calculates efficient frontier using Modern Portfolio Theory.

    This class implements various portfolio optimization techniques
    including mean-variance optimization, risk parity, and CVaR optimization.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize efficient frontier calculator.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.expected_returns = expected_returns
        self.cov_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        self.assets = list(expected_returns.index)

    def optimize(
        self,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimalPortfolio:
        """
        Optimize portfolio for given objective.

        Args:
            objective: Optimization objective
            constraints: Additional constraints

        Returns:
            OptimalPortfolio with optimal weights
        """
        if objective == OptimizationObjective.MAX_SHARPE:
            return self._max_sharpe()
        elif objective == OptimizationObjective.MIN_VARIANCE:
            return self._min_variance()
        elif objective == OptimizationObjective.MAX_RETURN:
            return self._max_return(constraints)
        elif objective == OptimizationObjective.RISK_PARITY:
            return self._risk_parity()
        elif objective == OptimizationObjective.MAX_DIVERSIFICATION:
            return self._max_diversification()
        elif objective == OptimizationObjective.MEAN_CVaR:
            return self._mean_cvar_optimization(constraints)
        else:
            raise ValueError(f"Unknown objective: {objective}")

    def _max_sharpe(self) -> OptimalPortfolio:
        """Maximize Sharpe ratio."""
        # Use CVXPY for convex optimization
        weights = cp.Variable(self.n_assets)
        returns = self.expected_returns.values
        cov = self.cov_matrix.values

        # Portfolio return and risk
        portfolio_return = returns @ weights
        portfolio_risk = cp.quad_form(weights, cov)

        # Constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]

        # We can't directly maximize Sharpe ratio (non-convex)
        # Instead, we solve for different target returns and find max Sharpe
        best_sharpe = -np.inf
        best_weights = None

        target_returns = np.linspace(float(np.min(returns)), float(np.max(returns)), 50)

        for target_return in target_returns:
            prob = cp.Problem(
                cp.Minimize(portfolio_risk),
                constraints + [portfolio_return >= target_return]
            )

            try:
                prob.solve()

                if prob.status == cp.OPTIMAL:
                    w = weights.value
                    ret = returns @ w
                    vol = np.sqrt(w @ self.cov_matrix.values @ w)
                    sharpe = (ret - self.risk_free_rate) / vol

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_weights = w
            except:
                continue

        if best_weights is None:
            # Fallback to equal weights
            best_weights = np.ones(self.n_assets) / self.n_assets

        return self._create_optimal_portfolio(best_weights)

    def _min_variance(self) -> OptimalPortfolio:
        """Find minimum variance portfolio."""
        weights = cp.Variable(self.n_assets)
        cov = self.cov_matrix.values

        # Minimize variance
        portfolio_risk = cp.quad_form(weights, cov)

        # Constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]

        prob = cp.Problem(cp.Minimize(portfolio_risk), constraints)
        prob.solve()

        if prob.status != cp.OPTIMAL:
            # Fallback to equal weights
            weights_value = np.ones(self.n_assets) / self.n_assets
        else:
            weights_value = weights.value

        return self._create_optimal_portfolio(weights_value)

    def _max_return(self, constraints: Optional[Dict] = None) -> OptimalPortfolio:
        """Maximize expected return subject to risk constraint."""
        weights = cp.Variable(self.n_assets)
        returns = self.expected_returns.values

        # Maximize return
        portfolio_return = returns @ weights

        # Default constraints
        cons = [
            cp.sum(weights) == 1,
            weights >= 0
        ]

        # Add risk constraint if specified
        if constraints and 'max_volatility' in constraints:
            max_vol = constraints['max_volatility']
            cov = self.cov_matrix.values
            portfolio_risk = cp.quad_form(weights, cov)
            cons.append(portfolio_risk <= max_vol ** 2)

        prob = cp.Problem(cp.Maximize(portfolio_return), cons)
        prob.solve()

        if prob.status != cp.OPTIMAL:
            # Fallback
            weights_value = np.ones(self.n_assets) / self.n_assets
        else:
            weights_value = weights.value

        return self._create_optimal_portfolio(weights_value)

    def _risk_parity(self) -> OptimalPortfolio:
        """Calculate risk parity portfolio."""
        # Risk parity: equal risk contribution from each asset
        # This is a non-convex problem, use iterative solution

        weights = np.ones(self.n_assets) / self.n_assets
        cov = self.cov_matrix.values

        for _ in range(100):  # Iterations
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights @ cov @ weights)
            marginal_contrib = cov @ weights
            contrib = weights * marginal_contrib / portfolio_vol

            # Target equal contribution
            target_contrib = np.ones(self.n_assets) / self.n_assets

            # Update weights
            weights = weights * (target_contrib / contrib)
            weights = weights / weights.sum()

        return self._create_optimal_portfolio(weights)

    def _max_diversification(self) -> OptimalPortfolio:
        """Maximize diversification ratio."""
        # Diversification ratio = weighted average volatility / portfolio volatility

        def neg_diversification_ratio(weights):
            weights = weights.reshape(-1, 1)
            # Individual volatilities
            vols = np.sqrt(np.diag(self.cov_matrix.values))
            # Weighted average volatility
            avg_vol = np.sum(weights.flatten() * vols)
            # Portfolio volatility
            port_vol = np.sqrt(weights.T @ self.cov_matrix.values @ weights)[0, 0]
            return -avg_vol / port_vol if port_vol > 0 else 0

        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        ]

        # Bounds
        bounds = [(0, 1) for _ in range(self.n_assets)]

        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            neg_diversification_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        weights = result.x if result.success else x0

        return self._create_optimal_portfolio(weights)

    def _mean_cvar_optimization(
        self,
        constraints: Optional[Dict] = None,
        alpha: float = 0.05
    ) -> OptimalPortfolio:
        """Mean-CVaR optimization."""
        # This would require historical returns data
        # For now, return min variance as approximation
        return self._min_variance()

    def _create_optimal_portfolio(self, weights: np.ndarray) -> OptimalPortfolio:
        """Create OptimalPortfolio object from weights."""
        weights_dict = {asset: weight for asset, weight in zip(self.assets, weights)}

        # Calculate metrics
        expected_return = weights @ self.expected_returns.values
        variance = weights @ self.cov_matrix.values @ weights
        volatility = np.sqrt(variance)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        return OptimalPortfolio(
            weights=weights_dict,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            objective_value=sharpe_ratio  # Default to Sharpe
        )

    def calculate_frontier(
        self,
        n_points: int = 100,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier points.

        Args:
            n_points: Number of points on frontier
            min_return: Minimum return for frontier
            max_return: Maximum return for frontier

        Returns:
            DataFrame with frontier points
        """
        min_return = min_return or self.expected_returns.min()
        max_return = max_return or self.expected_returns.max()

        target_returns = np.linspace(min_return, max_return, n_points)
        frontier_points = []

        for target in target_returns:
            # Minimize variance for target return
            weights = cp.Variable(self.n_assets)
            returns = self.expected_returns.values
            cov = self.cov_matrix.values

            portfolio_return = returns @ weights
            portfolio_risk = cp.quad_form(weights, cov)

            constraints = [
                cp.sum(weights) == 1,
                weights >= 0,
                portfolio_return >= target
            ]

            prob = cp.Problem(cp.Minimize(portfolio_risk), constraints)

            try:
                prob.solve()

                if prob.status == cp.OPTIMAL:
                    w = weights.value
                    vol = np.sqrt(prob.value)
                    ret = returns @ w
                    sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0

                    frontier_points.append({
                        'return': ret,
                        'volatility': vol,
                        'sharpe_ratio': sharpe
                    })
            except:
                continue

        return pd.DataFrame(frontier_points)