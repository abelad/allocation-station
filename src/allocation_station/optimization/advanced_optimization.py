"""Advanced portfolio optimization techniques including Black-Litterman, HRP, robust optimization, and more."""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import cvxpy as cp
from pydantic import BaseModel, Field
import warnings


# ============================================================================
# Black-Litterman Model
# ============================================================================

class View(BaseModel):
    """Represents an investor view in Black-Litterman model."""

    assets: List[str] = Field(..., description="Assets involved in view")
    weights: List[float] = Field(..., description="Weights for each asset")
    expected_return: float = Field(..., description="Expected return for this view")
    confidence: float = Field(1.0, ge=0.0, description="Confidence in view (0-1)")


class BlackLittermanModel:
    """
    Black-Litterman portfolio optimization model.

    Combines market equilibrium with investor views to generate
    posterior expected returns.
    """

    def __init__(
        self,
        market_caps: Dict[str, float],
        risk_free_rate: float = 0.02,
        risk_aversion: float = 2.5,
        tau: float = 0.05
    ):
        """
        Initialize Black-Litterman model.

        Args:
            market_caps: Market capitalizations by asset
            risk_free_rate: Risk-free rate
            risk_aversion: Market risk aversion coefficient
            tau: Uncertainty in prior (typically 0.01-0.05)
        """
        self.market_caps = market_caps
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.tau = tau

    def calculate_market_implied_returns(
        self,
        covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate market-implied equilibrium returns.

        Args:
            covariance_matrix: Asset covariance matrix

        Returns:
            Series of implied returns
        """
        # Market weights from market caps
        total_market_cap = sum(self.market_caps.values())
        market_weights = {asset: cap / total_market_cap
                         for asset, cap in self.market_caps.items()}

        # Convert to array
        assets = list(covariance_matrix.columns)
        w = np.array([market_weights.get(asset, 0) for asset in assets])

        # Implied returns: π = λ * Σ * w
        implied_returns = self.risk_aversion * (covariance_matrix.values @ w)

        return pd.Series(implied_returns, index=assets)

    def calculate_posterior_returns(
        self,
        prior_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        views: List[View]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate posterior returns incorporating views.

        Args:
            prior_returns: Prior (equilibrium) returns
            covariance_matrix: Asset covariance matrix
            views: List of investor views

        Returns:
            Tuple of (posterior returns, posterior covariance)
        """
        n_assets = len(prior_returns)
        assets = list(prior_returns.index)

        # Build P matrix (view portfolio weights)
        P = np.zeros((len(views), n_assets))
        for i, view in enumerate(views):
            for j, asset in enumerate(view.assets):
                if asset in assets:
                    asset_idx = assets.index(asset)
                    weight_idx = view.assets.index(asset)
                    P[i, asset_idx] = view.weights[weight_idx]

        # Build Q vector (view returns)
        Q = np.array([view.expected_return for view in views])

        # Build Ω matrix (view uncertainty)
        # Diagonal matrix with uncertainty proportional to view confidence
        view_vars = []
        for view in views:
            # Lower confidence = higher uncertainty
            uncertainty = (1.0 / view.confidence) if view.confidence > 0 else 1e6
            view_vars.append(uncertainty * self.tau)
        Omega = np.diag(view_vars)

        # Black-Litterman formula
        # Posterior expected returns
        tau_sigma = self.tau * covariance_matrix.values

        # M = [(τΣ)^-1 + P'Ω^-1P]^-1
        term1 = np.linalg.inv(tau_sigma)
        term2 = P.T @ np.linalg.inv(Omega) @ P
        M = np.linalg.inv(term1 + term2)

        # μ_BL = M[(τΣ)^-1 π + P'Ω^-1 Q]
        posterior_returns = M @ (term1 @ prior_returns.values + P.T @ np.linalg.inv(Omega) @ Q)

        # Posterior covariance
        # Σ_BL = Σ + M
        posterior_cov = covariance_matrix.values + M

        return (
            pd.Series(posterior_returns, index=assets),
            pd.DataFrame(posterior_cov, index=assets, columns=assets)
        )

    def optimize(
        self,
        covariance_matrix: pd.DataFrame,
        views: Optional[List[View]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio using Black-Litterman.

        Args:
            covariance_matrix: Asset covariance matrix
            views: Investor views (optional)
            constraints: Portfolio constraints

        Returns:
            Optimal weights
        """
        # Get prior returns
        prior_returns = self.calculate_market_implied_returns(covariance_matrix)

        # Get posterior returns if views provided
        if views:
            posterior_returns, posterior_cov = self.calculate_posterior_returns(
                prior_returns, covariance_matrix, views
            )
        else:
            posterior_returns = prior_returns
            posterior_cov = covariance_matrix

        # Mean-variance optimization with posterior returns
        n = len(posterior_returns)

        # Variables
        w = cp.Variable(n)

        # Objective: maximize return - risk_aversion * variance
        ret = posterior_returns.values @ w
        risk = cp.quad_form(w, posterior_cov.values)
        objective = cp.Maximize(ret - (self.risk_aversion / 2) * risk)

        # Constraints
        constraints_list = [cp.sum(w) == 1]  # Fully invested

        if constraints:
            if 'long_only' in constraints and constraints['long_only']:
                constraints_list.append(w >= 0)

            if 'max_weight' in constraints:
                constraints_list.append(w <= constraints['max_weight'])

            if 'min_weight' in constraints:
                constraints_list.append(w >= constraints['min_weight'])

        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()

        if w.value is None:
            raise ValueError("Optimization failed")

        return dict(zip(posterior_returns.index, w.value))


# ============================================================================
# Robust Optimization
# ============================================================================

class UncertaintySet(str, Enum):
    """Types of uncertainty sets for robust optimization."""
    BOX = "box"
    ELLIPSOIDAL = "ellipsoidal"
    POLYHEDRAL = "polyhedral"


class RobustOptimizer:
    """
    Robust portfolio optimization.

    Optimizes portfolios considering parameter uncertainty.
    """

    def __init__(
        self,
        uncertainty_set: UncertaintySet = UncertaintySet.ELLIPSOIDAL,
        kappa: float = 1.0
    ):
        """
        Initialize robust optimizer.

        Args:
            uncertainty_set: Type of uncertainty set
            kappa: Size of uncertainty set
        """
        self.uncertainty_set = uncertainty_set
        self.kappa = kappa

    def optimize_worst_case_var(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        return_estimates_std: Optional[pd.Series] = None,
        target_return: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Optimize for worst-case variance.

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            return_estimates_std: Std dev of return estimates
            target_return: Target return constraint

        Returns:
            Optimal weights
        """
        n = len(expected_returns)

        # Variables
        w = cp.Variable(n)

        # Worst-case variance with ellipsoidal uncertainty
        if return_estimates_std is None:
            return_estimates_std = pd.Series(
                np.ones(n) * 0.1,  # Default 10% estimation error
                index=expected_returns.index
            )

        # Objective: minimize worst-case variance
        nominal_var = cp.quad_form(w, covariance_matrix.values)

        # Add uncertainty term
        uncertainty_term = self.kappa * cp.norm(
            cp.multiply(return_estimates_std.values, w)
        )
        worst_case_var = nominal_var + uncertainty_term

        objective = cp.Minimize(worst_case_var)

        # Constraints
        constraints_list = [
            cp.sum(w) == 1,
            w >= 0  # Long only
        ]

        if target_return is not None:
            constraints_list.append(expected_returns.values @ w >= target_return)

        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()

        if w.value is None:
            raise ValueError("Optimization failed")

        return dict(zip(expected_returns.index, w.value))

    def optimize_worst_case_cvar(
        self,
        returns_scenarios: pd.DataFrame,
        alpha: float = 0.95
    ) -> Dict[str, float]:
        """
        Optimize worst-case CVaR.

        Args:
            returns_scenarios: Scenario returns (scenarios x assets)
            alpha: CVaR confidence level

        Returns:
            Optimal weights
        """
        n_assets = returns_scenarios.shape[1]
        n_scenarios = returns_scenarios.shape[0]

        # Variables
        w = cp.Variable(n_assets)
        z = cp.Variable(n_scenarios)
        gamma = cp.Variable()

        # CVaR constraints
        portfolio_returns = returns_scenarios.values @ w

        # Objective: maximize worst-case CVaR
        cvar = gamma - (1 / (1 - alpha)) * cp.sum(z) / n_scenarios
        objective = cp.Maximize(cvar)

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            z >= 0,
            z >= gamma - portfolio_returns
        ]

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if w.value is None:
            raise ValueError("Optimization failed")

        return dict(zip(returns_scenarios.columns, w.value))


# ============================================================================
# Hierarchical Risk Parity (HRP)
# ============================================================================

class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity portfolio optimization.

    Uses hierarchical clustering to build portfolios with balanced risk.
    """

    def __init__(self, linkage_method: str = 'single'):
        """
        Initialize HRP optimizer.

        Args:
            linkage_method: Hierarchical clustering method
        """
        self.linkage_method = linkage_method

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Get quasi-diagonal order from linkage matrix."""
        link = link.astype(int)
        sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_idx.max() >= num_items:
            sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
            df0 = sort_idx[sort_idx >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_idx[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_idx = pd.concat([sort_idx, df0])
            sort_idx = sort_idx.sort_index()
            sort_idx.index = range(sort_idx.shape[0])

        return sort_idx.tolist()

    def _get_cluster_var(
        self,
        cov: pd.DataFrame,
        cluster_items: List[int]
    ) -> float:
        """Calculate cluster variance."""
        cov_slice = cov.iloc[cluster_items, cluster_items]
        w = self._get_ivp(cov_slice)
        cluster_var = np.dot(np.dot(w, cov_slice), w)
        return cluster_var

    def _get_ivp(self, cov: pd.DataFrame) -> np.ndarray:
        """Get inverse variance portfolio weights."""
        ivp = 1.0 / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def _get_rec_bipart(
        self,
        cov: pd.DataFrame,
        sort_idx: List[int]
    ) -> pd.Series:
        """Get recursive bisection weights."""
        w = pd.Series(1, index=sort_idx)
        cluster_items = [sort_idx]

        while len(cluster_items) > 0:
            cluster_items = [
                i[j:k] for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]

            for i in range(0, len(cluster_items), 2):
                cluster0 = cluster_items[i]
                cluster1 = cluster_items[i + 1]

                cluster_var0 = self._get_cluster_var(cov, cluster0)
                cluster_var1 = self._get_cluster_var(cov, cluster1)

                alpha = 1 - cluster_var0 / (cluster_var0 + cluster_var1)

                w[cluster0] *= alpha
                w[cluster1] *= 1 - alpha

        return w

    def optimize(
        self,
        returns: pd.DataFrame,
        covariance_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio using HRP.

        Args:
            returns: Asset returns
            covariance_matrix: Covariance matrix (computed if not provided)

        Returns:
            Optimal weights
        """
        if covariance_matrix is None:
            covariance_matrix = returns.cov()

        # Calculate correlation matrix
        corr_matrix = returns.corr()

        # Convert correlation to distance
        dist = np.sqrt(0.5 * (1 - corr_matrix))

        # Hierarchical clustering
        dist_condensed = squareform(dist, checks=False)
        link = linkage(dist_condensed, method=self.linkage_method)

        # Get quasi-diagonal order
        sort_idx = self._get_quasi_diag(link)

        # Recursive bisection
        weights = self._get_rec_bipart(covariance_matrix, sort_idx)

        return weights.to_dict()


# ============================================================================
# Mean-CVaR Optimization
# ============================================================================

class MeanCVaROptimizer:
    """
    Mean-CVaR portfolio optimization.

    Optimizes portfolios using Conditional Value at Risk.
    """

    def __init__(self, alpha: float = 0.95):
        """
        Initialize Mean-CVaR optimizer.

        Args:
            alpha: Confidence level for CVaR (e.g., 0.95 for 95%)
        """
        self.alpha = alpha

    def optimize(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        cvar_limit: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio using mean-CVaR.

        Args:
            returns: Historical returns (scenarios x assets)
            target_return: Target expected return
            cvar_limit: Maximum acceptable CVaR

        Returns:
            Optimal weights
        """
        n_assets = returns.shape[1]
        n_scenarios = returns.shape[0]

        # Variables
        w = cp.Variable(n_assets)
        z = cp.Variable(n_scenarios)
        gamma = cp.Variable()

        # Portfolio returns for each scenario
        portfolio_returns = returns.values @ w

        # CVaR definition
        cvar = gamma - (1 / ((1 - self.alpha) * n_scenarios)) * cp.sum(z)

        # Objective: maximize expected return - lambda * CVaR
        # Or minimize CVaR with return constraint
        if target_return is not None:
            objective = cp.Minimize(-cvar)
            expected_return = cp.sum(returns.values @ w) / n_scenarios
            constraints = [expected_return >= target_return]
        else:
            # Multi-objective: maximize return and minimize CVaR
            expected_return = cp.sum(returns.values @ w) / n_scenarios
            objective = cp.Maximize(expected_return - cvar)
            constraints = []

        # Add CVaR constraints
        constraints.extend([
            cp.sum(w) == 1,
            w >= 0,
            z >= 0,
            z >= gamma - portfolio_returns
        ])

        if cvar_limit is not None:
            constraints.append(cvar >= -cvar_limit)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if w.value is None:
            raise ValueError("Optimization failed")

        return dict(zip(returns.columns, w.value))


# ============================================================================
# Kelly Criterion
# ============================================================================

class KellyCriterionOptimizer:
    """
    Kelly Criterion portfolio optimization.

    Maximizes expected log wealth growth.
    """

    def __init__(self, fractional: float = 1.0):
        """
        Initialize Kelly optimizer.

        Args:
            fractional: Fraction of Kelly bet (0.5 = half-Kelly)
        """
        self.fractional = fractional

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """
        Optimize using Kelly criterion.

        Kelly weights: w* = Σ^-1 * (μ - rf)

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            risk_free_rate: Risk-free rate

        Returns:
            Optimal weights
        """
        # Excess returns
        excess_returns = expected_returns - risk_free_rate

        # Kelly weights
        kelly_weights = np.linalg.inv(covariance_matrix.values) @ excess_returns.values

        # Apply fractional Kelly
        kelly_weights *= self.fractional

        # Normalize to sum to 1
        kelly_weights = kelly_weights / kelly_weights.sum()

        return dict(zip(expected_returns.index, kelly_weights))

    def optimize_constrained(
        self,
        returns: pd.DataFrame,
        leverage_limit: float = 1.0
    ) -> Dict[str, float]:
        """
        Optimize Kelly with constraints using optimization.

        Args:
            returns: Historical returns
            leverage_limit: Maximum leverage allowed

        Returns:
            Optimal weights
        """
        n_assets = returns.shape[1]

        # Variables
        w = cp.Variable(n_assets)

        # Objective: maximize expected log growth
        # Approximation: E[log(1 + r)] ≈ E[r] - 0.5*Var[r]
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values

        expected_log_growth = mean_returns @ w - 0.5 * cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(expected_log_growth * self.fractional)

        # Constraints
        constraints = [
            cp.sum(cp.abs(w)) <= leverage_limit,  # Leverage limit
            w >= -0.5,  # Limit short positions
            w <= 0.5    # Limit long positions
        ]

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if w.value is None:
            raise ValueError("Optimization failed")

        return dict(zip(returns.columns, w.value))


# ============================================================================
# Custom Objective Function Optimizer
# ============================================================================

class CustomObjectiveOptimizer:
    """
    Portfolio optimization with custom objective functions.

    Allows users to define their own optimization objectives.
    """

    def __init__(self):
        """Initialize custom optimizer."""
        pass

    def optimize(
        self,
        objective_func: Callable,
        n_assets: int,
        constraints: Optional[List] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        initial_guess: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Optimize with custom objective function.

        Args:
            objective_func: Function to minimize (takes weights, returns scalar)
            n_assets: Number of assets
            constraints: List of constraint dictionaries
            bounds: Bounds for each weight
            initial_guess: Initial weights

        Returns:
            Optimal weights
        """
        if initial_guess is None:
            initial_guess = np.ones(n_assets) / n_assets

        if bounds is None:
            bounds = [(0, 1)] * n_assets

        if constraints is None:
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # Optimize
        result = minimize(
            objective_func,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        return result.x

    def maximize_omega_ratio(
        self,
        returns: pd.DataFrame,
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """
        Maximize Omega ratio.

        Omega = E[max(r - threshold, 0)] / E[max(threshold - r, 0)]

        Args:
            returns: Historical returns
            threshold: Return threshold

        Returns:
            Optimal weights
        """
        def omega_objective(w):
            portfolio_returns = returns.values @ w
            gains = np.maximum(portfolio_returns - threshold, 0).mean()
            losses = np.maximum(threshold - portfolio_returns, 0).mean()
            omega = gains / (losses + 1e-10)
            return -omega  # Minimize negative = maximize

        weights = self.optimize(
            omega_objective,
            n_assets=returns.shape[1]
        )

        return dict(zip(returns.columns, weights))

    def minimize_max_drawdown(
        self,
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Minimize maximum drawdown.

        Args:
            returns: Historical returns

        Returns:
            Optimal weights
        """
        def max_drawdown_objective(w):
            portfolio_returns = returns.values @ w
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return -drawdown.min()  # Minimize max drawdown

        weights = self.optimize(
            max_drawdown_objective,
            n_assets=returns.shape[1]
        )

        return dict(zip(returns.columns, weights))


# ============================================================================
# Multi-Period Optimization
# ============================================================================

class MultiPeriodOptimizer:
    """
    Multi-period portfolio optimization.

    Optimizes portfolios over multiple time periods considering
    transaction costs and dynamic constraints.
    """

    def __init__(
        self,
        n_periods: int,
        transaction_cost: float = 0.001
    ):
        """
        Initialize multi-period optimizer.

        Args:
            n_periods: Number of periods
            transaction_cost: Transaction cost (as fraction)
        """
        self.n_periods = n_periods
        self.transaction_cost = transaction_cost

    def optimize(
        self,
        expected_returns: pd.DataFrame,
        covariance_matrices: List[pd.DataFrame],
        initial_weights: Dict[str, float],
        target_return: Optional[float] = None
    ) -> List[Dict[str, float]]:
        """
        Optimize portfolio over multiple periods.

        Args:
            expected_returns: Expected returns for each period (periods x assets)
            covariance_matrices: Covariance matrix for each period
            initial_weights: Initial portfolio weights
            target_return: Target return per period

        Returns:
            List of optimal weights for each period
        """
        n_assets = expected_returns.shape[1]
        assets = expected_returns.columns

        # Variables for each period
        w = {}
        for t in range(self.n_periods):
            w[t] = cp.Variable(n_assets)

        # Transaction variables
        trades = {}
        for t in range(self.n_periods):
            trades[t] = cp.Variable(n_assets)

        # Objective: maximize terminal wealth minus transaction costs
        terminal_wealth = 1.0
        transaction_costs = 0.0

        for t in range(self.n_periods):
            # Return for period t
            period_return = expected_returns.iloc[t].values @ w[t]
            terminal_wealth *= (1 + period_return)

            # Transaction costs
            if t == 0:
                initial_w = np.array([initial_weights.get(asset, 0) for asset in assets])
                trades[t] = cp.abs(w[t] - initial_w)
            else:
                trades[t] = cp.abs(w[t] - w[t-1])

            transaction_costs += self.transaction_cost * cp.sum(trades[t])

        objective = cp.Maximize(terminal_wealth - transaction_costs)

        # Constraints
        constraints = []
        for t in range(self.n_periods):
            constraints.extend([
                cp.sum(w[t]) == 1,
                w[t] >= 0
            ])

            if target_return is not None:
                constraints.append(expected_returns.iloc[t].values @ w[t] >= target_return)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if any(w[t].value is None for t in range(self.n_periods)):
            raise ValueError("Optimization failed")

        # Return weights for each period
        results = []
        for t in range(self.n_periods):
            results.append(dict(zip(assets, w[t].value)))

        return results

    def optimize_with_rebalancing_costs(
        self,
        returns_scenarios: List[pd.DataFrame],
        initial_weights: Dict[str, float],
        rebalancing_dates: List[int]
    ) -> List[Dict[str, float]]:
        """
        Optimize considering rebalancing costs.

        Args:
            returns_scenarios: Returns for each scenario
            initial_weights: Initial weights
            rebalancing_dates: Periods when rebalancing is allowed

        Returns:
            Optimal weights for rebalancing periods
        """
        # Simplified implementation
        # In practice, would use stochastic programming

        n_assets = returns_scenarios[0].shape[1]
        assets = returns_scenarios[0].columns

        results = []
        current_weights = initial_weights

        for period in rebalancing_dates:
            # Use mean returns across scenarios for this period
            mean_returns = pd.concat(
                [scenario.iloc[period] for scenario in returns_scenarios]
            ).mean()

            # Simple mean-variance optimization
            w = cp.Variable(n_assets)
            ret = mean_returns.values @ w

            # Calculate covariance from scenarios
            scenario_returns = np.array([
                scenario.iloc[period].values for scenario in returns_scenarios
            ])
            cov = np.cov(scenario_returns.T)
            risk = cp.quad_form(w, cov)

            # Include rebalancing cost
            current_w = np.array([current_weights.get(asset, 0) for asset in assets])
            rebal_cost = self.transaction_cost * cp.sum(cp.abs(w - current_w))

            objective = cp.Maximize(ret - 0.5 * risk - rebal_cost)

            constraints = [
                cp.sum(w) == 1,
                w >= 0
            ]

            problem = cp.Problem(objective, constraints)
            problem.solve()

            if w.value is not None:
                current_weights = dict(zip(assets, w.value))
                results.append(current_weights.copy())

        return results
