"""Portfolio allocation strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from ..core import Portfolio, Asset, AssetClass


class StrategyType(str, Enum):
    """Types of allocation strategies."""

    STRATEGIC = "strategic"  # Long-term, fixed allocation
    TACTICAL = "tactical"  # Dynamic, market-based allocation
    DYNAMIC = "dynamic"  # Rule-based dynamic allocation
    CONSTANT_MIX = "constant_mix"  # Constant rebalancing to target
    BUY_AND_HOLD = "buy_and_hold"  # No rebalancing
    CPPI = "cppi"  # Constant Proportion Portfolio Insurance
    TARGET_DATE = "target_date"  # Glide path based on target date
    RISK_PARITY = "risk_parity"  # Equal risk contribution
    MEAN_VARIANCE = "mean_variance"  # Modern Portfolio Theory optimization


class AllocationRule(BaseModel):
    """Rule for dynamic allocation adjustments."""

    name: str = Field(..., description="Rule name")
    condition: str = Field(..., description="Condition expression")
    action: Dict[str, Any] = Field(..., description="Action to take when condition is met")
    priority: int = Field(0, description="Rule priority (higher executes first)")


class AllocationStrategy(BaseModel, ABC):
    """
    Abstract base class for portfolio allocation strategies.

    This class defines the interface for all allocation strategies
    and provides common functionality for strategy management.
    """

    name: str = Field(..., description="Strategy name")
    description: Optional[str] = Field(None, description="Strategy description")
    strategy_type: StrategyType = Field(..., description="Type of strategy")

    # Base allocations
    target_allocation: Dict[str, float] = Field(..., description="Target asset allocation")
    asset_class_targets: Optional[Dict[AssetClass, float]] = Field(None, description="Asset class targets")

    # Constraints
    min_allocation: Dict[str, float] = Field(default_factory=dict, description="Minimum allocation per asset")
    max_allocation: Dict[str, float] = Field(default_factory=dict, description="Maximum allocation per asset")

    # Risk parameters
    max_volatility: Optional[float] = Field(None, description="Maximum portfolio volatility")
    max_drawdown: Optional[float] = Field(None, description="Maximum acceptable drawdown")
    risk_tolerance: float = Field(0.5, description="Risk tolerance (0=conservative, 1=aggressive)")

    # Rules for dynamic strategies
    allocation_rules: List[AllocationRule] = Field(default_factory=list, description="Dynamic allocation rules")

    # Performance tracking
    backtest_results: Optional[Dict[str, Any]] = Field(None, description="Backtesting results")
    live_performance: Optional[Dict[str, Any]] = Field(None, description="Live performance metrics")

    @validator("target_allocation")
    def validate_allocation(cls, v):
        """Ensure allocations sum to 1."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.001:
            if 0.99 <= total <= 1.01:
                return {k: w/total for k, w in v.items()}
            else:
                raise ValueError(f"Target allocation must sum to 1.0, got {total}")
        return v

    @abstractmethod
    def calculate_allocation(
        self,
        portfolio: Portfolio,
        market_data: Optional[pd.DataFrame] = None,
        current_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate the target allocation for the portfolio.

        Args:
            portfolio: Current portfolio
            market_data: Market data for decision making
            current_date: Current date for time-based decisions

        Returns:
            Dictionary of symbol to target weight
        """
        pass

    @abstractmethod
    def should_rebalance(
        self,
        portfolio: Portfolio,
        market_data: Optional[pd.DataFrame] = None
    ) -> bool:
        """
        Determine if the portfolio should be rebalanced.

        Args:
            portfolio: Current portfolio
            market_data: Current market data

        Returns:
            True if rebalancing is needed
        """
        pass

    def apply_constraints(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """
        Apply min/max constraints to allocation.

        Args:
            allocation: Proposed allocation

        Returns:
            Constrained allocation
        """
        constrained = allocation.copy()

        # Apply min/max constraints
        for symbol, weight in allocation.items():
            if symbol in self.min_allocation:
                constrained[symbol] = max(weight, self.min_allocation[symbol])
            if symbol in self.max_allocation:
                constrained[symbol] = min(weight, self.max_allocation[symbol])

        # Renormalize after constraints
        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v/total for k, v in constrained.items()}

        return constrained

    def evaluate_risk(self, portfolio: Portfolio) -> Dict[str, float]:
        """
        Evaluate portfolio risk metrics against strategy limits.

        Args:
            portfolio: Portfolio to evaluate

        Returns:
            Dictionary of risk metrics and violations
        """
        metrics = portfolio.calculate_portfolio_metrics()
        violations = {}

        if self.max_volatility and metrics.volatility > self.max_volatility:
            violations['volatility'] = metrics.volatility - self.max_volatility

        if self.max_drawdown and abs(metrics.max_drawdown) > abs(self.max_drawdown):
            violations['drawdown'] = abs(metrics.max_drawdown) - abs(self.max_drawdown)

        return violations


class StrategicAllocation(AllocationStrategy):
    """
    Strategic (buy-and-hold) allocation strategy.

    Maintains a fixed allocation with periodic rebalancing.
    """

    strategy_type: StrategyType = Field(StrategyType.STRATEGIC, const=True)
    rebalance_frequency: str = Field("quarterly", description="Rebalancing frequency")
    rebalance_threshold: float = Field(0.05, description="Drift threshold for rebalancing")

    def calculate_allocation(
        self,
        portfolio: Portfolio,
        market_data: Optional[pd.DataFrame] = None,
        current_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Return the fixed target allocation."""
        return self.apply_constraints(self.target_allocation)

    def should_rebalance(
        self,
        portfolio: Portfolio,
        market_data: Optional[pd.DataFrame] = None
    ) -> bool:
        """Check if portfolio has drifted beyond threshold."""
        if not portfolio.allocation:
            return True

        current_weights = portfolio.calculate_weights()

        for symbol, target_weight in self.target_allocation.items():
            current_weight = current_weights.get(symbol, 0)
            drift = abs(current_weight - target_weight)
            if drift > self.rebalance_threshold:
                return True

        return False


class TacticalAllocation(AllocationStrategy):
    """
    Tactical allocation strategy with dynamic adjustments.

    Adjusts allocation based on market conditions and signals.
    """

    strategy_type: StrategyType = Field(StrategyType.TACTICAL, const=True)

    # Tactical parameters
    momentum_window: int = Field(20, description="Momentum calculation window")
    volatility_window: int = Field(30, description="Volatility calculation window")
    signal_threshold: float = Field(0.02, description="Signal threshold for allocation change")

    # Tactical tilts
    max_tilt: float = Field(0.2, description="Maximum tilt from strategic allocation")
    tilt_factors: Dict[str, str] = Field(default_factory=dict, description="Factors for tilting")

    def calculate_allocation(
        self,
        portfolio: Portfolio,
        market_data: Optional[pd.DataFrame] = None,
        current_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate tactical allocation based on market signals.

        Args:
            portfolio: Current portfolio
            market_data: Market data for signals
            current_date: Current date

        Returns:
            Tactical allocation weights
        """
        base_allocation = self.target_allocation.copy()

        if market_data is None:
            return self.apply_constraints(base_allocation)

        # Calculate market signals
        signals = self._calculate_signals(market_data)

        # Adjust allocation based on signals
        tactical_allocation = {}
        for symbol, base_weight in base_allocation.items():
            if symbol in signals:
                # Apply tactical tilt based on signal
                signal_strength = signals[symbol]
                tilt = min(self.max_tilt * signal_strength, self.max_tilt)
                tactical_allocation[symbol] = base_weight * (1 + tilt)
            else:
                tactical_allocation[symbol] = base_weight

        # Normalize to sum to 1
        total = sum(tactical_allocation.values())
        if total > 0:
            tactical_allocation = {k: v/total for k, v in tactical_allocation.items()}

        return self.apply_constraints(tactical_allocation)

    def _calculate_signals(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate market signals for tactical adjustments.

        Args:
            market_data: Market data

        Returns:
            Dictionary of symbol to signal strength (-1 to 1)
        """
        signals = {}

        # Calculate momentum signals
        if 'returns' in market_data.columns:
            momentum = market_data['returns'].rolling(self.momentum_window).mean()
            latest_momentum = momentum.iloc[-1] if not momentum.empty else 0

            # Convert to signal (-1 to 1)
            if abs(latest_momentum) > self.signal_threshold:
                signals['momentum'] = np.tanh(latest_momentum / self.signal_threshold)

        # Calculate volatility signals
        if 'returns' in market_data.columns:
            volatility = market_data['returns'].rolling(self.volatility_window).std()
            latest_vol = volatility.iloc[-1] if not volatility.empty else 0
            mean_vol = volatility.mean()

            if mean_vol > 0:
                vol_signal = (mean_vol - latest_vol) / mean_vol
                signals['volatility'] = np.clip(vol_signal, -1, 1)

        return signals

    def should_rebalance(
        self,
        portfolio: Portfolio,
        market_data: Optional[pd.DataFrame] = None
    ) -> bool:
        """
        Check if tactical rebalancing is needed based on signals.

        Args:
            portfolio: Current portfolio
            market_data: Market data

        Returns:
            True if rebalancing is needed
        """
        if market_data is None:
            return False

        # Calculate current tactical allocation
        current_weights = portfolio.calculate_weights()
        target_tactical = self.calculate_allocation(portfolio, market_data)

        # Check if allocation has changed significantly
        for symbol, target_weight in target_tactical.items():
            current_weight = current_weights.get(symbol, 0)
            if abs(current_weight - target_weight) > self.signal_threshold:
                return True

        return False


class RiskParityStrategy(AllocationStrategy):
    """
    Risk parity allocation strategy.

    Allocates capital such that each asset contributes equally to portfolio risk.
    """

    strategy_type: StrategyType = Field(StrategyType.RISK_PARITY, const=True)
    risk_measure: str = Field("volatility", description="Risk measure to use")
    leverage_allowed: bool = Field(False, description="Allow leverage in allocation")
    max_leverage: float = Field(1.0, description="Maximum leverage if allowed")

    def calculate_allocation(
        self,
        portfolio: Portfolio,
        market_data: Optional[pd.DataFrame] = None,
        current_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate risk parity allocation.

        Args:
            portfolio: Portfolio with assets
            market_data: Historical market data
            current_date: Current date

        Returns:
            Risk parity weights
        """
        # Get covariance matrix
        cov_matrix = portfolio.calculate_covariance_matrix()

        if cov_matrix.empty:
            # Fallback to equal weight if no data
            n_assets = len(portfolio.assets)
            equal_weight = 1.0 / n_assets
            return {symbol: equal_weight for symbol in portfolio.assets.keys()}

        # Calculate risk parity weights
        symbols = list(cov_matrix.columns)
        n = len(symbols)

        # Initial guess (equal weights)
        weights = np.ones(n) / n

        # Iterative solution for risk parity
        for _ in range(100):  # Max iterations
            # Calculate portfolio risk contributions
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol

            # Target equal risk contribution
            target_contrib = np.ones(n) / n

            # Update weights
            weights = weights * (target_contrib / contrib)
            weights = weights / weights.sum()

        # Convert to dictionary
        allocation = {symbols[i]: weights[i] for i in range(n)}

        return self.apply_constraints(allocation)

    def should_rebalance(
        self,
        portfolio: Portfolio,
        market_data: Optional[pd.DataFrame] = None
    ) -> bool:
        """Check if risk contributions have deviated from parity."""
        if not portfolio.allocation:
            return True

        # Calculate current risk contributions
        weights = portfolio.calculate_weights()
        cov_matrix = portfolio.calculate_covariance_matrix()

        if cov_matrix.empty:
            return False

        weights_array = np.array([weights.get(s, 0) for s in cov_matrix.columns])
        portfolio_vol = np.sqrt(weights_array @ cov_matrix @ weights_array)

        if portfolio_vol == 0:
            return False

        marginal_contrib = cov_matrix @ weights_array
        contrib = weights_array * marginal_contrib / portfolio_vol

        # Check if contributions are approximately equal
        target_contrib = 1.0 / len(weights)
        max_deviation = max(abs(c - target_contrib) for c in contrib)

        return max_deviation > 0.05  # 5% deviation threshold