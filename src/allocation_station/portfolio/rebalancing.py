"""Portfolio rebalancing strategies and utilities."""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from ..core import Portfolio


class RebalanceFrequency(str, Enum):
    """Rebalancing frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    THRESHOLD = "threshold"
    NEVER = "never"


class RebalancingStrategy(BaseModel):
    """
    Manages portfolio rebalancing logic and execution.
    """

    name: str = Field(..., description="Strategy name")
    frequency: RebalanceFrequency = Field(..., description="Rebalancing frequency")

    # Threshold-based parameters
    threshold: float = Field(0.05, description="Drift threshold for rebalancing")
    use_bands: bool = Field(False, description="Use tolerance bands")
    band_width: float = Field(0.02, description="Width of tolerance bands")

    # Cost considerations
    transaction_cost: float = Field(0.001, description="Transaction cost as percentage")
    min_trade_size: float = Field(100, description="Minimum trade size in dollars")
    tax_aware: bool = Field(False, description="Consider tax implications")

    # Timing constraints
    blackout_dates: List[datetime] = Field(default_factory=list, description="Dates to avoid rebalancing")
    last_rebalance_date: Optional[datetime] = Field(None, description="Last rebalance date")

    def should_rebalance(
        self,
        portfolio: Portfolio,
        current_date: Optional[datetime] = None
    ) -> bool:
        """
        Determine if portfolio should be rebalanced.

        Args:
            portfolio: Current portfolio
            current_date: Current date for time-based checks

        Returns:
            True if rebalancing is needed
        """
        if self.frequency == RebalanceFrequency.NEVER:
            return False

        current_date = current_date or datetime.now()

        # Check blackout dates
        if current_date.date() in [d.date() for d in self.blackout_dates]:
            return False

        # Time-based rebalancing
        if self.frequency != RebalanceFrequency.THRESHOLD:
            return self._check_time_based_rebalance(current_date)

        # Threshold-based rebalancing
        return self._check_threshold_rebalance(portfolio)

    def _check_time_based_rebalance(self, current_date: datetime) -> bool:
        """Check if enough time has passed for rebalancing."""
        if not self.last_rebalance_date:
            return True

        days_since = (current_date - self.last_rebalance_date).days

        frequency_days = {
            RebalanceFrequency.DAILY: 1,
            RebalanceFrequency.WEEKLY: 7,
            RebalanceFrequency.MONTHLY: 30,
            RebalanceFrequency.QUARTERLY: 90,
            RebalanceFrequency.SEMI_ANNUAL: 180,
            RebalanceFrequency.ANNUAL: 365,
        }

        required_days = frequency_days.get(self.frequency, 30)
        return days_since >= required_days

    def _check_threshold_rebalance(self, portfolio: Portfolio) -> bool:
        """Check if portfolio drift exceeds threshold."""
        if not portfolio.target_allocation or not portfolio.allocation:
            return False

        current_weights = portfolio.calculate_weights()

        for symbol, target_weight in portfolio.target_allocation.items():
            current_weight = current_weights.get(symbol, 0)
            drift = abs(current_weight - target_weight)

            if self.use_bands:
                # Check if outside tolerance band
                if drift > target_weight * self.band_width:
                    return True
            else:
                # Simple threshold check
                if drift > self.threshold:
                    return True

        return False

    def calculate_trades(
        self,
        portfolio: Portfolio,
        target_allocation: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate rebalancing trades.

        Args:
            portfolio: Current portfolio
            target_allocation: Target weights

        Returns:
            Dictionary of trades with amounts and costs
        """
        if not portfolio.allocation:
            return {}

        total_value = sum(portfolio.allocation.values.values())
        trades = {}

        for symbol, target_weight in target_allocation.items():
            target_value = total_value * target_weight
            current_value = portfolio.allocation.values.get(symbol, 0)
            trade_value = target_value - current_value

            # Check minimum trade size
            if abs(trade_value) >= self.min_trade_size:
                trade_cost = abs(trade_value) * self.transaction_cost

                trades[symbol] = {
                    'amount': trade_value,
                    'shares': None,  # To be calculated based on current prices
                    'cost': trade_cost,
                    'type': 'buy' if trade_value > 0 else 'sell'
                }

        return trades

    def optimize_trades(
        self,
        trades: Dict[str, Dict[str, float]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize trades to minimize costs while achieving target.

        Args:
            trades: Initial trade calculations
            constraints: Additional constraints

        Returns:
            Optimized trades
        """
        # Calculate total transaction cost
        total_cost = sum(t['cost'] for t in trades.values())

        # Simple optimization: cancel small trades if cost is high
        optimized = {}
        for symbol, trade in trades.items():
            cost_ratio = trade['cost'] / abs(trade['amount'])
            if cost_ratio < 0.02:  # Less than 2% cost
                optimized[symbol] = trade

        return optimized