"""Withdrawal strategies for portfolio management."""

from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd
from ..core import Portfolio


class WithdrawalMethod(str, Enum):
    """Types of withdrawal methods."""

    FIXED_AMOUNT = "fixed_amount"  # Fixed dollar amount
    FIXED_PERCENTAGE = "fixed_percentage"  # Fixed percentage of portfolio
    VARIABLE_PERCENTAGE = "variable_percentage"  # Variable based on performance
    FOUR_PERCENT_RULE = "four_percent_rule"  # Traditional 4% rule
    GUYTON_KLINGER = "guyton_klinger"  # Dynamic with guardrails
    CONSTANT_DOLLAR = "constant_dollar"  # Inflation-adjusted fixed amount
    FLOOR_CEILING = "floor_ceiling"  # With min/max bounds
    BUCKET_STRATEGY = "bucket_strategy"  # Time-segmented buckets
    RMD = "rmd"  # Required Minimum Distribution


class WithdrawalRule(BaseModel):
    """Rule for dynamic withdrawal adjustments."""

    name: str = Field(..., description="Rule name")
    trigger_condition: str = Field(..., description="Condition that triggers rule")
    adjustment_factor: float = Field(..., description="Adjustment to apply")
    min_withdrawal: Optional[float] = Field(None, description="Minimum withdrawal amount")
    max_withdrawal: Optional[float] = Field(None, description="Maximum withdrawal amount")


class WithdrawalStrategy(BaseModel):
    """
    Manages portfolio withdrawal strategies for retirement planning.

    This class implements various withdrawal strategies to help investors
    manage portfolio decumulation while maintaining sustainability.
    """

    name: str = Field(..., description="Strategy name")
    description: Optional[str] = Field(None, description="Strategy description")
    method: WithdrawalMethod = Field(..., description="Withdrawal method")

    # Base parameters
    initial_withdrawal_rate: float = Field(0.04, description="Initial withdrawal rate (e.g., 0.04 for 4%)")
    withdrawal_frequency: str = Field("monthly", description="Frequency of withdrawals")
    inflation_adjustment: bool = Field(True, description="Adjust for inflation")
    inflation_rate: float = Field(0.025, description="Expected inflation rate")

    # Dynamic adjustment parameters
    adjustment_rules: List[WithdrawalRule] = Field(default_factory=list, description="Dynamic adjustment rules")
    guardrails: Optional[Tuple[float, float]] = Field(None, description="Min/max withdrawal rates")

    # Floor and ceiling
    withdrawal_floor: Optional[float] = Field(None, description="Minimum withdrawal amount")
    withdrawal_ceiling: Optional[float] = Field(None, description="Maximum withdrawal amount")

    # Bucket strategy parameters
    buckets: Optional[List[Dict[str, Any]]] = Field(None, description="Bucket definitions")
    bucket_refill_threshold: float = Field(0.5, description="Threshold to refill bucket")

    # Tracking
    withdrawal_history: List[Dict[str, Any]] = Field(default_factory=list, description="Historical withdrawals")
    cumulative_withdrawn: float = Field(0, description="Total amount withdrawn")

    @validator("initial_withdrawal_rate")
    def validate_withdrawal_rate(cls, v):
        """Ensure withdrawal rate is reasonable."""
        if v <= 0 or v > 0.2:  # Max 20% withdrawal rate
            raise ValueError("Withdrawal rate must be between 0 and 0.2 (20%)")
        return v

    def calculate_withdrawal(
        self,
        portfolio: Portfolio,
        current_date: Optional[datetime] = None,
        market_conditions: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate the withdrawal amount for the current period.

        Args:
            portfolio: Current portfolio
            current_date: Current date for calculation
            market_conditions: Current market conditions

        Returns:
            Withdrawal amount
        """
        if self.method == WithdrawalMethod.FIXED_AMOUNT:
            return self._fixed_amount_withdrawal(portfolio)
        elif self.method == WithdrawalMethod.FIXED_PERCENTAGE:
            return self._fixed_percentage_withdrawal(portfolio)
        elif self.method == WithdrawalMethod.FOUR_PERCENT_RULE:
            return self._four_percent_rule_withdrawal(portfolio)
        elif self.method == WithdrawalMethod.GUYTON_KLINGER:
            return self._guyton_klinger_withdrawal(portfolio, market_conditions)
        elif self.method == WithdrawalMethod.CONSTANT_DOLLAR:
            return self._constant_dollar_withdrawal(portfolio)
        elif self.method == WithdrawalMethod.FLOOR_CEILING:
            return self._floor_ceiling_withdrawal(portfolio)
        elif self.method == WithdrawalMethod.BUCKET_STRATEGY:
            return self._bucket_strategy_withdrawal(portfolio, current_date)
        else:
            return self._default_withdrawal(portfolio)

    def _fixed_amount_withdrawal(self, portfolio: Portfolio) -> float:
        """Calculate fixed amount withdrawal."""
        base_amount = portfolio.initial_value * self.initial_withdrawal_rate

        if self.inflation_adjustment and self.withdrawal_history:
            years_elapsed = len(self.withdrawal_history) / self._get_periods_per_year()
            inflation_factor = (1 + self.inflation_rate) ** years_elapsed
            return base_amount * inflation_factor

        return base_amount

    def _fixed_percentage_withdrawal(self, portfolio: Portfolio) -> float:
        """Calculate fixed percentage withdrawal."""
        current_value = portfolio.current_value or portfolio.initial_value
        return current_value * self.initial_withdrawal_rate

    def _four_percent_rule_withdrawal(self, portfolio: Portfolio) -> float:
        """Implement traditional 4% rule with inflation adjustment."""
        if not self.withdrawal_history:
            # First withdrawal
            return portfolio.initial_value * 0.04

        # Subsequent withdrawals - adjust for inflation
        last_withdrawal = self.withdrawal_history[-1]['amount']
        if self.inflation_adjustment:
            return last_withdrawal * (1 + self.inflation_rate / self._get_periods_per_year())

        return last_withdrawal

    def _guyton_klinger_withdrawal(
        self,
        portfolio: Portfolio,
        market_conditions: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Implement Guyton-Klinger dynamic withdrawal strategy.

        This strategy adjusts withdrawals based on portfolio performance
        and market conditions using guardrails.
        """
        current_value = portfolio.current_value or portfolio.initial_value

        if not self.withdrawal_history:
            # Initial withdrawal
            return current_value * self.initial_withdrawal_rate

        # Get last withdrawal
        last_withdrawal = self.withdrawal_history[-1]['amount']
        base_withdrawal = last_withdrawal

        # Apply inflation adjustment
        if self.inflation_adjustment:
            base_withdrawal *= (1 + self.inflation_rate / self._get_periods_per_year())

        # Calculate current withdrawal rate
        current_rate = base_withdrawal / current_value

        # Apply guardrails
        if self.guardrails:
            min_rate, max_rate = self.guardrails

            # Capital preservation rule - reduce if rate too high
            if current_rate > max_rate:
                base_withdrawal = current_value * max_rate * 0.9  # 10% reduction

            # Prosperity rule - increase if rate too low
            elif current_rate < min_rate:
                base_withdrawal = current_value * min_rate * 1.1  # 10% increase

        # Apply portfolio management rule (skip inflation if portfolio declined)
        if market_conditions and market_conditions.get('portfolio_return', 0) < 0:
            base_withdrawal = last_withdrawal  # No inflation adjustment

        return self._apply_floor_ceiling(base_withdrawal)

    def _constant_dollar_withdrawal(self, portfolio: Portfolio) -> float:
        """Calculate constant dollar (inflation-adjusted) withdrawal."""
        base_amount = portfolio.initial_value * self.initial_withdrawal_rate

        if self.withdrawal_history:
            years_elapsed = len(self.withdrawal_history) / self._get_periods_per_year()
            inflation_factor = (1 + self.inflation_rate) ** years_elapsed
            return base_amount * inflation_factor

        return base_amount

    def _floor_ceiling_withdrawal(self, portfolio: Portfolio) -> float:
        """Calculate withdrawal with floor and ceiling constraints."""
        current_value = portfolio.current_value or portfolio.initial_value
        base_withdrawal = current_value * self.initial_withdrawal_rate

        return self._apply_floor_ceiling(base_withdrawal)

    def _bucket_strategy_withdrawal(
        self,
        portfolio: Portfolio,
        current_date: Optional[datetime] = None
    ) -> float:
        """
        Implement bucket strategy for withdrawals.

        Divides portfolio into time-based buckets (e.g., short, medium, long-term).
        """
        if not self.buckets:
            # Default to simple fixed withdrawal if buckets not defined
            return self._fixed_percentage_withdrawal(portfolio)

        # Withdraw from first bucket (cash/short-term)
        if self.buckets[0].get('balance', 0) > 0:
            withdrawal_amount = min(
                self.buckets[0]['target_withdrawal'],
                self.buckets[0]['balance']
            )

            # Update bucket balance
            self.buckets[0]['balance'] -= withdrawal_amount

            # Check if bucket needs refilling
            if self.buckets[0]['balance'] < self.buckets[0]['target_withdrawal'] * self.bucket_refill_threshold:
                self._refill_bucket(0, portfolio)

            return withdrawal_amount

        # If first bucket empty, refill and try again
        self._refill_bucket(0, portfolio)
        return self.buckets[0].get('target_withdrawal', 0)

    def _refill_bucket(self, bucket_index: int, portfolio: Portfolio):
        """Refill a bucket from the next bucket or portfolio."""
        if bucket_index >= len(self.buckets) - 1:
            return  # Can't refill last bucket

        target_balance = self.buckets[bucket_index]['target_balance']
        current_balance = self.buckets[bucket_index].get('balance', 0)
        needed = target_balance - current_balance

        # Transfer from next bucket
        next_bucket = self.buckets[bucket_index + 1]
        available = next_bucket.get('balance', 0)
        transfer = min(needed, available)

        self.buckets[bucket_index]['balance'] = current_balance + transfer
        next_bucket['balance'] = available - transfer

    def _default_withdrawal(self, portfolio: Portfolio) -> float:
        """Default withdrawal calculation."""
        current_value = portfolio.current_value or portfolio.initial_value
        return current_value * self.initial_withdrawal_rate

    def _apply_floor_ceiling(self, amount: float) -> float:
        """Apply floor and ceiling constraints to withdrawal amount."""
        if self.withdrawal_floor:
            amount = max(amount, self.withdrawal_floor)
        if self.withdrawal_ceiling:
            amount = min(amount, self.withdrawal_ceiling)
        return amount

    def _get_periods_per_year(self) -> int:
        """Get number of withdrawal periods per year."""
        frequency_map = {
            'monthly': 12,
            'quarterly': 4,
            'semi-annual': 2,
            'annual': 1,
            'weekly': 52,
        }
        return frequency_map.get(self.withdrawal_frequency, 12)

    def record_withdrawal(self, amount: float, date: datetime, portfolio_value: float):
        """
        Record a withdrawal transaction.

        Args:
            amount: Withdrawal amount
            date: Date of withdrawal
            portfolio_value: Portfolio value after withdrawal
        """
        withdrawal_record = {
            'date': date.isoformat(),
            'amount': amount,
            'portfolio_value_after': portfolio_value,
            'cumulative_withdrawn': self.cumulative_withdrawn + amount,
        }

        self.withdrawal_history.append(withdrawal_record)
        self.cumulative_withdrawn += amount

    def calculate_sustainability(
        self,
        portfolio: Portfolio,
        time_horizon: int = 30,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate withdrawal sustainability metrics.

        Args:
            portfolio: Current portfolio
            time_horizon: Years to project
            confidence_level: Confidence level for success probability

        Returns:
            Dictionary with sustainability metrics
        """
        current_value = portfolio.current_value or portfolio.initial_value
        annual_withdrawal = self.calculate_withdrawal(portfolio) * self._get_periods_per_year()

        # Simple sustainability check
        years_sustainable = current_value / annual_withdrawal if annual_withdrawal > 0 else float('inf')

        # Success probability (simplified - would use Monte Carlo in practice)
        withdrawal_rate = annual_withdrawal / current_value
        success_probability = max(0, min(1, 1 - (withdrawal_rate - 0.03) * 10))

        return {
            'years_sustainable': years_sustainable,
            'success_probability': success_probability,
            'current_withdrawal_rate': withdrawal_rate,
            'annual_withdrawal': annual_withdrawal,
            'total_withdrawn': self.cumulative_withdrawn,
            'remaining_value': current_value,
        }

    def optimize_withdrawal(
        self,
        portfolio: Portfolio,
        objectives: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Optimize withdrawal amount based on multiple objectives.

        Args:
            portfolio: Current portfolio
            objectives: Weighted objectives (e.g., {'sustainability': 0.5, 'income': 0.5})
            constraints: Additional constraints

        Returns:
            Optimized withdrawal amount
        """
        # This would implement multi-objective optimization
        # For now, return standard calculation
        return self.calculate_withdrawal(portfolio)