"""Portfolio class definition and portfolio management utilities."""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from .asset import Asset, AssetClass


class PortfolioMetrics(BaseModel):
    """Performance and risk metrics for a portfolio."""

    total_value: float = Field(..., description="Total portfolio value")
    returns: float = Field(..., description="Portfolio returns")
    volatility: float = Field(..., description="Portfolio volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    var_95: float = Field(..., description="Value at Risk (95%)")
    cvar_95: float = Field(..., description="Conditional Value at Risk (95%)")
    beta: Optional[float] = Field(None, description="Portfolio beta")
    alpha: Optional[float] = Field(None, description="Portfolio alpha")
    tracking_error: Optional[float] = Field(None, description="Tracking error vs benchmark")
    information_ratio: Optional[float] = Field(None, description="Information ratio")


class Allocation(BaseModel):
    """Represents the allocation of assets in a portfolio."""

    weights: Dict[str, float] = Field(..., description="Asset weights by symbol")
    quantities: Dict[str, float] = Field(..., description="Asset quantities by symbol")
    values: Dict[str, float] = Field(..., description="Asset values by symbol")

    @validator("weights")
    def validate_weights(cls, v):
        """Ensure weights sum to approximately 1."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.001:
            # Normalize if close but not exact
            if 0.99 <= total <= 1.01:
                return {k: w/total for k, w in v.items()}
            else:
                raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v


class Portfolio(BaseModel):
    """
    Represents a portfolio of assets with allocation and performance tracking.

    This class manages a collection of assets, their allocations, and provides
    methods for portfolio analysis, rebalancing, and performance calculation.
    """

    name: str = Field(..., description="Portfolio name")
    description: Optional[str] = Field(None, description="Portfolio description")

    # Portfolio composition
    assets: Dict[str, Asset] = Field(default_factory=dict, description="Assets in portfolio")
    allocation: Optional[Allocation] = Field(None, description="Current allocation")

    # Portfolio parameters
    initial_value: float = Field(100000, description="Initial portfolio value")
    current_value: Optional[float] = Field(None, description="Current portfolio value")
    cash_balance: float = Field(0, description="Cash balance")

    # Dates
    inception_date: datetime = Field(default_factory=datetime.now, description="Portfolio inception")
    last_rebalance_date: Optional[datetime] = Field(None, description="Last rebalance date")

    # Performance tracking
    value_history: Optional[pd.DataFrame] = Field(None, description="Historical portfolio values")
    transaction_history: List[Dict[str, Any]] = Field(default_factory=list, description="Transaction log")

    # Risk parameters
    target_allocation: Optional[Dict[str, float]] = Field(None, description="Target allocation weights")
    rebalance_threshold: float = Field(0.05, description="Rebalancing threshold (5%)")

    # Metrics
    metrics: Optional[PortfolioMetrics] = Field(None, description="Portfolio metrics")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            pd.DataFrame: lambda v: v.to_dict() if v is not None else None,
            datetime: lambda v: v.isoformat() if v is not None else None,
        }

    def add_asset(self, asset: Asset, weight: Optional[float] = None, quantity: Optional[float] = None):
        """
        Add an asset to the portfolio.

        Args:
            asset: Asset to add
            weight: Target weight in portfolio (0-1)
            quantity: Number of shares/units
        """
        self.assets[asset.symbol] = asset

        if weight is not None and self.target_allocation is not None:
            self.target_allocation[asset.symbol] = weight
            # Renormalize weights
            total = sum(self.target_allocation.values())
            self.target_allocation = {k: v/total for k, v in self.target_allocation.items()}

    def remove_asset(self, symbol: str):
        """Remove an asset from the portfolio."""
        if symbol in self.assets:
            del self.assets[symbol]
            if self.target_allocation and symbol in self.target_allocation:
                del self.target_allocation[symbol]
                # Renormalize weights
                total = sum(self.target_allocation.values())
                if total > 0:
                    self.target_allocation = {k: v/total for k, v in self.target_allocation.items()}

    def calculate_weights(self) -> Dict[str, float]:
        """Calculate current portfolio weights based on asset values."""
        if not self.allocation or not self.allocation.values:
            return {}

        total_value = sum(self.allocation.values.values())
        if total_value == 0:
            return {}

        return {symbol: value/total_value for symbol, value in self.allocation.values.items()}

    def get_asset_class_allocation(self) -> Dict[AssetClass, float]:
        """Get allocation by asset class."""
        class_allocation = {}
        weights = self.calculate_weights()

        for symbol, weight in weights.items():
            if symbol in self.assets:
                asset_class = self.assets[symbol].asset_class
                class_allocation[asset_class] = class_allocation.get(asset_class, 0) + weight

        return class_allocation

    def calculate_returns(self, period: str = "1y") -> pd.Series:
        """
        Calculate portfolio returns over a specified period.

        Args:
            period: Time period for returns calculation

        Returns:
            Series of portfolio returns
        """
        if self.value_history is None:
            raise ValueError("Value history required to calculate returns")

        return self.value_history['value'].pct_change()

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix of asset returns."""
        returns_data = {}

        for symbol, asset in self.assets.items():
            if asset.price_history is not None:
                returns_data[symbol] = asset.get_returns()

        if not returns_data:
            raise ValueError("No price history available for assets")

        returns_df = pd.DataFrame(returns_data)
        return returns_df.corr()

    def calculate_covariance_matrix(self) -> pd.DataFrame:
        """Calculate covariance matrix of asset returns."""
        returns_data = {}

        for symbol, asset in self.assets.items():
            if asset.price_history is not None:
                returns_data[symbol] = asset.get_returns()

        if not returns_data:
            raise ValueError("No price history available for assets")

        returns_df = pd.DataFrame(returns_data)
        # Annualize covariance (assuming daily returns)
        return returns_df.cov() * 252

    def calculate_portfolio_metrics(self, risk_free_rate: float = 0.02) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics.

        Args:
            risk_free_rate: Annual risk-free rate

        Returns:
            PortfolioMetrics object
        """
        if not self.assets or not self.allocation:
            raise ValueError("Portfolio must have assets and allocation")

        # Get weights and returns
        weights = np.array(list(self.calculate_weights().values()))

        # Collect returns for each asset
        returns_list = []
        for symbol in self.allocation.weights.keys():
            if symbol in self.assets:
                asset = self.assets[symbol]
                if asset.price_history is not None:
                    returns_list.append(asset.get_returns())

        if not returns_list:
            raise ValueError("No price history available for portfolio assets")

        # Create returns dataframe
        returns_df = pd.concat(returns_list, axis=1)
        returns_df.columns = list(self.allocation.weights.keys())

        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Annual metrics
        trading_days = 252
        annual_return = portfolio_returns.mean() * trading_days
        annual_volatility = portfolio_returns.std() * np.sqrt(trading_days)

        # Sharpe ratio
        sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(trading_days)
        sortino = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # VaR and CVaR
        var_95 = portfolio_returns.quantile(0.05) * np.sqrt(trading_days)
        cvar_95 = portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean() * np.sqrt(trading_days)

        # Calculate total value
        total_value = sum(self.allocation.values.values()) if self.allocation else self.initial_value

        self.metrics = PortfolioMetrics(
            total_value=total_value,
            returns=annual_return,
            volatility=annual_volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95
        )

        return self.metrics

    def needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing based on drift from target."""
        if not self.target_allocation or not self.allocation:
            return False

        current_weights = self.calculate_weights()

        for symbol, target_weight in self.target_allocation.items():
            current_weight = current_weights.get(symbol, 0)
            drift = abs(current_weight - target_weight)
            if drift > self.rebalance_threshold:
                return True

        return False

    def get_rebalancing_trades(self) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance to target allocation.

        Returns:
            Dictionary of symbol to trade amount (positive = buy, negative = sell)
        """
        if not self.target_allocation or not self.allocation:
            return {}

        total_value = sum(self.allocation.values.values())
        trades = {}

        for symbol, target_weight in self.target_allocation.items():
            target_value = total_value * target_weight
            current_value = self.allocation.values.get(symbol, 0)
            trade_value = target_value - current_value

            if abs(trade_value) > 0.01:  # Minimum trade threshold
                trades[symbol] = trade_value

        return trades

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "assets": {symbol: asset.to_dict() for symbol, asset in self.assets.items()},
            "allocation": self.allocation.dict() if self.allocation else None,
            "initial_value": self.initial_value,
            "current_value": self.current_value,
            "target_allocation": self.target_allocation,
            "metrics": self.metrics.dict() if self.metrics else None,
            "inception_date": self.inception_date.isoformat(),
        }