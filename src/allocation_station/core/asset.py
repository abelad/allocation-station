"""Asset class definitions and asset-related utilities."""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd


class AssetClass(str, Enum):
    """Enumeration of asset classes."""

    EQUITY = "equity"
    BOND = "bond"
    CASH = "cash"
    COMMODITY = "commodity"
    REAL_ESTATE = "real_estate"
    CRYPTOCURRENCY = "cryptocurrency"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"


class AssetMetrics(BaseModel):
    """Statistical metrics for an asset."""

    expected_return: Optional[float] = Field(None, description="Expected annual return")
    volatility: Optional[float] = Field(None, description="Annual volatility (standard deviation)")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    beta: Optional[float] = Field(None, description="Beta relative to market")
    alpha: Optional[float] = Field(None, description="Alpha (excess return)")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")
    var_95: Optional[float] = Field(None, description="Value at Risk (95% confidence)")
    cvar_95: Optional[float] = Field(None, description="Conditional Value at Risk (95%)")
    skewness: Optional[float] = Field(None, description="Return distribution skewness")
    kurtosis: Optional[float] = Field(None, description="Return distribution kurtosis")


class Asset(BaseModel):
    """
    Represents a single asset in a portfolio.

    This class encapsulates all relevant information about an asset,
    including its identification, classification, and performance metrics.
    """

    symbol: str = Field(..., description="Asset ticker symbol or identifier")
    name: str = Field(..., description="Full name of the asset")
    asset_class: AssetClass = Field(..., description="Classification of the asset")

    # Additional metadata
    exchange: Optional[str] = Field(None, description="Exchange where asset is traded")
    currency: str = Field("USD", description="Currency denomination")
    isin: Optional[str] = Field(None, description="International Securities Identification Number")
    cusip: Optional[str] = Field(None, description="CUSIP identifier")

    # Performance metrics
    metrics: Optional[AssetMetrics] = Field(None, description="Statistical metrics")

    # Historical data
    price_history: Optional[pd.DataFrame] = Field(None, description="Historical price data")

    # Risk parameters
    sector: Optional[str] = Field(None, description="Sector classification")
    industry: Optional[str] = Field(None, description="Industry classification")
    market_cap: Optional[float] = Field(None, description="Market capitalization")

    # ETF-specific fields
    expense_ratio: Optional[float] = Field(None, description="Expense ratio for ETFs/mutual funds")
    holdings: Optional[List[Dict[str, float]]] = Field(None, description="ETF/fund holdings")

    # Bond-specific fields
    maturity_date: Optional[datetime] = Field(None, description="Maturity date for bonds")
    coupon_rate: Optional[float] = Field(None, description="Coupon rate for bonds")
    credit_rating: Optional[str] = Field(None, description="Credit rating for bonds")
    duration: Optional[float] = Field(None, description="Duration for bonds")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            pd.DataFrame: lambda v: v.to_dict() if v is not None else None,
            datetime: lambda v: v.isoformat() if v is not None else None,
        }

    @validator("symbol")
    def validate_symbol(cls, v):
        """Ensure symbol is uppercase and valid."""
        return v.upper().strip()

    @validator("expense_ratio")
    def validate_expense_ratio(cls, v):
        """Ensure expense ratio is a valid percentage."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Expense ratio must be between 0 and 1")
        return v

    def calculate_metrics(self, risk_free_rate: float = 0.02) -> AssetMetrics:
        """
        Calculate statistical metrics from price history.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation

        Returns:
            AssetMetrics object with calculated values
        """
        if self.price_history is None or self.price_history.empty:
            raise ValueError("Price history is required to calculate metrics")

        # Calculate returns
        returns = self.price_history['close'].pct_change().dropna()

        # Annualized metrics (assuming daily data)
        trading_days = 252
        annual_return = returns.mean() * trading_days
        annual_volatility = returns.std() * np.sqrt(trading_days)

        # Sharpe ratio
        sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        self.metrics = AssetMetrics(
            expected_return=annual_return,
            volatility=annual_volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            var_95=var_95 * np.sqrt(trading_days),  # Annualized
            cvar_95=cvar_95 * np.sqrt(trading_days),  # Annualized
            skewness=skewness,
            kurtosis=kurtosis
        )

        return self.metrics

    def get_returns(self, period: Optional[str] = None) -> pd.Series:
        """
        Get returns for the asset over a specified period.

        Args:
            period: Time period ('1d', '1w', '1m', '3m', '1y', 'all')

        Returns:
            Series of returns
        """
        if self.price_history is None:
            raise ValueError("Price history not available")

        returns = self.price_history['close'].pct_change()

        if period:
            period_map = {
                '1d': 1,
                '1w': 5,
                '1m': 21,
                '3m': 63,
                '6m': 126,
                '1y': 252,
            }
            if period in period_map:
                return returns.tail(period_map[period])

        return returns

    def to_dict(self) -> Dict[str, Any]:
        """Convert asset to dictionary representation."""
        data = self.dict(exclude={'price_history'})
        if self.price_history is not None:
            data['has_price_history'] = True
            data['price_history_shape'] = self.price_history.shape
        return data