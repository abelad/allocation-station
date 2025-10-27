"""Enhanced asset classes with support for derivatives, REITs, crypto, commodities, and alternatives."""

from enum import Enum
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, date
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd

from .asset import Asset, AssetClass, AssetMetrics


class OptionType(str, Enum):
    """Types of options."""
    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    """Option exercise styles."""
    AMERICAN = "american"
    EUROPEAN = "european"
    BERMUDAN = "bermudan"


class FutureType(str, Enum):
    """Types of futures contracts."""
    COMMODITY = "commodity"
    FINANCIAL = "financial"
    CURRENCY = "currency"
    INDEX = "index"


class REITType(str, Enum):
    """Types of Real Estate Investment Trusts."""
    EQUITY = "equity"  # Own and operate income-producing real estate
    MORTGAGE = "mortgage"  # Provide financing for income-producing real estate
    HYBRID = "hybrid"  # Combination of equity and mortgage REITs


class CryptoType(str, Enum):
    """Types of cryptocurrencies."""
    COIN = "coin"  # Native blockchain currency (BTC, ETH)
    TOKEN = "token"  # Built on another blockchain (ERC-20, etc.)
    STABLECOIN = "stablecoin"  # Pegged to fiat currency
    DEFI = "defi"  # Decentralized finance tokens


class CommodityType(str, Enum):
    """Types of commodities."""
    ENERGY = "energy"  # Oil, natural gas, etc.
    METALS = "metals"  # Gold, silver, copper, etc.
    AGRICULTURE = "agriculture"  # Corn, wheat, soybeans, etc.
    LIVESTOCK = "livestock"  # Cattle, hogs, etc.


class AlternativeType(str, Enum):
    """Types of alternative investments."""
    PRIVATE_EQUITY = "private_equity"
    VENTURE_CAPITAL = "venture_capital"
    HEDGE_FUND = "hedge_fund"
    REAL_ASSETS = "real_assets"
    INFRASTRUCTURE = "infrastructure"
    ART_COLLECTIBLES = "art_collectibles"


class StructuredProductType(str, Enum):
    """Types of structured products."""
    PRINCIPAL_PROTECTED = "principal_protected"
    YIELD_ENHANCEMENT = "yield_enhancement"
    PARTICIPATION = "participation"
    LEVERAGED = "leveraged"


class OptionAsset(Asset):
    """
    Represents an options contract.

    Includes Greeks, strike price, expiration, and other option-specific attributes.
    """

    asset_class: AssetClass = Field(AssetClass.ETF, description="Set to ETF for options")
    option_type: OptionType = Field(..., description="Call or Put")
    option_style: OptionStyle = Field(OptionStyle.AMERICAN, description="Exercise style")

    # Contract specifications
    underlying_symbol: str = Field(..., description="Underlying asset symbol")
    strike_price: float = Field(..., description="Strike price")
    expiration_date: date = Field(..., description="Expiration date")
    contract_size: int = Field(100, description="Number of shares per contract")

    # Greeks
    delta: Optional[float] = Field(None, description="Delta: sensitivity to underlying price")
    gamma: Optional[float] = Field(None, description="Gamma: rate of change of delta")
    theta: Optional[float] = Field(None, description="Theta: time decay")
    vega: Optional[float] = Field(None, description="Vega: sensitivity to volatility")
    rho: Optional[float] = Field(None, description="Rho: sensitivity to interest rates")

    # Pricing
    implied_volatility: Optional[float] = Field(None, description="Implied volatility")
    intrinsic_value: Optional[float] = Field(None, description="Intrinsic value")
    time_value: Optional[float] = Field(None, description="Time value")

    # Trading
    bid: Optional[float] = Field(None, description="Bid price")
    ask: Optional[float] = Field(None, description="Ask price")
    open_interest: Optional[int] = Field(None, description="Open interest")
    volume: Optional[int] = Field(None, description="Trading volume")

    @validator("expiration_date")
    def validate_expiration(cls, v):
        """Ensure expiration date is in the future."""
        if isinstance(v, date) and v < date.today():
            raise ValueError("Expiration date must be in the future")
        return v

    def calculate_intrinsic_value(self, underlying_price: float) -> float:
        """
        Calculate intrinsic value of the option.

        Args:
            underlying_price: Current price of underlying asset

        Returns:
            Intrinsic value
        """
        if self.option_type == OptionType.CALL:
            intrinsic = max(0, underlying_price - self.strike_price)
        else:  # PUT
            intrinsic = max(0, self.strike_price - underlying_price)

        self.intrinsic_value = intrinsic
        return intrinsic

    def calculate_time_value(self, market_price: float, underlying_price: float) -> float:
        """
        Calculate time value of the option.

        Args:
            market_price: Current market price of option
            underlying_price: Current price of underlying asset

        Returns:
            Time value
        """
        intrinsic = self.calculate_intrinsic_value(underlying_price)
        time_val = max(0, market_price - intrinsic)
        self.time_value = time_val
        return time_val

    def days_to_expiration(self) -> int:
        """Calculate days until expiration."""
        return (self.expiration_date - date.today()).days

    def is_in_the_money(self, underlying_price: float) -> bool:
        """Check if option is in the money."""
        if self.option_type == OptionType.CALL:
            return underlying_price > self.strike_price
        else:
            return underlying_price < self.strike_price

    def moneyness(self, underlying_price: float) -> str:
        """
        Determine moneyness of option.

        Returns:
            'ITM' (in-the-money), 'ATM' (at-the-money), or 'OTM' (out-of-the-money)
        """
        ratio = underlying_price / self.strike_price

        # Consider ATM if within 2% of strike
        if 0.98 <= ratio <= 1.02:
            return "ATM"
        elif self.is_in_the_money(underlying_price):
            return "ITM"
        else:
            return "OTM"


class REITAsset(Asset):
    """
    Represents a Real Estate Investment Trust.

    Includes property type, dividend yield, FFO, and REIT-specific metrics.
    """

    asset_class: AssetClass = Field(AssetClass.REAL_ESTATE, description="REIT asset class")
    reit_type: REITType = Field(..., description="Type of REIT")

    # Property details
    property_types: List[str] = Field(default_factory=list, description="Types of properties")
    geographic_focus: List[str] = Field(default_factory=list, description="Geographic regions")

    # Financial metrics
    dividend_yield: Optional[float] = Field(None, description="Annual dividend yield")
    ffo_per_share: Optional[float] = Field(None, description="Funds From Operations per share")
    affo_per_share: Optional[float] = Field(None, description="Adjusted FFO per share")

    # REIT-specific ratios
    price_to_ffo: Optional[float] = Field(None, description="Price to FFO ratio")
    debt_to_equity: Optional[float] = Field(None, description="Debt to equity ratio")
    occupancy_rate: Optional[float] = Field(None, description="Portfolio occupancy rate")

    # Income metrics
    net_operating_income: Optional[float] = Field(None, description="Net operating income")
    total_properties: Optional[int] = Field(None, description="Number of properties")
    square_footage: Optional[float] = Field(None, description="Total square footage")

    @validator("dividend_yield", "occupancy_rate")
    def validate_percentage(cls, v):
        """Ensure percentage fields are valid."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Percentage must be between 0 and 1")
        return v

    def calculate_reit_metrics(self) -> Dict[str, float]:
        """Calculate REIT-specific performance metrics."""
        metrics = {}

        if self.ffo_per_share and self.metrics and self.metrics.expected_return:
            # FFO yield
            current_price = self.market_cap / 1e9 if self.market_cap else None
            if current_price:
                metrics['ffo_yield'] = self.ffo_per_share / current_price

        if self.dividend_yield:
            metrics['dividend_yield'] = self.dividend_yield

        if self.occupancy_rate:
            metrics['occupancy_rate'] = self.occupancy_rate

        return metrics


class CryptoAsset(Asset):
    """
    Represents a cryptocurrency asset.

    Includes blockchain-specific attributes, volatility adjustments, and crypto metrics.
    """

    asset_class: AssetClass = Field(AssetClass.CRYPTOCURRENCY, description="Crypto asset class")
    crypto_type: CryptoType = Field(..., description="Type of cryptocurrency")

    # Blockchain details
    blockchain: str = Field(..., description="Underlying blockchain")
    consensus_mechanism: Optional[str] = Field(None, description="Proof of Work, Proof of Stake, etc.")

    # Supply metrics
    circulating_supply: Optional[float] = Field(None, description="Circulating supply")
    total_supply: Optional[float] = Field(None, description="Total supply")
    max_supply: Optional[float] = Field(None, description="Maximum supply (if capped)")

    # Crypto-specific metrics
    market_dominance: Optional[float] = Field(None, description="Market dominance percentage")
    trading_volume_24h: Optional[float] = Field(None, description="24-hour trading volume")
    volatility_30d: Optional[float] = Field(None, description="30-day volatility")

    # DeFi metrics (if applicable)
    total_value_locked: Optional[float] = Field(None, description="Total Value Locked (for DeFi)")
    staking_apy: Optional[float] = Field(None, description="Staking APY")

    # Technical indicators
    hash_rate: Optional[float] = Field(None, description="Network hash rate (for PoW)")
    active_addresses: Optional[int] = Field(None, description="Number of active addresses")

    # Volatility adjustment factor
    volatility_multiplier: float = Field(1.5, description="Volatility adjustment multiplier")

    def calculate_adjusted_volatility(self) -> float:
        """
        Calculate volatility with crypto-specific adjustment.

        Cryptocurrencies typically have higher volatility than traditional assets.

        Returns:
            Adjusted volatility
        """
        if not self.metrics or not self.metrics.volatility:
            raise ValueError("Base volatility metrics required")

        # Apply crypto volatility multiplier
        adjusted_vol = self.metrics.volatility * self.volatility_multiplier

        # Additional adjustment for smaller market cap coins
        if self.market_cap and self.market_cap < 1e9:  # < $1B market cap
            adjusted_vol *= 1.2  # 20% additional volatility for smaller coins

        return adjusted_vol

    def calculate_risk_score(self) -> float:
        """
        Calculate risk score for cryptocurrency (0-100).

        Returns:
            Risk score where 100 is highest risk
        """
        risk_score = 50  # Base score

        # Adjust for market cap
        if self.market_cap:
            if self.market_cap > 100e9:  # > $100B
                risk_score -= 20
            elif self.market_cap < 1e9:  # < $1B
                risk_score += 20

        # Adjust for volatility
        if self.volatility_30d:
            if self.volatility_30d > 0.5:  # > 50% volatility
                risk_score += 15
            elif self.volatility_30d < 0.2:  # < 20% volatility
                risk_score -= 10

        # Adjust for type
        if self.crypto_type == CryptoType.STABLECOIN:
            risk_score -= 30
        elif self.crypto_type == CryptoType.COIN:
            risk_score -= 10

        return max(0, min(100, risk_score))


class CommodityAsset(Asset):
    """
    Represents a commodity or commodity futures contract.
    """

    asset_class: AssetClass = Field(AssetClass.COMMODITY, description="Commodity asset class")
    commodity_type: CommodityType = Field(..., description="Type of commodity")

    # Contract specifications (for futures)
    contract_month: Optional[str] = Field(None, description="Delivery month (e.g., 'Dec2024')")
    contract_size: Optional[float] = Field(None, description="Size of one contract")
    unit: Optional[str] = Field(None, description="Unit of measurement (barrels, ounces, etc.)")

    # Trading details
    tick_size: Optional[float] = Field(None, description="Minimum price movement")
    tick_value: Optional[float] = Field(None, description="Dollar value of one tick")

    # Commodity-specific metrics
    spot_price: Optional[float] = Field(None, description="Current spot price")
    futures_price: Optional[float] = Field(None, description="Futures contract price")
    storage_costs: Optional[float] = Field(None, description="Annual storage costs")
    convenience_yield: Optional[float] = Field(None, description="Convenience yield")

    # Market structure
    contango_backwardation: Optional[str] = Field(None, description="'contango' or 'backwardation'")
    roll_yield: Optional[float] = Field(None, description="Expected roll yield")

    def calculate_carry_cost(self, risk_free_rate: float) -> float:
        """
        Calculate cost of carry for commodity.

        Args:
            risk_free_rate: Risk-free interest rate

        Returns:
            Annualized cost of carry
        """
        storage = self.storage_costs if self.storage_costs else 0
        convenience = self.convenience_yield if self.convenience_yield else 0

        carry_cost = risk_free_rate + storage - convenience
        return carry_cost

    def is_contango(self) -> Optional[bool]:
        """
        Determine if market is in contango.

        Returns:
            True if contango, False if backwardation, None if unknown
        """
        if self.spot_price and self.futures_price:
            return self.futures_price > self.spot_price
        return None


class FutureContract(Asset):
    """
    Represents a futures contract.
    """

    asset_class: AssetClass = Field(AssetClass.COMMODITY, description="Futures")
    future_type: FutureType = Field(..., description="Type of future")

    # Contract specifications
    underlying_symbol: str = Field(..., description="Underlying asset")
    contract_month: str = Field(..., description="Delivery/expiration month")
    contract_size: float = Field(..., description="Size of contract")

    # Pricing
    initial_margin: Optional[float] = Field(None, description="Initial margin requirement")
    maintenance_margin: Optional[float] = Field(None, description="Maintenance margin")

    # Trading
    settlement_type: str = Field("cash", description="'cash' or 'physical' settlement")
    last_trading_day: Optional[date] = Field(None, description="Last trading day")

    # Greeks (for financial futures)
    duration: Optional[float] = Field(None, description="Duration for bond futures")
    basis: Optional[float] = Field(None, description="Basis (futures - spot)")

    def calculate_leverage(self, contract_value: float) -> float:
        """
        Calculate leverage ratio.

        Args:
            contract_value: Current value of contract

        Returns:
            Leverage ratio
        """
        if not self.initial_margin:
            return 1.0

        return contract_value / self.initial_margin


class AlternativeAsset(Asset):
    """
    Represents alternative investments like private equity, hedge funds, etc.
    """

    asset_class: AssetClass = Field(AssetClass.ETF, description="Alternative investment")
    alternative_type: AlternativeType = Field(..., description="Type of alternative investment")

    # Investment details
    vintage_year: Optional[int] = Field(None, description="Year of investment/fund launch")
    investment_period: Optional[int] = Field(None, description="Investment period in years")
    lock_up_period: Optional[int] = Field(None, description="Lock-up period in years")

    # Fee structure
    management_fee: Optional[float] = Field(None, description="Annual management fee")
    performance_fee: Optional[float] = Field(None, description="Performance fee (carried interest)")
    hurdle_rate: Optional[float] = Field(None, description="Hurdle rate for performance fees")

    # Performance metrics
    irr: Optional[float] = Field(None, description="Internal Rate of Return")
    moic: Optional[float] = Field(None, description="Multiple on Invested Capital")
    tvpi: Optional[float] = Field(None, description="Total Value to Paid-In")
    dpi: Optional[float] = Field(None, description="Distributions to Paid-In")
    rvpi: Optional[float] = Field(None, description="Residual Value to Paid-In")

    # Liquidity
    liquidity_tier: Optional[str] = Field(None, description="Daily, Monthly, Quarterly, Annual, Illiquid")
    redemption_notice: Optional[int] = Field(None, description="Redemption notice period in days")

    # Strategy
    strategy: Optional[str] = Field(None, description="Investment strategy")
    geographic_focus: Optional[List[str]] = Field(None, description="Geographic regions")
    sector_focus: Optional[List[str]] = Field(None, description="Sector focus")

    @validator("management_fee", "performance_fee")
    def validate_fees(cls, v):
        """Ensure fees are reasonable percentages."""
        if v is not None and (v < 0 or v > 0.5):  # Max 50% fee
            raise ValueError("Fee must be between 0 and 0.5")
        return v

    def calculate_net_return(self, gross_return: float) -> float:
        """
        Calculate net return after fees.

        Args:
            gross_return: Gross return before fees

        Returns:
            Net return after management and performance fees
        """
        # Deduct management fee
        mgmt_fee = self.management_fee if self.management_fee else 0
        net = gross_return - mgmt_fee

        # Apply performance fee if above hurdle
        if self.performance_fee:
            hurdle = self.hurdle_rate if self.hurdle_rate else 0
            if net > hurdle:
                excess = net - hurdle
                perf_fee = excess * self.performance_fee
                net -= perf_fee

        return net


class StructuredProduct(Asset):
    """
    Represents structured products combining derivatives and traditional securities.
    """

    asset_class: AssetClass = Field(AssetClass.ETF, description="Structured product")
    product_type: StructuredProductType = Field(..., description="Type of structured product")

    # Structure components
    underlying_assets: List[str] = Field(default_factory=list, description="Underlying assets")

    # Protection/participation
    protection_level: Optional[float] = Field(None, description="Capital protection level (0-1)")
    participation_rate: Optional[float] = Field(None, description="Participation in upside")
    cap_level: Optional[float] = Field(None, description="Maximum return cap")
    barrier_level: Optional[float] = Field(None, description="Barrier/knock-in level")

    # Terms
    maturity_date: date = Field(..., description="Maturity date")
    observation_frequency: Optional[str] = Field(None, description="How often performance is observed")

    # Pricing
    issue_price: float = Field(100.0, description="Issue price (typically 100)")
    embedded_option_value: Optional[float] = Field(None, description="Value of embedded options")

    # Issuer
    issuer: str = Field(..., description="Issuing institution")
    issuer_credit_rating: Optional[str] = Field(None, description="Credit rating of issuer")

    @validator("protection_level", "participation_rate")
    def validate_rates(cls, v):
        """Ensure rates are valid."""
        if v is not None and (v < 0 or v > 5):  # Allow participation > 100%
            raise ValueError("Rate must be between 0 and 5")
        return v

    def calculate_payoff(
        self,
        underlying_return: float,
        at_maturity: bool = True
    ) -> float:
        """
        Calculate payoff of structured product.

        Args:
            underlying_return: Return of underlying asset(s)
            at_maturity: Whether calculation is at maturity

        Returns:
            Payoff value
        """
        # Apply participation rate
        participation = self.participation_rate if self.participation_rate else 1.0
        payoff = underlying_return * participation

        # Apply cap if exists
        if self.cap_level:
            payoff = min(payoff, self.cap_level)

        # Apply protection level
        if self.protection_level and at_maturity:
            payoff = max(payoff, -self.protection_level)

        # Check barrier
        if self.barrier_level and underlying_return < self.barrier_level:
            # Barrier breached - protection may be lost
            if at_maturity:
                payoff = underlying_return  # Full downside exposure

        return self.issue_price * (1 + payoff)
