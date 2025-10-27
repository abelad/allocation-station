"""Advanced portfolio features including tax-loss harvesting, multi-currency, transitions, factors, ESG, leverage, and insurance."""

from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, date
from enum import Enum
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd
from dataclasses import dataclass


# ============================================================================
# Tax-Loss Harvesting
# ============================================================================

class TaxLot(BaseModel):
    """Represents a tax lot for an asset position."""

    asset_symbol: str
    purchase_date: date
    quantity: float
    cost_basis: float  # Cost per share/unit
    current_price: Optional[float] = None

    def calculate_gain_loss(self, current_price: Optional[float] = None) -> float:
        """Calculate unrealized gain/loss for this lot."""
        price = current_price or self.current_price
        if price is None:
            raise ValueError("Current price required")

        return (price - self.cost_basis) * self.quantity

    def is_short_term(self, as_of_date: Optional[date] = None) -> bool:
        """Check if position is short-term (held < 1 year)."""
        ref_date = as_of_date or date.today()
        holding_period = (ref_date - self.purchase_date).days
        return holding_period < 365

    def get_holding_period_days(self, as_of_date: Optional[date] = None) -> int:
        """Get holding period in days."""
        ref_date = as_of_date or date.today()
        return (ref_date - self.purchase_date).days


class TaxLossHarvestingStrategy:
    """
    Tax-loss harvesting strategy to realize losses for tax benefits.

    Identifies opportunities to sell positions at a loss and replace
    with similar (but not substantially identical) assets.
    """

    def __init__(
        self,
        short_term_rate: float = 0.37,  # Ordinary income tax rate
        long_term_rate: float = 0.20,   # Long-term capital gains rate
        wash_sale_days: int = 30,       # Wash sale rule period
        min_loss_threshold: float = 100.0  # Minimum loss to harvest
    ):
        """
        Initialize tax-loss harvesting strategy.

        Args:
            short_term_rate: Tax rate for short-term gains
            long_term_rate: Tax rate for long-term gains
            wash_sale_days: Days to avoid wash sale
            min_loss_threshold: Minimum loss to consider harvesting
        """
        self.short_term_rate = short_term_rate
        self.long_term_rate = long_term_rate
        self.wash_sale_days = wash_sale_days
        self.min_loss_threshold = min_loss_threshold

    def identify_harvest_opportunities(
        self,
        tax_lots: List[TaxLot],
        recent_trades: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify tax-loss harvesting opportunities.

        Args:
            tax_lots: List of tax lots
            recent_trades: Recent trades to check wash sale

        Returns:
            List of harvest opportunities with details
        """
        opportunities = []

        for lot in tax_lots:
            if lot.current_price is None:
                continue

            # Calculate loss
            loss = lot.calculate_gain_loss()

            if loss < -self.min_loss_threshold:
                # Check wash sale rule
                if not self._violates_wash_sale(lot, recent_trades):
                    is_st = lot.is_short_term()
                    tax_rate = self.short_term_rate if is_st else self.long_term_rate
                    tax_benefit = abs(loss) * tax_rate

                    opportunities.append({
                        'asset_symbol': lot.asset_symbol,
                        'lot_purchase_date': lot.purchase_date,
                        'quantity': lot.quantity,
                        'cost_basis': lot.cost_basis,
                        'current_price': lot.current_price,
                        'unrealized_loss': loss,
                        'is_short_term': is_st,
                        'tax_rate': tax_rate,
                        'tax_benefit': tax_benefit,
                        'holding_period_days': lot.get_holding_period_days()
                    })

        # Sort by tax benefit (descending)
        opportunities.sort(key=lambda x: x['tax_benefit'], reverse=True)

        return opportunities

    def _violates_wash_sale(
        self,
        lot: TaxLot,
        recent_trades: Optional[List[Dict]]
    ) -> bool:
        """Check if selling this lot would violate wash sale rule."""
        if not recent_trades:
            return False

        cutoff_date = date.today() - timedelta(days=self.wash_sale_days)

        for trade in recent_trades:
            if (trade['symbol'] == lot.asset_symbol and
                trade['action'] == 'buy' and
                trade['date'] >= cutoff_date):
                return True

        return False

    def suggest_replacement_assets(
        self,
        sold_asset: str,
        available_assets: List[str]
    ) -> List[str]:
        """
        Suggest replacement assets that won't trigger wash sale.

        In practice, this would use correlation/similarity analysis.

        Args:
            sold_asset: Asset being sold
            available_assets: Potential replacements

        Returns:
            List of suitable replacement assets
        """
        # Simple placeholder - in reality would analyze correlations
        replacements = [asset for asset in available_assets if asset != sold_asset]
        return replacements[:5]  # Return top 5


# ============================================================================
# Multi-Currency Support
# ============================================================================

class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    CNY = "CNY"


class CurrencyExposure(BaseModel):
    """Currency exposure for a position."""

    base_currency: Currency
    position_currency: Currency
    exposure_amount: float
    hedge_ratio: float = Field(0.0, ge=0.0, le=1.0)  # 0 = unhedged, 1 = fully hedged


class MultiCurrencyPortfolio:
    """
    Portfolio with multi-currency support and FX risk management.
    """

    def __init__(
        self,
        base_currency: Currency = Currency.USD,
        fx_rates: Optional[Dict[Tuple[Currency, Currency], float]] = None
    ):
        """
        Initialize multi-currency portfolio.

        Args:
            base_currency: Base currency for reporting
            fx_rates: Foreign exchange rates
        """
        self.base_currency = base_currency
        self.fx_rates = fx_rates or {}
        self.currency_exposures: List[CurrencyExposure] = []

    def add_currency_exposure(self, exposure: CurrencyExposure):
        """Add currency exposure to track."""
        self.currency_exposures.append(exposure)

    def convert_to_base(
        self,
        amount: float,
        from_currency: Currency
    ) -> float:
        """
        Convert amount to base currency.

        Args:
            amount: Amount in foreign currency
            from_currency: Currency to convert from

        Returns:
            Amount in base currency
        """
        if from_currency == self.base_currency:
            return amount

        rate_key = (from_currency, self.base_currency)
        if rate_key in self.fx_rates:
            return amount * self.fx_rates[rate_key]

        # Try inverse rate
        inverse_key = (self.base_currency, from_currency)
        if inverse_key in self.fx_rates:
            return amount / self.fx_rates[inverse_key]

        raise ValueError(f"No exchange rate available for {from_currency} to {self.base_currency}")

    def calculate_fx_exposure(self) -> Dict[Currency, float]:
        """
        Calculate total FX exposure by currency.

        Returns:
            Dictionary of currency to exposure amount (in base currency)
        """
        exposure_by_currency = {}

        for exp in self.currency_exposures:
            if exp.position_currency not in exposure_by_currency:
                exposure_by_currency[exp.position_currency] = 0.0

            # Calculate unhedged exposure
            unhedged_ratio = 1.0 - exp.hedge_ratio
            base_exposure = self.convert_to_base(
                exp.exposure_amount * unhedged_ratio,
                exp.position_currency
            )

            exposure_by_currency[exp.position_currency] += base_exposure

        return exposure_by_currency

    def calculate_fx_var(
        self,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1
    ) -> float:
        """
        Calculate Value at Risk from FX exposure.

        Args:
            confidence_level: Confidence level for VaR
            time_horizon_days: Time horizon in days

        Returns:
            VaR in base currency
        """
        # Simplified VaR calculation
        # In practice, would use historical volatilities and correlations

        exposures = self.calculate_fx_exposure()

        # Assume typical FX volatility of 10% annually
        fx_vol = 0.10
        daily_vol = fx_vol / np.sqrt(252)
        horizon_vol = daily_vol * np.sqrt(time_horizon_days)

        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence_level)

        # Total exposure
        total_exposure = sum(exposures.values())

        # VaR
        var = abs(z_score) * horizon_vol * total_exposure

        return var


# ============================================================================
# Portfolio Transition Analysis
# ============================================================================

class TransitionCost(BaseModel):
    """Costs associated with portfolio transition."""

    commission_rate: float = Field(0.001, description="Commission as % of trade value")
    bid_ask_spread: float = Field(0.002, description="Bid-ask spread as % of price")
    market_impact: float = Field(0.001, description="Market impact as % of trade value")
    tax_cost: float = Field(0.0, description="Capital gains tax cost")


class PortfolioTransition:
    """
    Analyzes and optimizes portfolio transitions from current to target allocation.
    """

    def __init__(self, transition_costs: Optional[TransitionCost] = None):
        """Initialize portfolio transition analyzer."""
        self.costs = transition_costs or TransitionCost()

    def calculate_transition_plan(
        self,
        current_holdings: Dict[str, float],
        target_holdings: Dict[str, float],
        current_prices: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal transition plan.

        Args:
            current_holdings: Current positions (symbol -> quantity)
            target_holdings: Target positions (symbol -> quantity)
            current_prices: Current prices (symbol -> price)
            constraints: Trading constraints

        Returns:
            Transition plan with trades and costs
        """
        trades = {}
        total_trade_value = 0

        # Calculate required trades
        all_symbols = set(list(current_holdings.keys()) + list(target_holdings.keys()))

        for symbol in all_symbols:
            current_qty = current_holdings.get(symbol, 0)
            target_qty = target_holdings.get(symbol, 0)
            trade_qty = target_qty - current_qty

            if abs(trade_qty) > 0.0001:  # Minimum trade threshold
                price = current_prices.get(symbol, 0)
                trade_value = abs(trade_qty * price)
                total_trade_value += trade_value

                trades[symbol] = {
                    'quantity': trade_qty,
                    'direction': 'buy' if trade_qty > 0 else 'sell',
                    'value': trade_value,
                    'price': price
                }

        # Calculate transition costs
        total_costs = self._calculate_total_costs(total_trade_value)

        return {
            'trades': trades,
            'total_trade_value': total_trade_value,
            'costs': total_costs,
            'net_transition_cost': sum(total_costs.values()),
            'cost_as_pct': sum(total_costs.values()) / total_trade_value if total_trade_value > 0 else 0
        }

    def _calculate_total_costs(self, trade_value: float) -> Dict[str, float]:
        """Calculate all transition costs."""
        return {
            'commissions': trade_value * self.costs.commission_rate,
            'bid_ask_spread': trade_value * self.costs.bid_ask_spread,
            'market_impact': trade_value * self.costs.market_impact,
            'tax_cost': self.costs.tax_cost
        }

    def optimize_transition_schedule(
        self,
        trades: Dict[str, Dict],
        n_days: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Spread trades over multiple days to minimize market impact.

        Args:
            trades: Trades to execute
            n_days: Number of days to spread over

        Returns:
            Daily trade schedule
        """
        schedule = []

        for day in range(n_days):
            daily_trades = {}

            for symbol, trade_info in trades.items():
                # Execute proportionally each day
                daily_qty = trade_info['quantity'] / n_days

                daily_trades[symbol] = {
                    'quantity': daily_qty,
                    'direction': trade_info['direction'],
                    'price': trade_info['price']
                }

            schedule.append({
                'day': day + 1,
                'trades': daily_trades
            })

        return schedule


# ============================================================================
# Factor-Based Allocation
# ============================================================================

class Factor(str, Enum):
    """Investment factors."""
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    SIZE = "size"
    LOW_VOLATILITY = "low_volatility"
    DIVIDEND_YIELD = "dividend_yield"


class FactorExposure(BaseModel):
    """Factor exposure for an asset."""

    asset_symbol: str
    factor_loadings: Dict[Factor, float]  # Factor -> loading

    def get_loading(self, factor: Factor) -> float:
        """Get loading for a specific factor."""
        return self.factor_loadings.get(factor, 0.0)


class FactorAllocationStrategy:
    """
    Factor-based portfolio allocation strategy.

    Constructs portfolios targeting specific factor exposures.
    """

    def __init__(
        self,
        target_factors: Dict[Factor, float],
        factor_exposures: Dict[str, FactorExposure]
    ):
        """
        Initialize factor allocation strategy.

        Args:
            target_factors: Target factor exposures
            factor_exposures: Factor exposures by asset
        """
        self.target_factors = target_factors
        self.factor_exposures = factor_exposures

    def calculate_portfolio_factors(
        self,
        weights: Dict[str, float]
    ) -> Dict[Factor, float]:
        """
        Calculate portfolio-level factor exposures.

        Args:
            weights: Portfolio weights by symbol

        Returns:
            Portfolio factor exposures
        """
        portfolio_factors = {factor: 0.0 for factor in Factor}

        for symbol, weight in weights.items():
            if symbol in self.factor_exposures:
                exposure = self.factor_exposures[symbol]

                for factor in Factor:
                    loading = exposure.get_loading(factor)
                    portfolio_factors[factor] += weight * loading

        return portfolio_factors

    def optimize_for_factors(
        self,
        available_assets: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio to match target factor exposures.

        Args:
            available_assets: Assets available for portfolio
            constraints: Additional constraints

        Returns:
            Optimal weights
        """
        # Simplified optimization - in practice would use quadratic programming
        # This is a placeholder showing the structure

        n_assets = len(available_assets)
        equal_weight = 1.0 / n_assets

        # Start with equal weight
        weights = {asset: equal_weight for asset in available_assets}

        return weights


# ============================================================================
# Leveraged Portfolios
# ============================================================================

class LeverageType(str, Enum):
    """Types of leverage."""
    MARGIN = "margin"
    FUTURES = "futures"
    OPTIONS = "options"
    SWAPS = "swaps"


class LeveragedPosition(BaseModel):
    """Represents a leveraged position."""

    asset_symbol: str
    position_value: float
    leverage_ratio: float
    leverage_type: LeverageType
    margin_requirement: float
    interest_rate: Optional[float] = None  # For margin leverage

    def calculate_exposure(self) -> float:
        """Calculate total exposure including leverage."""
        return self.position_value * self.leverage_ratio

    def calculate_margin_call_level(self, maintenance_margin: float = 0.25) -> float:
        """Calculate price level that triggers margin call."""
        exposure = self.calculate_exposure()
        margin_call_value = self.margin_requirement / maintenance_margin
        return margin_call_value / self.leverage_ratio


class LeveragedPortfolio:
    """
    Portfolio with leverage support and risk management.
    """

    def __init__(
        self,
        max_leverage: float = 2.0,
        margin_call_threshold: float = 0.25
    ):
        """
        Initialize leveraged portfolio.

        Args:
            max_leverage: Maximum allowed leverage
            margin_call_threshold: Maintenance margin threshold
        """
        self.max_leverage = max_leverage
        self.margin_call_threshold = margin_call_threshold
        self.leveraged_positions: List[LeveragedPosition] = []

    def add_leveraged_position(self, position: LeveragedPosition):
        """Add a leveraged position."""
        if position.leverage_ratio > self.max_leverage:
            raise ValueError(f"Leverage {position.leverage_ratio} exceeds maximum {self.max_leverage}")

        self.leveraged_positions.append(position)

    def calculate_total_leverage(self) -> float:
        """Calculate portfolio-level leverage ratio."""
        total_exposure = sum(pos.calculate_exposure() for pos in self.leveraged_positions)
        total_equity = sum(pos.position_value for pos in self.leveraged_positions)

        return total_exposure / total_equity if total_equity > 0 else 0

    def calculate_margin_requirements(self) -> float:
        """Calculate total margin requirements."""
        return sum(pos.margin_requirement for pos in self.leveraged_positions)

    def calculate_leverage_costs(self) -> float:
        """Calculate total leverage costs (interest, fees)."""
        total_cost = 0

        for pos in self.leveraged_positions:
            if pos.interest_rate and pos.leverage_type == LeverageType.MARGIN:
                borrowed_amount = pos.calculate_exposure() - pos.position_value
                annual_cost = borrowed_amount * pos.interest_rate
                total_cost += annual_cost

        return total_cost


# ============================================================================
# Portfolio Insurance (CPPI)
# ============================================================================

class CPPIStrategy:
    """
    Constant Proportion Portfolio Insurance (CPPI) strategy.

    Dynamically allocates between risky and safe assets to protect capital.
    """

    def __init__(
        self,
        floor_value: float,
        multiplier: float = 3.0,
        risky_asset_return: float = 0.08,
        safe_asset_return: float = 0.02
    ):
        """
        Initialize CPPI strategy.

        Args:
            floor_value: Minimum portfolio value to maintain
            multiplier: CPPI multiplier (typically 2-5)
            risky_asset_return: Expected return of risky asset
            safe_asset_return: Return of safe asset
        """
        self.floor_value = floor_value
        self.multiplier = multiplier
        self.risky_asset_return = risky_asset_return
        self.safe_asset_return = safe_asset_return

    def calculate_allocation(
        self,
        current_value: float
    ) -> Tuple[float, float]:
        """
        Calculate CPPI allocation.

        Args:
            current_value: Current portfolio value

        Returns:
            Tuple of (risky_allocation, safe_allocation) in dollars
        """
        # Calculate cushion
        cushion = max(0, current_value - self.floor_value)

        # Exposure to risky asset
        risky_exposure = self.multiplier * cushion

        # Cap at current value (cannot exceed 100%)
        risky_allocation = min(risky_exposure, current_value)
        safe_allocation = current_value - risky_allocation

        return risky_allocation, safe_allocation

    def calculate_allocation_pct(
        self,
        current_value: float
    ) -> Tuple[float, float]:
        """
        Calculate CPPI allocation as percentages.

        Args:
            current_value: Current portfolio value

        Returns:
            Tuple of (risky_pct, safe_pct)
        """
        risky_amt, safe_amt = self.calculate_allocation(current_value)

        if current_value > 0:
            return risky_amt / current_value, safe_amt / current_value

        return 0.0, 1.0

    def simulate_path(
        self,
        initial_value: float,
        n_periods: int = 252,
        risky_vol: float = 0.20
    ) -> pd.DataFrame:
        """
        Simulate CPPI strategy path.

        Args:
            initial_value: Initial portfolio value
            n_periods: Number of periods to simulate
            risky_vol: Volatility of risky asset

        Returns:
            DataFrame with simulation results
        """
        results = []
        value = initial_value

        for period in range(n_periods):
            # Calculate allocation
            risky_amt, safe_amt = self.calculate_allocation(value)
            risky_pct, safe_pct = self.calculate_allocation_pct(value)

            # Simulate returns
            risky_return = np.random.normal(
                self.risky_asset_return / 252,
                risky_vol / np.sqrt(252)
            )
            safe_return = self.safe_asset_return / 252

            # Update value
            value = risky_amt * (1 + risky_return) + safe_amt * (1 + safe_return)

            results.append({
                'period': period,
                'value': value,
                'risky_allocation': risky_amt,
                'safe_allocation': safe_amt,
                'risky_pct': risky_pct,
                'safe_pct': safe_pct,
                'cushion': max(0, value - self.floor_value)
            })

        return pd.DataFrame(results)
