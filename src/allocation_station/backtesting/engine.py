"""Backtesting engine for portfolio strategies."""

from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from tqdm import tqdm
import warnings
from ..core import Portfolio, Asset, AssetClass
from ..portfolio import AllocationStrategy, WithdrawalStrategy, RebalancingStrategy
from ..data import MarketDataProvider


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""

    # Time period
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")

    # Initial conditions
    initial_capital: float = Field(100000, description="Initial portfolio value")
    initial_allocation: Optional[Dict[str, float]] = Field(None, description="Initial weights")

    # Costs and constraints
    transaction_cost: float = Field(0.001, description="Transaction cost (percentage)")
    slippage: float = Field(0.0005, description="Slippage (percentage)")
    tax_rate: float = Field(0.15, description="Capital gains tax rate")
    min_trade_size: float = Field(100, description="Minimum trade size in dollars")

    # Rebalancing
    rebalance_frequency: str = Field("monthly", description="Rebalancing frequency")
    rebalance_threshold: float = Field(0.05, description="Drift threshold for rebalancing")

    # Benchmark
    benchmark_symbol: Optional[str] = Field("SPY", description="Benchmark symbol for comparison")

    # Settings
    verbose: bool = Field(True, description="Show progress")
    save_history: bool = Field(True, description="Save detailed history")

    @validator("end_date")
    def validate_dates(cls, v, values):
        """Ensure end date is after start date."""
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("End date must be after start date")
        return v


@dataclass
class BacktestResults:
    """Results from backtesting."""

    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float

    # Risk metrics
    var_95: float
    cvar_95: float
    beta: float
    alpha: float

    # Trading metrics
    total_trades: int
    turnover: float
    transaction_costs: float
    taxes_paid: float

    # Comparison metrics
    benchmark_return: Optional[float]
    tracking_error: Optional[float]
    information_ratio: Optional[float]

    # Time series
    portfolio_values: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    transactions: pd.DataFrame

    # Analysis
    monthly_returns: pd.Series
    annual_returns: pd.Series
    rolling_sharpe: pd.Series

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'beta': self.beta,
            'alpha': self.alpha,
            'total_trades': self.total_trades,
            'turnover': self.turnover,
            'transaction_costs': self.transaction_costs,
            'benchmark_return': self.benchmark_return,
            'tracking_error': self.tracking_error,
            'information_ratio': self.information_ratio,
        }

    def summary(self) -> str:
        """Generate summary string of results."""
        summary = f"""
Backtest Results Summary
========================
Total Return: {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Volatility: {self.volatility:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Sortino Ratio: {self.sortino_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}

Risk Metrics:
VaR (95%): {self.var_95:.2%}
CVaR (95%): {self.cvar_95:.2%}

Trading Metrics:
Total Trades: {self.total_trades}
Turnover: {self.turnover:.2%}
Transaction Costs: ${self.transaction_costs:,.2f}
"""
        if self.benchmark_return:
            summary += f"""
Benchmark Comparison:
Benchmark Return: {self.benchmark_return:.2%}
Tracking Error: {self.tracking_error:.2%}
Information Ratio: {self.information_ratio:.2f}
"""
        return summary


class BacktestEngine:
    """
    Engine for backtesting portfolio strategies.

    This class handles historical strategy backtesting with realistic
    market conditions, transaction costs, and performance tracking.
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_provider: Optional[MarketDataProvider] = None
    ):
        """Initialize backtest engine."""
        self.config = config
        self.data_provider = data_provider or MarketDataProvider()

        # State tracking
        self._portfolio_history = []
        self._transaction_history = []
        self._weights_history = []

    def backtest(
        self,
        portfolio: Portfolio,
        strategy: AllocationStrategy,
        withdrawal_strategy: Optional[WithdrawalStrategy] = None
    ) -> BacktestResults:
        """
        Run backtest for a portfolio strategy.

        Args:
            portfolio: Portfolio to backtest
            strategy: Allocation strategy to test
            withdrawal_strategy: Optional withdrawal strategy

        Returns:
            BacktestResults with performance metrics
        """
        # Fetch historical data
        symbols = list(portfolio.assets.keys())
        market_data = self._fetch_market_data(symbols)

        if market_data.empty:
            raise ValueError("No market data available for backtesting")

        # Initialize portfolio tracking
        portfolio_values = []
        weights_history = []
        transactions = []

        # Get trading dates
        dates = market_data.index.unique().sort_values()

        # Initialize portfolio
        current_weights = strategy.target_allocation.copy()
        current_value = self.config.initial_capital
        shares = self._calculate_initial_shares(current_weights, current_value, market_data.iloc[0])

        # Progress bar
        iterator = dates
        if self.config.verbose:
            iterator = tqdm(dates, desc="Backtesting")

        # Run backtest
        for date in iterator:
            # Get current prices
            current_prices = self._get_prices_for_date(market_data, date, symbols)

            # Calculate current portfolio value
            current_value = self._calculate_portfolio_value(shares, current_prices)

            # Apply withdrawal if configured
            if withdrawal_strategy:
                withdrawal = self._calculate_withdrawal(
                    current_value,
                    date,
                    withdrawal_strategy
                )
                current_value -= withdrawal

            # Check if rebalancing needed
            if self._should_rebalance(date):
                # Calculate new allocation
                new_weights = strategy.calculate_allocation(portfolio, market_data, date)

                # Execute rebalancing trades
                trades, costs = self._execute_rebalance(
                    shares,
                    current_weights,
                    new_weights,
                    current_value,
                    current_prices
                )

                # Update shares and record transactions
                shares = self._update_shares(shares, trades)
                current_value -= costs
                transactions.extend(trades)

                # Update current weights
                current_weights = new_weights

            # Record portfolio state
            portfolio_values.append({
                'date': date,
                'value': current_value,
                'weights': current_weights.copy()
            })

        # Create DataFrames from history
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)

        # Calculate performance metrics
        results = self._calculate_performance_metrics(
            portfolio_df,
            transactions,
            market_data
        )

        return results

    def _fetch_market_data(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch historical market data."""
        data = self.data_provider.fetch_historical_data(
            symbols=symbols,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            frequency='daily'
        )

        if self.config.benchmark_symbol:
            benchmark_data = self.data_provider.fetch_historical_data(
                symbols=self.config.benchmark_symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                frequency='daily'
            )
            data = pd.concat([data, benchmark_data])

        return data

    def _get_prices_for_date(
        self,
        market_data: pd.DataFrame,
        date: pd.Timestamp,
        symbols: List[str]
    ) -> Dict[str, float]:
        """Get prices for all symbols on a specific date."""
        prices = {}
        date_data = market_data.loc[date]

        for symbol in symbols:
            if symbol in date_data.index:
                prices[symbol] = date_data.loc[symbol, 'close']
            else:
                # Use last known price if not available
                symbol_data = market_data[market_data['symbol'] == symbol]
                if not symbol_data.empty:
                    last_price = symbol_data[symbol_data.index < date]['close'].iloc[-1]
                    prices[symbol] = last_price

        return prices

    def _calculate_initial_shares(
        self,
        weights: Dict[str, float],
        capital: float,
        initial_prices: pd.Series
    ) -> Dict[str, float]:
        """Calculate initial share quantities."""
        shares = {}

        for symbol, weight in weights.items():
            if symbol in initial_prices.index:
                price = initial_prices.loc[symbol, 'close']
                value = capital * weight
                shares[symbol] = value / price if price > 0 else 0

        return shares

    def _calculate_portfolio_value(
        self,
        shares: Dict[str, float],
        prices: Dict[str, float]
    ) -> float:
        """Calculate current portfolio value."""
        total_value = 0

        for symbol, quantity in shares.items():
            if symbol in prices:
                total_value += quantity * prices[symbol]

        return total_value

    def _should_rebalance(self, date: pd.Timestamp) -> bool:
        """Check if portfolio should be rebalanced."""
        # Implement rebalancing logic based on frequency
        if self.config.rebalance_frequency == 'daily':
            return True
        elif self.config.rebalance_frequency == 'weekly':
            return date.weekday() == 0  # Monday
        elif self.config.rebalance_frequency == 'monthly':
            return date.day == 1 or date.is_month_start
        elif self.config.rebalance_frequency == 'quarterly':
            return date.is_quarter_start
        elif self.config.rebalance_frequency == 'annual':
            return date.is_year_start

        return False

    def _execute_rebalance(
        self,
        current_shares: Dict[str, float],
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float]
    ) -> Tuple[List[Dict], float]:
        """Execute rebalancing trades."""
        trades = []
        total_cost = 0

        for symbol in target_weights.keys():
            if symbol not in prices:
                continue

            # Calculate target shares
            target_value = portfolio_value * target_weights[symbol]
            target_shares = target_value / prices[symbol]

            # Calculate trade
            current_quantity = current_shares.get(symbol, 0)
            trade_shares = target_shares - current_quantity
            trade_value = abs(trade_shares * prices[symbol])

            # Check minimum trade size
            if trade_value >= self.config.min_trade_size:
                # Calculate costs
                transaction_cost = trade_value * self.config.transaction_cost
                slippage_cost = trade_value * self.config.slippage
                total_trade_cost = transaction_cost + slippage_cost

                trades.append({
                    'symbol': symbol,
                    'shares': trade_shares,
                    'price': prices[symbol],
                    'value': trade_value,
                    'cost': total_trade_cost,
                    'type': 'buy' if trade_shares > 0 else 'sell'
                })

                total_cost += total_trade_cost

        return trades, total_cost

    def _update_shares(
        self,
        current_shares: Dict[str, float],
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Update share quantities after trades."""
        new_shares = current_shares.copy()

        for trade in trades:
            symbol = trade['symbol']
            new_shares[symbol] = new_shares.get(symbol, 0) + trade['shares']

        return new_shares

    def _calculate_withdrawal(
        self,
        portfolio_value: float,
        date: pd.Timestamp,
        strategy: WithdrawalStrategy
    ) -> float:
        """Calculate withdrawal amount."""
        # Simple withdrawal calculation
        return portfolio_value * strategy.initial_withdrawal_rate / 12  # Monthly

    def _calculate_performance_metrics(
        self,
        portfolio_df: pd.DataFrame,
        transactions: List[Dict],
        market_data: pd.DataFrame
    ) -> BacktestResults:
        """Calculate comprehensive performance metrics."""
        # Calculate returns
        portfolio_values = portfolio_df['value']
        returns = portfolio_values.pct_change().dropna()

        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        risk_free = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = (annualized_return - risk_free) / volatility if volatility > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free) / downside_vol if downside_vol > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # Trading metrics
        total_trades = len(transactions)
        transaction_costs = sum(t['cost'] for t in transactions)

        # Turnover (simplified)
        turnover = total_trades / len(portfolio_df) if len(portfolio_df) > 0 else 0

        # Benchmark comparison
        benchmark_return = None
        tracking_error = None
        information_ratio = None

        if self.config.benchmark_symbol:
            benchmark_data = market_data[market_data['symbol'] == self.config.benchmark_symbol]
            if not benchmark_data.empty:
                benchmark_returns = benchmark_data['close'].pct_change().dropna()
                benchmark_return = (benchmark_data['close'].iloc[-1] - benchmark_data['close'].iloc[0]) / benchmark_data['close'].iloc[0]

                # Align returns
                aligned_returns = pd.concat([returns, benchmark_returns], axis=1).dropna()
                if not aligned_returns.empty:
                    active_returns = aligned_returns.iloc[:, 0] - aligned_returns.iloc[:, 1]
                    tracking_error = active_returns.std() * np.sqrt(252)
                    information_ratio = (annualized_return - benchmark_return) / tracking_error if tracking_error > 0 else 0

        # Time series analysis
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        annual_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        rolling_sharpe = returns.rolling(window=252).apply(
            lambda x: (x.mean() * 252 - risk_free) / (x.std() * np.sqrt(252))
        )

        # Create weights history DataFrame
        weights_list = [w['weights'] for w in portfolio_df.to_dict('records')]
        weights_df = pd.DataFrame(weights_list, index=portfolio_df.index)

        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=0,  # Would need market data to calculate
            alpha=0,  # Would need market data to calculate
            total_trades=total_trades,
            turnover=turnover,
            transaction_costs=transaction_costs,
            taxes_paid=0,  # Simplified - would need trade details
            benchmark_return=benchmark_return,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            portfolio_values=portfolio_values,
            returns=returns,
            weights_history=weights_df,
            transactions=pd.DataFrame(transactions),
            monthly_returns=monthly_returns,
            annual_returns=annual_returns,
            rolling_sharpe=rolling_sharpe
        )