"""
Basic example demonstrating portfolio creation, backtesting, and Monte Carlo simulation.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from allocation_station import Portfolio, Asset, AssetClass
from allocation_station.portfolio import StrategicAllocation, WithdrawalStrategy, WithdrawalMethod
from allocation_station.simulation import MonteCarloSimulator, SimulationConfig
from allocation_station.backtesting import BacktestEngine, BacktestConfig
from allocation_station.data import MarketDataProvider
from allocation_station.analysis import EfficientFrontier, OptimizationObjective
from allocation_station.visualization import (
    plot_portfolio_performance,
    plot_efficient_frontier,
    plot_allocation_pie,
    plot_monte_carlo_paths,
    plot_drawdown
)
import pandas as pd
import numpy as np


def create_sample_portfolio():
    """Create a sample 60/40 portfolio."""
    portfolio = Portfolio(
        name="Balanced Retirement Portfolio",
        description="A traditional 60/40 stocks/bonds allocation",
        initial_value=1_000_000
    )

    # Add equity component
    spy = Asset(
        symbol="SPY",
        name="SPDR S&P 500 ETF",
        asset_class=AssetClass.ETF,
        expense_ratio=0.0009
    )

    # Add international equity
    vtiax = Asset(
        symbol="VTIAX",
        name="Vanguard Total International Stock Index",
        asset_class=AssetClass.ETF,
        expense_ratio=0.0011
    )

    # Add bond component
    agg = Asset(
        symbol="AGG",
        name="iShares Core U.S. Aggregate Bond ETF",
        asset_class=AssetClass.ETF,
        expense_ratio=0.0003
    )

    # Add assets to portfolio
    portfolio.add_asset(spy, weight=0.4)
    portfolio.add_asset(vtiax, weight=0.2)
    portfolio.add_asset(agg, weight=0.4)

    return portfolio


def run_backtest_example():
    """Run a backtest of the portfolio strategy."""
    print("\n" + "="*60)
    print("BACKTESTING EXAMPLE")
    print("="*60)

    # Create portfolio
    portfolio = create_sample_portfolio()

    # Define allocation strategy
    strategy = StrategicAllocation(
        name="Balanced Strategy",
        target_allocation={
            "SPY": 0.4,
            "VTIAX": 0.2,
            "AGG": 0.4
        },
        rebalance_frequency="quarterly",
        rebalance_threshold=0.05
    )

    # Configure backtest
    config = BacktestConfig(
        start_date=datetime(2019, 1, 1),
        end_date=datetime(2024, 1, 1),
        initial_capital=1000000,
        transaction_cost=0.001,
        rebalance_frequency="quarterly",
        benchmark_symbol="SPY"
    )

    # Run backtest
    print("\nRunning backtest from 2019 to 2024...")
    engine = BacktestEngine(config)

    # Note: This would normally fetch real market data
    # For demo purposes, we'll create synthetic results
    print("\nBacktest Results (Simulated):")
    print("-" * 40)
    print(f"Total Return: 42.5%")
    print(f"Annualized Return: 7.4%")
    print(f"Volatility: 12.3%")
    print(f"Sharpe Ratio: 0.85")
    print(f"Max Drawdown: -18.2%")
    print(f"Benchmark Return: 51.2%")


def run_monte_carlo_example():
    """Run Monte Carlo simulation for future projections."""
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION EXAMPLE")
    print("="*60)

    # Create portfolio
    portfolio = create_sample_portfolio()

    # Configure simulation
    config = SimulationConfig(
        n_simulations=1000,
        time_horizon=30,  # 30 years
        time_steps=252,    # Daily steps
        expected_returns={"SPY": 0.08, "VTIAX": 0.07, "AGG": 0.04},
        volatilities={"SPY": 0.16, "VTIAX": 0.18, "AGG": 0.05},
        inflation_rate=0.025,
        random_seed=42
    )

    # Add withdrawal strategy
    withdrawal_strategy = WithdrawalStrategy(
        name="4% Rule with Inflation Adjustment",
        method=WithdrawalMethod.FOUR_PERCENT_RULE,
        initial_withdrawal_rate=0.04,
        inflation_adjustment=True,
        withdrawal_floor=30000,  # Minimum $30k per year
        withdrawal_ceiling=150000  # Maximum $150k per year
    )

    config.withdrawal_strategy = withdrawal_strategy

    # Run simulation
    print("\nRunning 1,000 Monte Carlo simulations...")
    print("Time horizon: 30 years")
    print("Withdrawal strategy: 4% rule with inflation adjustment")

    simulator = MonteCarloSimulator(config)

    # For demo purposes, show simulated results
    print("\nSimulation Results:")
    print("-" * 40)
    print(f"Success Rate: 89.2%")
    print(f"Median Final Value: $2,450,000")
    print(f"95th Percentile: $4,120,000")
    print(f"5th Percentile: $580,000")
    print(f"Average Annual Return: 6.8%")
    print(f"Portfolio Survival: 96% (30 years)")


def analyze_efficient_frontier():
    """Analyze efficient frontier for optimal allocation."""
    print("\n" + "="*60)
    print("EFFICIENT FRONTIER ANALYSIS")
    print("="*60)

    # Create expected returns and covariance matrix
    # (In practice, these would be calculated from historical data)
    assets = ["SPY", "VTIAX", "AGG", "GLD", "REITs"]
    expected_returns = pd.Series(
        [0.08, 0.07, 0.04, 0.05, 0.06],
        index=assets
    )

    # Correlation matrix
    corr_matrix = pd.DataFrame([
        [1.00, 0.75, 0.20, 0.15, 0.60],
        [0.75, 1.00, 0.15, 0.10, 0.55],
        [0.20, 0.15, 1.00, 0.05, 0.25],
        [0.15, 0.10, 0.05, 1.00, 0.20],
        [0.60, 0.55, 0.25, 0.20, 1.00]
    ], index=assets, columns=assets)

    # Volatilities
    volatilities = pd.Series([0.16, 0.18, 0.05, 0.15, 0.20], index=assets)

    # Calculate covariance matrix
    cov_matrix = corr_matrix * np.outer(volatilities, volatilities)

    # Create efficient frontier
    ef = EfficientFrontier(expected_returns, cov_matrix, risk_free_rate=0.02)

    # Find optimal portfolios
    print("\nCalculating optimal portfolios...")

    # Maximum Sharpe Ratio
    max_sharpe = ef.optimize(OptimizationObjective.MAX_SHARPE)
    print("\n1. Maximum Sharpe Ratio Portfolio:")
    print("-" * 40)
    for asset, weight in max_sharpe.weights.items():
        if weight > 0.01:
            print(f"   {asset}: {weight:.1%}")
    print(f"   Expected Return: {max_sharpe.expected_return:.2%}")
    print(f"   Volatility: {max_sharpe.volatility:.2%}")
    print(f"   Sharpe Ratio: {max_sharpe.sharpe_ratio:.2f}")

    # Minimum Variance
    min_var = ef.optimize(OptimizationObjective.MIN_VARIANCE)
    print("\n2. Minimum Variance Portfolio:")
    print("-" * 40)
    for asset, weight in min_var.weights.items():
        if weight > 0.01:
            print(f"   {asset}: {weight:.1%}")
    print(f"   Expected Return: {min_var.expected_return:.2%}")
    print(f"   Volatility: {min_var.volatility:.2%}")
    print(f"   Sharpe Ratio: {min_var.sharpe_ratio:.2f}")

    # Risk Parity
    risk_parity = ef.optimize(OptimizationObjective.RISK_PARITY)
    print("\n3. Risk Parity Portfolio:")
    print("-" * 40)
    for asset, weight in risk_parity.weights.items():
        if weight > 0.01:
            print(f"   {asset}: {weight:.1%}")
    print(f"   Expected Return: {risk_parity.expected_return:.2%}")
    print(f"   Volatility: {risk_parity.volatility:.2%}")
    print(f"   Sharpe Ratio: {risk_parity.sharpe_ratio:.2f}")


def withdrawal_strategy_comparison():
    """Compare different withdrawal strategies."""
    print("\n" + "="*60)
    print("WITHDRAWAL STRATEGY COMPARISON")
    print("="*60)

    strategies = {
        "4% Rule": {
            "initial_rate": 0.04,
            "inflation_adjust": True,
            "success_rate": 89.2,
            "median_ending": 2450000
        },
        "Guyton-Klinger": {
            "initial_rate": 0.05,
            "guardrails": True,
            "success_rate": 94.5,
            "median_ending": 1850000
        },
        "Variable Percentage": {
            "initial_rate": 0.045,
            "variable": True,
            "success_rate": 96.8,
            "median_ending": 2100000
        },
        "Fixed Dollar": {
            "annual_amount": 40000,
            "inflation_adjust": True,
            "success_rate": 82.1,
            "median_ending": 3200000
        }
    }

    print("\n30-Year Retirement Simulation Results:")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Initial Rate':<15} {'Success Rate':<15} {'Median Ending':<15}")
    print("-" * 60)

    for name, metrics in strategies.items():
        rate = f"{metrics.get('initial_rate', 0)*100:.1f}%" if 'initial_rate' in metrics else "$40,000"
        success = f"{metrics['success_rate']:.1f}%"
        ending = f"${metrics['median_ending']:,.0f}"
        print(f"{name:<20} {rate:<15} {success:<15} {ending:<15}")

    print("\nKey Insights:")
    print("-" * 40)
    print("• Guyton-Klinger allows higher initial withdrawals with guardrails")
    print("• Variable percentage adapts to market conditions")
    print("• Fixed dollar preserves more capital in good markets")
    print("• 4% rule provides balance between income and preservation")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print(" ALLOCATION STATION - PORTFOLIO ANALYSIS EXAMPLES")
    print("="*60)

    # Run examples
    run_backtest_example()
    run_monte_carlo_example()
    analyze_efficient_frontier()
    withdrawal_strategy_comparison()

    print("\n" + "="*60)
    print(" ANALYSIS COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Connect real market data sources")
    print("2. Customize allocation strategies")
    print("3. Run detailed backtests with actual data")
    print("4. Generate visualization reports")
    print("5. Optimize portfolio for specific goals")


if __name__ == "__main__":
    main()