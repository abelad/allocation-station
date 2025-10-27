# Allocation Station User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Working with Assets and Portfolios](#working-with-assets-and-portfolios)
6. [Allocation Strategies](#allocation-strategies)
7. [Monte Carlo Simulations](#monte-carlo-simulations)
8. [Backtesting](#backtesting)
9. [Portfolio Optimization](#portfolio-optimization)
10. [Risk Analysis](#risk-analysis)
11. [Withdrawal Strategies](#withdrawal-strategies)
12. [Visualization](#visualization)
13. [Data Sources](#data-sources)
14. [Advanced Features](#advanced-features)
15. [Best Practices](#best-practices)
16. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is Allocation Station?

Allocation Station is a comprehensive Python framework for portfolio management, asset allocation strategy testing, and retirement planning analysis. It provides tools for:

- **Portfolio Construction**: Build and manage multi-asset portfolios
- **Strategy Testing**: Backtest allocation strategies against historical data
- **Risk Analysis**: Comprehensive risk metrics and stress testing
- **Monte Carlo Simulation**: Project future portfolio outcomes under uncertainty
- **Optimization**: Find optimal allocations using Modern Portfolio Theory
- **Retirement Planning**: Test withdrawal strategies and longevity scenarios
- **Visualization**: Interactive charts and dashboards

### Who Should Use This?

- **Individual Investors**: Test retirement strategies and portfolio allocations
- **Financial Advisors**: Analyze client portfolios and create allocation recommendations
- **Researchers**: Study portfolio theory and market behavior
- **Quantitative Analysts**: Build and test systematic investment strategies
- **Students**: Learn about portfolio management and financial modeling

### Key Features

- Multiple asset classes: stocks, bonds, ETFs, REITs, commodities, crypto, derivatives
- Advanced allocation strategies: strategic, tactical, risk parity, factor-based
- Sophisticated simulation engine with regime-switching and GARCH models
- Comprehensive risk analysis: VaR, CVaR, drawdown, stress testing
- Machine learning integration for forecasting
- Multiple data sources: Yahoo Finance, Alpha Vantage, FRED
- Interactive dashboards with Plotly
- Broker integrations for live trading

---

## Installation

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM (8GB recommended for large simulations)
- **Storage**: 500MB for package and data cache

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/allocation-station.git
cd allocation-station

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the package
pip install -e .
```

### Install with Optional Dependencies

```bash
# For development (includes testing and linting tools)
pip install -e .[dev]

# For documentation building
pip install -e .[docs]

# Install everything
pip install -e .[dev,docs]
```

### Verify Installation

```python
import allocation_station
print(allocation_station.__version__)  # Should print: 0.1.0
```

---

## Quick Start

### Your First Portfolio

Here's a simple example to get you started:

```python
from allocation_station import Portfolio, Asset, AssetClass
from allocation_station.portfolio import StrategicAllocation
from allocation_station.simulation import MonteCarloSimulator, SimulationConfig

# Step 1: Create a portfolio
portfolio = Portfolio(
    name="My Retirement Portfolio",
    initial_value=500000
)

# Step 2: Add assets
spy = Asset(
    symbol="SPY",
    name="S&P 500 ETF",
    asset_class=AssetClass.ETF,
    expense_ratio=0.0009
)
agg = Asset(
    symbol="AGG",
    name="Bond ETF",
    asset_class=AssetClass.ETF,
    expense_ratio=0.0003
)

portfolio.add_asset(spy, weight=0.6)
portfolio.add_asset(agg, weight=0.4)

# Step 3: Define allocation strategy
strategy = StrategicAllocation(
    name="60/40 Portfolio",
    target_allocation={"SPY": 0.6, "AGG": 0.4},
    rebalance_frequency="quarterly"
)

# Step 4: Run Monte Carlo simulation
config = SimulationConfig(
    n_simulations=1000,
    time_horizon=30,
    expected_returns={"SPY": 0.08, "AGG": 0.04},
    volatilities={"SPY": 0.16, "AGG": 0.05}
)

simulator = MonteCarloSimulator(config)
results = simulator.simulate(portfolio, strategy)

# Step 5: View results
print(f"Success Rate: {results.success_rate:.1%}")
print(f"Median Final Value: ${results.median_final_value:,.0f}")
```

---

## Core Concepts

### Assets

An **Asset** represents a financial instrument in your portfolio:

```python
from allocation_station import Asset, AssetClass

# Basic equity
stock = Asset(
    symbol="AAPL",
    name="Apple Inc.",
    asset_class=AssetClass.EQUITY
)

# ETF with expense ratio
etf = Asset(
    symbol="VTI",
    name="Vanguard Total Stock Market ETF",
    asset_class=AssetClass.ETF,
    expense_ratio=0.0003
)

# Bond with yield
bond = Asset(
    symbol="TLT",
    name="20+ Year Treasury Bond ETF",
    asset_class=AssetClass.BOND,
    current_yield=0.04
)
```

**Supported Asset Classes:**
- `EQUITY`: Individual stocks
- `BOND`: Bonds and fixed income
- `ETF`: Exchange-traded funds
- `MUTUAL_FUND`: Mutual funds
- `REIT`: Real estate investment trusts
- `COMMODITY`: Commodities and futures
- `CRYPTO`: Cryptocurrencies
- `CASH`: Cash and money market
- `OPTION`: Options contracts
- `ALTERNATIVE`: Private equity, hedge funds

### Portfolios

A **Portfolio** is a collection of assets with specific weights:

```python
from allocation_station import Portfolio

# Create portfolio
portfolio = Portfolio(
    name="Growth Portfolio",
    description="Aggressive growth strategy",
    initial_value=100000,
    currency="USD"
)

# Add assets with weights
portfolio.add_asset(Asset(symbol="VTI"), weight=0.7)
portfolio.add_asset(Asset(symbol="VXUS"), weight=0.2)
portfolio.add_asset(Asset(symbol="BND"), weight=0.1)

# Check allocation
print(portfolio.get_allocation())  # {'VTI': 0.7, 'VXUS': 0.2, 'BND': 0.1}

# Get portfolio metrics
metrics = portfolio.calculate_metrics()
print(f"Expected Return: {metrics['expected_return']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### Allocation Strategies

**Strategies** define how your portfolio is managed over time:

1. **Strategic (Buy & Hold)**: Fixed allocation with periodic rebalancing
2. **Tactical**: Dynamic allocation based on market conditions
3. **Risk Parity**: Equal risk contribution from each asset
4. **Target Date**: Glide path toward conservative allocation
5. **Factor-Based**: Allocation based on factor exposures

### Time Horizons

- **Short-term**: < 5 years
- **Medium-term**: 5-15 years
- **Long-term**: 15+ years (typical for retirement planning)

---

## Working with Assets and Portfolios

### Creating Assets

#### Basic Asset Creation

```python
from allocation_station import Asset, AssetClass

# Minimal asset
asset = Asset(symbol="MSFT", name="Microsoft")

# Detailed asset
asset = Asset(
    symbol="AAPL",
    name="Apple Inc.",
    asset_class=AssetClass.EQUITY,
    sector="Technology",
    expense_ratio=0.0,
    tax_treatment="qualified"
)
```

#### Enhanced Asset Types

```python
from allocation_station.core import (
    OptionAsset, OptionType,
    REITAsset, PropertyType,
    CryptoAsset,
    CommodityAsset, CommodityType
)

# Option contract
call_option = OptionAsset(
    symbol="AAPL",
    option_type=OptionType.CALL,
    strike_price=180.0,
    expiration_date="2025-06-20",
    premium=5.50
)

# REIT with specific metrics
reit = REITAsset(
    symbol="O",
    name="Realty Income",
    property_type=PropertyType.RETAIL,
    ffo=3.50,  # Funds From Operations
    affo=3.25,  # Adjusted FFO
    occupancy_rate=0.98
)

# Cryptocurrency
bitcoin = CryptoAsset(
    symbol="BTC-USD",
    name="Bitcoin",
    network="Bitcoin",
    volatility_adjustment=2.0  # Higher risk weighting
)

# Commodity
gold = CommodityAsset(
    symbol="GLD",
    commodity_type=CommodityType.PRECIOUS_METAL,
    storage_cost=0.004,  # Annual storage cost
    convenience_yield=0.01
)
```

### Building Portfolios

#### Adding Assets

```python
portfolio = Portfolio(name="Balanced Portfolio", initial_value=100000)

# Add single asset
portfolio.add_asset(Asset(symbol="SPY"), weight=0.6)

# Add multiple assets
assets_dict = {
    "SPY": 0.4,
    "VXUS": 0.2,
    "AGG": 0.3,
    "GLD": 0.1
}

for symbol, weight in assets_dict.items():
    portfolio.add_asset(Asset(symbol=symbol), weight=weight)
```

#### Modifying Allocations

```python
# Update weights
portfolio.update_allocation({"SPY": 0.5, "VXUS": 0.25, "AGG": 0.25})

# Rebalance to target allocation
portfolio.rebalance()

# Remove asset
portfolio.remove_asset("GLD")

# Get current values
current_values = portfolio.get_current_values()
```

#### Multi-Currency Portfolios

```python
from allocation_station.portfolio import MultiCurrencyPortfolio
from allocation_station.portfolio.advanced_features import FXRiskManager

# Create multi-currency portfolio
portfolio = MultiCurrencyPortfolio(
    name="Global Portfolio",
    base_currency="USD",
    initial_value=100000
)

# Add assets in different currencies
portfolio.add_asset(
    Asset(symbol="SPY"),  # USD
    weight=0.3,
    currency="USD"
)
portfolio.add_asset(
    Asset(symbol="EWU"),  # UK exposure
    weight=0.2,
    currency="GBP"
)
portfolio.add_asset(
    Asset(symbol="EWJ"),  # Japan exposure
    weight=0.2,
    currency="JPY"
)

# Setup FX hedging
fx_manager = FXRiskManager(hedge_ratio=0.5)  # 50% hedged
portfolio.set_fx_manager(fx_manager)

# Get FX exposure
fx_exposure = portfolio.get_fx_exposure()
```

---

## Allocation Strategies

### Strategic Allocation (Buy & Hold)

Fixed allocation with periodic rebalancing:

```python
from allocation_station.portfolio import StrategicAllocation

strategy = StrategicAllocation(
    name="Classic 60/40",
    target_allocation={
        "SPY": 0.6,
        "AGG": 0.4
    },
    rebalance_frequency="quarterly",  # or "monthly", "annual"
    rebalance_threshold=0.05,  # Rebalance if drift > 5%
    rebalance_method="to_target"  # or "cash_flow"
)

# Apply to portfolio
portfolio.set_strategy(strategy)
```

### Tactical Allocation

Dynamic allocation based on market signals:

```python
from allocation_station.portfolio import TacticalAllocation

strategy = TacticalAllocation(
    name="Momentum Strategy",
    base_allocation={
        "SPY": 0.5,
        "AGG": 0.3,
        "GLD": 0.2
    },
    signals={
        "moving_average_crossover": {
            "short_window": 50,
            "long_window": 200
        },
        "volatility_adjustment": {
            "target_volatility": 0.12,
            "max_leverage": 1.5
        }
    },
    adjustment_limits=0.2  # Max 20% deviation from base
)
```

### Risk Parity

Equal risk contribution from each asset:

```python
from allocation_station.portfolio import RiskParityAllocation

strategy = RiskParityAllocation(
    name="Risk Parity",
    assets=["SPY", "TLT", "GLD", "DBC"],  # Stocks, bonds, gold, commodities
    target_volatility=0.12,  # 12% annual volatility
    rebalance_frequency="monthly"
)

# Get risk contributions
risk_contrib = strategy.get_risk_contributions(portfolio)
```

### Target Date Glide Path

Automatic de-risking as target date approaches:

```python
from allocation_station.portfolio import TargetDateAllocation
from datetime import datetime

strategy = TargetDateAllocation(
    name="2045 Target Date",
    target_date=datetime(2045, 12, 31),
    current_date=datetime.now(),
    starting_allocation={
        "VTI": 0.9,  # 90% stocks initially
        "BND": 0.1
    },
    ending_allocation={
        "VTI": 0.3,  # 30% stocks at retirement
        "BND": 0.7
    },
    glide_path_type="linear"  # or "exponential"
)
```

### Factor-Based Allocation

Allocation based on factor exposures:

```python
from allocation_station.portfolio import FactorAllocation

strategy = FactorAllocation(
    name="Multi-Factor",
    target_factors={
        "value": 0.3,
        "momentum": 0.2,
        "quality": 0.2,
        "low_volatility": 0.3
    },
    factor_etfs={
        "value": "VTV",
        "momentum": "MTUM",
        "quality": "QUAL",
        "low_volatility": "USMV"
    },
    rebalance_frequency="monthly"
)
```

---

## Monte Carlo Simulations

### Basic Simulation

```python
from allocation_station.simulation import MonteCarloSimulator, SimulationConfig

# Configure simulation
config = SimulationConfig(
    n_simulations=10000,
    time_horizon=30,  # 30 years
    time_steps=252,   # Daily steps (trading days per year)
    expected_returns={
        "SPY": 0.08,  # 8% expected return
        "AGG": 0.04   # 4% expected return
    },
    volatilities={
        "SPY": 0.16,  # 16% volatility
        "AGG": 0.05   # 5% volatility
    },
    correlation_matrix=None,  # Will use historical if None
    inflation_rate=0.025,  # 2.5% inflation
    random_seed=42  # For reproducibility
)

# Run simulation
simulator = MonteCarloSimulator(config)
results = simulator.simulate(portfolio, strategy)

# Analyze results
print(f"Success Rate: {results.success_rate:.1%}")
print(f"Median Final Value: ${results.median_final_value:,.0f}")
print(f"10th Percentile: ${results.percentile(10):,.0f}")
print(f"90th Percentile: ${results.percentile(90):,.0f}")
print(f"Probability of Running Out: {results.ruin_probability:.1%}")
```

### Advanced Simulation Models

#### Regime-Switching Model

```python
from allocation_station.simulation import RegimeSwitchingSimulator

simulator = RegimeSwitchingSimulator(
    regimes={
        "bull": {"returns": 0.12, "volatility": 0.12, "probability": 0.6},
        "bear": {"returns": -0.05, "volatility": 0.25, "probability": 0.2},
        "normal": {"returns": 0.08, "volatility": 0.16, "probability": 0.2}
    },
    transition_matrix=[
        [0.7, 0.2, 0.1],  # From bull
        [0.3, 0.4, 0.3],  # From bear
        [0.3, 0.2, 0.5]   # From normal
    ]
)

results = simulator.simulate(portfolio, n_simulations=10000, time_horizon=30)
```

#### GARCH Volatility Model

```python
from allocation_station.simulation import GARCHSimulator

simulator = GARCHSimulator(
    omega=0.00001,  # Base variance
    alpha=0.1,      # ARCH term
    beta=0.85,      # GARCH term
    mean_return=0.08
)

results = simulator.simulate(portfolio, n_simulations=5000, time_horizon=30)
```

#### Copula-Based Dependencies

```python
from allocation_station.simulation import CopulaSimulator

simulator = CopulaSimulator(
    copula_type="t",  # t-copula for tail dependence
    degrees_of_freedom=5,
    expected_returns={"SPY": 0.08, "AGG": 0.04, "GLD": 0.05},
    volatilities={"SPY": 0.16, "AGG": 0.05, "GLD": 0.15}
)

results = simulator.simulate(portfolio, n_simulations=10000)
```

### Parallel Processing

```python
# Use parallel processing for faster simulations
config = SimulationConfig(
    n_simulations=100000,
    time_horizon=30,
    n_jobs=-1  # Use all available cores
)

simulator = MonteCarloSimulator(config)
results = simulator.simulate(portfolio, strategy)
```

---

## Backtesting

### Basic Backtest

```python
from allocation_station.backtesting import BacktestEngine, BacktestConfig
from datetime import datetime

# Configure backtest
config = BacktestConfig(
    start_date=datetime(2010, 1, 1),
    end_date=datetime(2024, 1, 1),
    initial_capital=100000,
    transaction_cost=0.001,  # 0.1% per trade
    slippage=0.0005,  # 0.05% slippage
    rebalance_frequency="quarterly",
    benchmark_symbol="SPY"
)

# Run backtest
engine = BacktestEngine(config)
results = engine.run(portfolio, strategy)

# View results
print(f"Total Return: {results.total_return:.2%}")
print(f"CAGR: {results.cagr:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
```

### Performance Metrics

```python
# Get detailed metrics
metrics = results.get_metrics()

print("\nPerformance Metrics:")
print(f"Alpha: {metrics['alpha']:.2%}")
print(f"Beta: {metrics['beta']:.2f}")
print(f"Information Ratio: {metrics['information_ratio']:.2f}")
print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")

# Benchmark comparison
print(f"\nVs. Benchmark ({config.benchmark_symbol}):")
print(f"Excess Return: {results.excess_return:.2%}")
print(f"Tracking Error: {results.tracking_error:.2%}")
```

### Transaction Costs Analysis

```python
# Analyze transaction costs
cost_analysis = results.get_transaction_costs()

print(f"Total Transaction Costs: ${cost_analysis['total_costs']:,.2f}")
print(f"Cost as % of Portfolio: {cost_analysis['cost_percentage']:.2%}")
print(f"Number of Rebalances: {cost_analysis['n_rebalances']}")
print(f"Average Cost per Rebalance: ${cost_analysis['avg_cost']:,.2f}")
```

---

## Portfolio Optimization

### Efficient Frontier

```python
from allocation_station.analysis import EfficientFrontier
import pandas as pd

# Define expected returns and covariance matrix
assets = ["SPY", "TLT", "GLD", "VNQ"]
expected_returns = pd.Series([0.08, 0.04, 0.05, 0.07], index=assets)
cov_matrix = portfolio.calculate_covariance_matrix()

# Create efficient frontier
ef = EfficientFrontier(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=0.02
)

# Find optimal portfolios
max_sharpe = ef.optimize("max_sharpe")
min_variance = ef.optimize("min_variance")
risk_parity = ef.optimize("risk_parity")

print("\nMaximum Sharpe Ratio Portfolio:")
for asset, weight in max_sharpe.weights.items():
    print(f"  {asset}: {weight:.1%}")
print(f"Expected Return: {max_sharpe.expected_return:.2%}")
print(f"Volatility: {max_sharpe.volatility:.2%}")
print(f"Sharpe Ratio: {max_sharpe.sharpe_ratio:.2f}")
```

### Black-Litterman Model

Combine market equilibrium with investor views:

```python
from allocation_station.optimization import BlackLittermanOptimizer

# Define market cap weights (equilibrium)
market_weights = {"SPY": 0.55, "TLT": 0.25, "GLD": 0.1, "VNQ": 0.1}

# Define investor views
views = [
    {"asset": "SPY", "view": 0.10, "confidence": 0.7},  # Bullish on stocks
    {"asset": "GLD", "view": 0.03, "confidence": 0.5}   # Neutral on gold
]

# Run Black-Litterman
bl = BlackLittermanOptimizer(
    market_weights=market_weights,
    risk_aversion=2.5,
    cov_matrix=cov_matrix
)

optimal_weights = bl.optimize(views)
```

### Constrained Optimization

```python
from allocation_station.optimization import ConstrainedOptimizer

# Define constraints
constraints = {
    "min_weights": {"SPY": 0.2, "AGG": 0.1},  # Minimums
    "max_weights": {"SPY": 0.7, "GLD": 0.15},  # Maximums
    "asset_class_limits": {
        "equity": {"min": 0.4, "max": 0.8},
        "bond": {"min": 0.2, "max": 0.6}
    },
    "sector_limits": {
        "technology": {"max": 0.3}
    }
}

optimizer = ConstrainedOptimizer(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    constraints=constraints
)

optimal_portfolio = optimizer.optimize("max_sharpe")
```

### Robust Optimization

Handle estimation uncertainty:

```python
from allocation_station.optimization import RobustOptimizer

# Robust optimization with uncertainty
optimizer = RobustOptimizer(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    return_uncertainty=0.02,  # 2% uncertainty in returns
    covariance_uncertainty=0.1,  # 10% uncertainty in covariance
    worst_case_weighting=0.5  # Balance expected and worst-case
)

robust_portfolio = optimizer.optimize()
```

---

## Risk Analysis

### Value at Risk (VaR) and CVaR

```python
from allocation_station.analysis import RiskAnalyzer

analyzer = RiskAnalyzer(portfolio)

# Calculate VaR
var_95 = analyzer.calculate_var(confidence_level=0.95, time_horizon=1)
var_99 = analyzer.calculate_var(confidence_level=0.99, time_horizon=1)

print(f"1-Day 95% VaR: ${var_95:,.0f}")
print(f"1-Day 99% VaR: ${var_99:,.0f}")

# Calculate CVaR (Conditional VaR / Expected Shortfall)
cvar_95 = analyzer.calculate_cvar(confidence_level=0.95)
print(f"1-Day 95% CVaR: ${cvar_95:,.0f}")
```

### Drawdown Analysis

```python
# Maximum drawdown
max_dd = analyzer.calculate_max_drawdown()
print(f"Maximum Drawdown: {max_dd:.2%}")

# Drawdown duration
dd_stats = analyzer.drawdown_statistics()
print(f"Average Drawdown Duration: {dd_stats['avg_duration']} days")
print(f"Longest Drawdown: {dd_stats['max_duration']} days")
print(f"Recovery Time: {dd_stats['avg_recovery_time']} days")
```

### Stress Testing

```python
from allocation_station.analysis import StressTester

tester = StressTester(portfolio)

# Predefined scenarios
scenarios = {
    "financial_crisis_2008": {"SPY": -0.37, "AGG": 0.05},
    "covid_crash_2020": {"SPY": -0.34, "AGG": 0.08},
    "dot_com_bubble": {"SPY": -0.49, "AGG": 0.12}
}

results = tester.run_scenarios(scenarios)

for scenario, outcome in results.items():
    print(f"\n{scenario}:")
    print(f"  Portfolio Loss: {outcome['portfolio_loss']:.2%}")
    print(f"  Recovery Time: {outcome['recovery_time']} months")
```

### Tail Risk Analysis

```python
from allocation_station.analysis import TailRiskAnalyzer

analyzer = TailRiskAnalyzer(portfolio)

# Analyze tail behavior
tail_metrics = analyzer.analyze_tails()

print(f"Left Tail Risk (5%): {tail_metrics['left_tail_5']:.2%}")
print(f"Right Tail (95%): {tail_metrics['right_tail_95']:.2%}")
print(f"Tail Ratio: {tail_metrics['tail_ratio']:.2f}")
print(f"Kurtosis: {tail_metrics['kurtosis']:.2f}")
```

### Correlation Breakdown

```python
from allocation_station.analysis import CorrelationAnalyzer

analyzer = CorrelationAnalyzer(portfolio)

# Check correlation stability
breakdown_risk = analyzer.check_correlation_breakdown()

print(f"Correlation Breakdown Risk: {breakdown_risk['risk_score']:.2f}")
print(f"Stable Period Correlation: {breakdown_risk['stable_corr']:.2f}")
print(f"Crisis Period Correlation: {breakdown_risk['crisis_corr']:.2f}")
```

---

## Withdrawal Strategies

### 4% Rule

Classic retirement withdrawal strategy:

```python
from allocation_station.portfolio import WithdrawalStrategy, WithdrawalMethod

strategy = WithdrawalStrategy(
    name="4% Rule",
    method=WithdrawalMethod.FOUR_PERCENT_RULE,
    initial_withdrawal_rate=0.04,
    inflation_adjustment=True
)

# Test strategy
results = simulator.simulate(
    portfolio,
    allocation_strategy,
    withdrawal_strategy=strategy
)

print(f"Success Rate: {results.success_rate:.1%}")
print(f"Median Withdrawals: ${results.median_withdrawals:,.0f}/year")
```

### Guyton-Klinger

Dynamic withdrawals with guardrails:

```python
strategy = WithdrawalStrategy(
    name="Guyton-Klinger",
    method=WithdrawalMethod.GUYTON_KLINGER,
    initial_withdrawal_rate=0.05,
    upper_guardrail=0.20,  # Increase withdrawal if +20%
    lower_guardrail=0.15,  # Decrease withdrawal if -15%
    prosperity_rule=True,
    capital_preservation_rule=True
)
```

### Variable Percentage Withdrawal

```python
strategy = WithdrawalStrategy(
    name="Variable Percentage",
    method=WithdrawalMethod.VARIABLE_PERCENTAGE,
    withdrawal_percentage=0.04,
    adjustment_frequency="annual",
    floor=30000,  # Minimum annual withdrawal
    ceiling=100000  # Maximum annual withdrawal
)
```

### Bucket Strategy

Time-segmented approach:

```python
from allocation_station.withdrawal import BucketStrategy

strategy = BucketStrategy(
    buckets={
        "immediate": {
            "years": 3,
            "allocation": {"CASH": 1.0},
            "amount": 120000
        },
        "short_term": {
            "years": 7,
            "allocation": {"AGG": 0.7, "SPY": 0.3},
            "amount": 280000
        },
        "long_term": {
            "years": 20,
            "allocation": {"SPY": 0.7, "VXUS": 0.3},
            "amount": 600000
        }
    },
    refill_strategy="annual_rebalance"
)
```

### Dynamic Programming Optimization

Optimal withdrawal using dynamic programming:

```python
from allocation_station.withdrawal import OptimalWithdrawal

optimizer = OptimalWithdrawal(
    portfolio=portfolio,
    time_horizon=30,
    utility_function="CRRA",  # Constant Relative Risk Aversion
    risk_aversion=3.0,
    bequest_motive=True,
    bequest_weight=0.3
)

optimal_path = optimizer.compute_optimal_withdrawals()
```

### Social Security Integration

```python
from allocation_station.withdrawal import SocialSecurityOptimizer

ss_optimizer = SocialSecurityOptimizer(
    birth_date="1960-05-15",
    full_retirement_benefit=2500,  # Monthly at FRA
    spousal_benefit=1250,
    spouse_birth_date="1962-08-20"
)

# Find optimal claiming age
optimal_strategy = ss_optimizer.optimize(
    portfolio=portfolio,
    life_expectancy=90,
    discount_rate=0.03
)

print(f"Optimal Claiming Age: {optimal_strategy['age']}")
print(f"Total Lifetime Value: ${optimal_strategy['lifetime_value']:,.0f}")
```

---

## Visualization

### Portfolio Performance Charts

```python
from allocation_station.visualization import (
    plot_portfolio_performance,
    plot_allocation_pie,
    plot_drawdown,
    plot_rolling_metrics
)

# Performance over time
fig = plot_portfolio_performance(
    portfolio_returns=results.returns,
    benchmark_returns=results.benchmark_returns,
    title="Portfolio Performance vs Benchmark"
)
fig.show()

# Current allocation
fig = plot_allocation_pie(
    portfolio.get_allocation(),
    title="Current Portfolio Allocation"
)
fig.show()

# Drawdown analysis
fig = plot_drawdown(
    returns=results.returns,
    title="Portfolio Drawdown Analysis"
)
fig.show()

# Rolling metrics
fig = plot_rolling_metrics(
    returns=results.returns,
    window=252,  # 1 year
    metrics=["sharpe", "volatility", "max_drawdown"]
)
fig.show()
```

### Efficient Frontier Visualization

```python
from allocation_station.visualization import plot_efficient_frontier

fig = plot_efficient_frontier(
    efficient_frontier=ef,
    portfolios={
        "Current": portfolio.get_allocation(),
        "Max Sharpe": max_sharpe.weights,
        "Min Variance": min_variance.weights
    },
    show_assets=True
)
fig.show()
```

### Monte Carlo Paths

```python
from allocation_station.visualization import plot_monte_carlo_paths

fig = plot_monte_carlo_paths(
    simulation_results=results,
    n_paths=100,  # Show 100 random paths
    show_percentiles=[10, 25, 50, 75, 90],
    title="30-Year Portfolio Projections"
)
fig.show()
```

### Interactive Dashboard

```python
from allocation_station.ui import create_dashboard

# Create interactive dashboard
app = create_dashboard(
    portfolio=portfolio,
    simulation_results=results,
    backtest_results=backtest_results
)

# Run dashboard
app.run_server(debug=True, port=8050)
# Open browser to http://localhost:8050
```

---

## Data Sources

### Yahoo Finance (Default)

```python
from allocation_station.data import MarketDataProvider

provider = MarketDataProvider(source="yahoo")

# Fetch historical data
data = provider.get_historical_data(
    symbols=["SPY", "AGG", "GLD"],
    start_date="2020-01-01",
    end_date="2024-01-01",
    frequency="daily"
)
```

### Alpha Vantage

```python
provider = MarketDataProvider(
    source="alphavantage",
    api_key="YOUR_API_KEY"
)

data = provider.get_historical_data(
    symbols=["AAPL"],
    start_date="2020-01-01"
)
```

### FRED (Federal Reserve Economic Data)

```python
from allocation_station.data import FREDDataProvider

fred = FREDDataProvider(api_key="YOUR_FRED_API_KEY")

# Get economic indicators
gdp = fred.get_series("GDP")
inflation = fred.get_series("CPIAUCSL")
unemployment = fred.get_series("UNRATE")
```

### Custom Data Source

```python
from allocation_station.data import CustomDataSource

class MyDataSource(CustomDataSource):
    def fetch_data(self, symbol, start_date, end_date):
        # Implement your data fetching logic
        return data_dataframe

provider = MarketDataProvider(source=MyDataSource())
```

---

## Advanced Features

### Tax-Loss Harvesting

```python
from allocation_station.portfolio import TaxLossHarvester

harvester = TaxLossHarvester(
    portfolio=portfolio,
    tax_rate=0.24,  # 24% marginal rate
    wash_sale_period=30,  # Days to avoid wash sale
    minimum_loss=1000  # Minimum loss to harvest
)

# Check for harvesting opportunities
opportunities = harvester.find_opportunities()

# Execute harvest
harvest_trades = harvester.execute_harvest(opportunities[0])
```

### Leveraged Portfolios

```python
from allocation_station.portfolio import LeveragedPortfolio

portfolio = LeveragedPortfolio(
    name="Leveraged 60/40",
    initial_value=100000,
    leverage_ratio=1.5,  # 150% exposure
    margin_rate=0.05,  # 5% borrowing cost
    maintenance_margin=0.3  # 30% maintenance requirement
)
```

### Factor Analysis

```python
from allocation_station.analysis import FactorAnalyzer

analyzer = FactorAnalyzer(portfolio)

# Decompose returns into factors
factor_exposure = analyzer.get_factor_exposures(
    factors=["market", "size", "value", "momentum", "quality"]
)

print("Factor Exposures:")
for factor, exposure in factor_exposure.items():
    print(f"  {factor}: {exposure:.2f}")
```

### Machine Learning Integration

```python
from allocation_station.ml import ReturnPredictor

# Train return prediction model
predictor = ReturnPredictor(
    model_type="random_forest",
    features=["momentum", "volatility", "volume", "sentiment"],
    lookback_period=252
)

predictor.train(historical_data)

# Predict future returns
predictions = predictor.predict(current_data)
```

---

## Best Practices

### Portfolio Construction

1. **Diversification**: Include 15-30 holdings across asset classes
2. **Rebalancing**: Quarterly or annually, with 5% threshold
3. **Cost Awareness**: Minimize expense ratios and transaction costs
4. **Tax Efficiency**: Use tax-advantaged accounts strategically
5. **Risk Management**: Monitor correlations and tail risks

### Simulation Configuration

1. **Number of Simulations**: Use at least 1,000 for preliminary, 10,000+ for final analysis
2. **Time Horizon**: Match your investment timeframe
3. **Return Assumptions**: Be conservative, use historical data
4. **Include Costs**: Model transaction costs and fees
5. **Stress Testing**: Test under adverse scenarios

### Backtesting

1. **Lookback Period**: Use at least 10 years of data
2. **Transaction Costs**: Include realistic costs (0.1-0.5%)
3. **Benchmark**: Always compare to relevant benchmark
4. **Out-of-Sample**: Test on data not used for optimization
5. **Robustness**: Test across different time periods

### Performance Monitoring

1. **Regular Reviews**: Monthly or quarterly
2. **Rebalance Discipline**: Stick to your rebalancing rules
3. **Cost Tracking**: Monitor all fees and expenses
4. **Risk Metrics**: Track Sharpe ratio, drawdown, volatility
5. **Goal Progress**: Monitor progress toward financial goals

---

## Troubleshooting

### Common Issues

#### Data Download Failures

```python
# Problem: Data download fails
# Solution: Check internet connection and API limits

try:
    data = provider.get_historical_data(symbols=["SPY"])
except Exception as e:
    print(f"Error: {e}")
    # Use cached data or try alternative source
    provider = MarketDataProvider(source="alphavantage")
```

#### Memory Issues with Large Simulations

```python
# Problem: Out of memory with 100,000 simulations
# Solution: Use batch processing

from allocation_station.simulation import BatchSimulator

simulator = BatchSimulator(
    batch_size=10000,  # Process 10k at a time
    n_simulations=100000
)
```

#### Optimization Not Converging

```python
# Problem: Optimizer fails to converge
# Solution: Relax constraints or use robust optimizer

optimizer = ConstrainedOptimizer(
    expected_returns=returns,
    cov_matrix=cov_matrix,
    solver="ECOS",  # Try different solver
    max_iterations=1000
)
```

### Performance Tips

1. **Use Parallel Processing**: Set `n_jobs=-1` for simulations
2. **Cache Data**: Enable caching for market data
3. **Reduce Time Steps**: Use monthly instead of daily for long horizons
4. **Vectorize Operations**: Avoid loops in custom code
5. **Profile Code**: Use profiling tools to find bottlenecks

### Getting Help

- **Documentation**: Check [API Reference](API_REFERENCE.md)
- **Examples**: Review example scripts in `examples/` folder
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: support@allocation-station.dev

---

## Next Steps

- Explore the [API Reference](API_REFERENCE.md) for detailed function documentation
- Review the [Cookbook](COOKBOOK.md) for common recipes and patterns
- Read the [Theoretical Background](THEORY.md) to understand the mathematics
- Check out [Advanced Tutorials](TUTORIALS.md) for in-depth guides
- Join our community for discussions and updates

---

**Last Updated**: January 2025
**Version**: 0.1.0
**License**: MIT
