# Allocation Station Tutorials

Step-by-step tutorials for mastering portfolio analysis and management.

## Table of Contents

1. [Tutorial 1: Building Your First Portfolio](#tutorial-1-building-your-first-portfolio)
2. [Tutorial 2: Running Monte Carlo Simulations](#tutorial-2-running-monte-carlo-simulations)
3. [Tutorial 3: Backtesting Strategies](#tutorial-3-backtesting-strategies)
4. [Tutorial 4: Portfolio Optimization](#tutorial-4-portfolio-optimization)
5. [Tutorial 5: Retirement Planning](#tutorial-5-retirement-planning)
6. [Tutorial 6: Risk Analysis](#tutorial-6-risk-analysis)
7. [Tutorial 7: Custom Data Sources](#tutorial-7-custom-data-sources)
8. [Tutorial 8: Building Interactive Dashboards](#tutorial-8-building-interactive-dashboards)
9. [Tutorial 9: Multi-Asset Portfolio with Alternatives](#tutorial-9-multi-asset-portfolio-with-alternatives)
10. [Tutorial 10: Machine Learning Integration](#tutorial-10-machine-learning-integration)

---

## Tutorial 1: Building Your First Portfolio

**Goal**: Create a simple balanced portfolio and understand its characteristics.

**Duration**: 15 minutes

### Step 1: Set Up Your Environment

```python
# Import required modules
from allocation_station import Portfolio, Asset, AssetClass
import pandas as pd
import numpy as np

# Verify installation
print("Allocation Station is ready!")
```

### Step 2: Create Portfolio Object

```python
# Create a portfolio for retirement
portfolio = Portfolio(
    name="Retirement Portfolio",
    description="Balanced allocation for retirement",
    initial_value=500000,  # Starting with $500k
    currency="USD"
)

print(f"Created portfolio: {portfolio.name}")
print(f"Initial value: ${portfolio.initial_value:,.0f}")
```

### Step 3: Define Assets

```python
# Define equity assets
spy = Asset(
    symbol="SPY",
    name="SPDR S&P 500 ETF",
    asset_class=AssetClass.ETF,
    expense_ratio=0.0009,  # 0.09% expense ratio
    sector="Equity"
)

vxus = Asset(
    symbol="VXUS",
    name="Vanguard Total International Stock ETF",
    asset_class=AssetClass.ETF,
    expense_ratio=0.0007,
    sector="International Equity"
)

# Define bond asset
agg = Asset(
    symbol="AGG",
    name="iShares Core U.S. Aggregate Bond ETF",
    asset_class=AssetClass.ETF,
    expense_ratio=0.0003,
    sector="Fixed Income"
)

# Define alternative
gld = Asset(
    symbol="GLD",
    name="SPDR Gold Shares",
    asset_class=AssetClass.COMMODITY,
    expense_ratio=0.0040,
    sector="Commodity"
)

print("Assets defined successfully")
```

### Step 4: Add Assets to Portfolio

```python
# Add assets with target allocation
# 60% stocks, 30% bonds, 10% alternatives
portfolio.add_asset(spy, weight=0.40)   # 40% US stocks
portfolio.add_asset(vxus, weight=0.20)  # 20% International stocks
portfolio.add_asset(agg, weight=0.30)   # 30% Bonds
portfolio.add_asset(gld, weight=0.10)   # 10% Gold

# Verify allocation
allocation = portfolio.get_allocation()
print("\nPortfolio Allocation:")
for symbol, weight in allocation.items():
    print(f"  {symbol}: {weight:.1%}")

# Check that weights sum to 100%
total_weight = sum(allocation.values())
print(f"\nTotal allocation: {total_weight:.1%}")
```

### Step 5: Calculate Portfolio Metrics

```python
# Get current dollar values
values = portfolio.get_current_values()
print("\nCurrent Position Values:")
for symbol, value in values.items():
    print(f"  {symbol}: ${value:,.0f}")

# Calculate portfolio metrics
# Note: This requires historical data
try:
    metrics = portfolio.calculate_metrics()
    print("\nPortfolio Metrics:")
    print(f"  Expected Return: {metrics['expected_return']:.2%}")
    print(f"  Volatility: {metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
except Exception as e:
    print(f"\nNote: Full metrics require market data: {e}")
```

### Step 6: Visualize Allocation

```python
from allocation_station.visualization import plot_allocation_pie

# Create pie chart of allocation
fig = plot_allocation_pie(
    allocation=portfolio.get_allocation(),
    title=f"{portfolio.name} - Asset Allocation"
)

# Save or display
fig.write_html("portfolio_allocation.html")
print("\nAllocation chart saved to portfolio_allocation.html")
```

### Key Takeaways

- Portfolios are collections of assets with specified weights
- Weights must sum to 1.0 (100%)
- Asset metadata (expense ratios, sectors) helps with analysis
- Portfolio metrics depend on historical market data

---

## Tutorial 2: Running Monte Carlo Simulations

**Goal**: Project future portfolio outcomes using Monte Carlo simulation.

**Duration**: 20 minutes

### Step 1: Set Up Portfolio (from Tutorial 1)

```python
from allocation_station import Portfolio, Asset, AssetClass
from allocation_station.portfolio import StrategicAllocation
from allocation_station.simulation import MonteCarloSimulator, SimulationConfig

# Create portfolio (reuse from Tutorial 1)
portfolio = Portfolio(name="Retirement Portfolio", initial_value=500000)
portfolio.add_asset(Asset(symbol="SPY"), weight=0.40)
portfolio.add_asset(Asset(symbol="VXUS"), weight=0.20)
portfolio.add_asset(Asset(symbol="AGG"), weight=0.30)
portfolio.add_asset(Asset(symbol="GLD"), weight=0.10)
```

### Step 2: Define Allocation Strategy

```python
# Create a buy-and-hold strategy with quarterly rebalancing
strategy = StrategicAllocation(
    name="Quarterly Rebalance",
    target_allocation={
        "SPY": 0.40,
        "VXUS": 0.20,
        "AGG": 0.30,
        "GLD": 0.10
    },
    rebalance_frequency="quarterly",
    rebalance_threshold=0.05  # Rebalance if drift exceeds 5%
)

print(f"Strategy: {strategy.name}")
print(f"Rebalance frequency: {strategy.rebalance_frequency}")
```

### Step 3: Configure Simulation Parameters

```python
# Set up simulation configuration
config = SimulationConfig(
    n_simulations=10000,      # Run 10,000 scenarios
    time_horizon=30,          # 30-year projection
    time_steps=252,           # Daily time steps (252 trading days/year)

    # Expected annual returns (historical averages)
    expected_returns={
        "SPY": 0.10,   # 10% for US stocks
        "VXUS": 0.09,  # 9% for international stocks
        "AGG": 0.04,   # 4% for bonds
        "GLD": 0.05    # 5% for gold
    },

    # Annual volatility (standard deviation)
    volatilities={
        "SPY": 0.18,   # 18% volatility
        "VXUS": 0.20,  # 20% volatility
        "AGG": 0.05,   # 5% volatility
        "GLD": 0.16    # 16% volatility
    },

    inflation_rate=0.025,     # 2.5% annual inflation
    random_seed=42,           # For reproducibility
    n_jobs=-1                 # Use all CPU cores
)

print(f"\nSimulation setup: {config.n_simulations:,} simulations")
print(f"Time horizon: {config.time_horizon} years")
```

### Step 4: Run the Simulation

```python
# Create simulator and run
simulator = MonteCarloSimulator(config)

print("\nRunning Monte Carlo simulation...")
print("This may take a few minutes...")

results = simulator.simulate(
    portfolio=portfolio,
    strategy=strategy
)

print("Simulation complete!")
```

### Step 5: Analyze Results

```python
# Summary statistics
print("\n" + "="*60)
print("SIMULATION RESULTS")
print("="*60)

print(f"\nSuccess Rate: {results.success_rate:.1%}")
print(f"  (Probability portfolio lasts 30 years)")

print(f"\nFinal Portfolio Values:")
print(f"  Median: ${results.median_final_value:,.0f}")
print(f"  Mean: ${results.mean_final_value:,.0f}")
print(f"  10th Percentile: ${results.percentile(10):,.0f}")
print(f"  25th Percentile: ${results.percentile(25):,.0f}")
print(f"  75th Percentile: ${results.percentile(75):,.0f}")
print(f"  90th Percentile: ${results.percentile(90):,.0f}")

print(f"\nRisk Metrics:")
print(f"  Probability of Ruin: {results.ruin_probability:.1%}")
print(f"  Worst Case (5th %ile): ${results.percentile(5):,.0f}")

# Get full summary
summary = results.summary()
print(f"\nAdditional Metrics:")
print(f"  Std Dev of Final Value: ${summary['std_final_value']:,.0f}")
print(f"  Coefficient of Variation: {summary['cv_final_value']:.2f}")
```

### Step 6: Visualize Results

```python
from allocation_station.visualization import plot_monte_carlo_paths

# Plot simulation paths
fig = plot_monte_carlo_paths(
    simulation_results=results,
    n_paths=100,  # Show 100 random paths
    show_percentiles=[10, 25, 50, 75, 90],  # Show key percentiles
    title="30-Year Monte Carlo Simulation"
)

fig.write_html("monte_carlo_results.html")
print("\nVisualization saved to monte_carlo_results.html")
```

### Step 7: Sensitivity Analysis

```python
# Test different scenarios
scenarios = {
    "Base Case": {"SPY": 0.10, "VXUS": 0.09, "AGG": 0.04, "GLD": 0.05},
    "Bull Market": {"SPY": 0.12, "VXUS": 0.11, "AGG": 0.05, "GLD": 0.06},
    "Bear Market": {"SPY": 0.07, "VXUS": 0.06, "AGG": 0.03, "GLD": 0.04},
}

print("\nSensitivity Analysis:")
print("-" * 60)

for scenario_name, returns in scenarios.items():
    config_scenario = SimulationConfig(
        n_simulations=5000,
        time_horizon=30,
        expected_returns=returns,
        volatilities=config.volatilities,
        random_seed=42
    )

    simulator_scenario = MonteCarloSimulator(config_scenario)
    results_scenario = simulator_scenario.simulate(portfolio, strategy)

    print(f"\n{scenario_name}:")
    print(f"  Success Rate: {results_scenario.success_rate:.1%}")
    print(f"  Median Final: ${results_scenario.median_final_value:,.0f}")
```

### Key Takeaways

- Monte Carlo simulations project multiple possible futures
- More simulations (10,000+) provide more reliable results
- Use realistic return and volatility assumptions
- Always run sensitivity analysis on key assumptions
- Percentiles help understand the range of outcomes

---

## Tutorial 3: Backtesting Strategies

**Goal**: Test allocation strategies using historical market data.

**Duration**: 25 minutes

### Step 1: Set Up Backtest

```python
from allocation_station.backtesting import BacktestEngine, BacktestConfig
from allocation_station.portfolio import StrategicAllocation
from datetime import datetime

# Create backtest configuration
config = BacktestConfig(
    start_date=datetime(2010, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=100000,
    transaction_cost=0.001,  # 0.1% per trade
    slippage=0.0005,         # 0.05% slippage
    rebalance_frequency="quarterly",
    benchmark_symbol="SPY"   # Compare to S&P 500
)

print(f"Backtest Period: {config.start_date.date()} to {config.end_date.date()}")
print(f"Initial Capital: ${config.initial_capital:,.0f}")
```

### Step 2: Define Strategy to Test

```python
# Test a 60/40 portfolio
strategy_60_40 = StrategicAllocation(
    name="Classic 60/40",
    target_allocation={
        "SPY": 0.60,  # 60% stocks
        "AGG": 0.40   # 40% bonds
    },
    rebalance_frequency="quarterly"
)

# Create portfolio
portfolio = Portfolio(
    name="60/40 Portfolio",
    initial_value=config.initial_capital
)
portfolio.add_asset(Asset(symbol="SPY"), weight=0.60)
portfolio.add_asset(Asset(symbol="AGG"), weight=0.40)
```

### Step 3: Run Backtest

```python
# Initialize backtest engine
engine = BacktestEngine(config)

print("\nRunning backtest...")
results = engine.run(portfolio, strategy_60_40)
print("Backtest complete!")
```

### Step 4: Analyze Performance

```python
# Performance metrics
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)

print(f"\nTotal Return: {results.total_return:.2%}")
print(f"CAGR: {results.cagr:.2%}")
print(f"Total Years: {(config.end_date - config.start_date).days / 365.25:.1f}")

print(f"\nRisk Metrics:")
print(f"  Annual Volatility: {results.volatility:.2%}")
print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"  Sortino Ratio: {results.sortino_ratio:.2f}")
print(f"  Calmar Ratio: {results.calmar_ratio:.2f}")
print(f"  Max Drawdown: {results.max_drawdown:.2%}")

print(f"\nTrading Stats:")
print(f"  Win Rate: {results.win_rate:.1%}")
print(f"  Best Month: {results.best_month:.2%}")
print(f"  Worst Month: {results.worst_month:.2%}")
```

### Step 5: Compare to Benchmark

```python
# Get detailed metrics including alpha/beta
metrics = results.get_metrics()

print(f"\nBenchmark Comparison (vs {config.benchmark_symbol}):")
print(f"  Benchmark Return: {results.benchmark_return:.2%}")
print(f"  Excess Return: {results.excess_return:.2%}")
print(f"  Alpha: {metrics['alpha']:.2%}")
print(f"  Beta: {metrics['beta']:.2f}")
print(f"  Information Ratio: {metrics['information_ratio']:.2f}")
print(f"  Tracking Error: {results.tracking_error:.2%}")
```

### Step 6: Analyze Transaction Costs

```python
# Transaction cost analysis
cost_analysis = results.get_transaction_costs()

print(f"\nTransaction Cost Analysis:")
print(f"  Total Costs: ${cost_analysis['total_costs']:,.2f}")
print(f"  Cost as % of Final Value: {cost_analysis['cost_percentage']:.2%}")
print(f"  Number of Rebalances: {cost_analysis['n_rebalances']}")
print(f"  Average Cost per Rebalance: ${cost_analysis['avg_cost']:,.2f}")
print(f"  Cost Impact on Returns: {cost_analysis['return_impact']:.2%}")
```

### Step 7: Compare Multiple Strategies

```python
# Test additional strategies
strategies_to_test = {
    "80/20 Aggressive": StrategicAllocation(
        name="80/20",
        target_allocation={"SPY": 0.80, "AGG": 0.20},
        rebalance_frequency="quarterly"
    ),
    "40/60 Conservative": StrategicAllocation(
        name="40/60",
        target_allocation={"SPY": 0.40, "AGG": 0.60},
        rebalance_frequency="quarterly"
    ),
    "Equal Weight": StrategicAllocation(
        name="Equal Weight",
        target_allocation={"SPY": 0.50, "AGG": 0.50},
        rebalance_frequency="quarterly"
    ),
}

print("\n" + "="*60)
print("STRATEGY COMPARISON")
print("="*60)

comparison_results = {}

for strat_name, strategy in strategies_to_test.items():
    # Create portfolio for each strategy
    port = Portfolio(name=strat_name, initial_value=config.initial_capital)
    for symbol, weight in strategy.target_allocation.items():
        port.add_asset(Asset(symbol=symbol), weight=weight)

    # Run backtest
    result = engine.run(port, strategy)
    comparison_results[strat_name] = result

# Display comparison table
print(f"\n{'Strategy':<20} {'CAGR':<10} {'Volatility':<12} {'Sharpe':<10} {'Max DD':<10}")
print("-" * 70)

for strat_name, result in comparison_results.items():
    print(f"{strat_name:<20} {result.cagr:>8.2%} {result.volatility:>10.2%} {result.sharpe_ratio:>8.2f} {result.max_drawdown:>8.2%}")
```

### Step 8: Visualize Results

```python
from allocation_station.visualization import (
    plot_portfolio_performance,
    plot_drawdown
)

# Plot performance comparison
fig = plot_portfolio_performance(
    portfolio_returns=results.returns,
    benchmark_returns=results.benchmark_returns,
    title="60/40 Portfolio vs S&P 500"
)
fig.write_html("backtest_performance.html")

# Plot drawdowns
fig_dd = plot_drawdown(
    returns=results.returns,
    title="Portfolio Drawdown Analysis"
)
fig_dd.write_html("backtest_drawdown.html")

print("\nVisualizations saved!")
```

### Key Takeaways

- Backtesting validates strategies with real historical data
- Always include transaction costs and slippage
- Compare to relevant benchmarks
- Consider multiple time periods and market conditions
- Past performance doesn't guarantee future results

---

## Tutorial 4: Portfolio Optimization

**Goal**: Find optimal portfolio allocations using Modern Portfolio Theory.

**Duration**: 30 minutes

### Step 1: Define Investment Universe

```python
from allocation_station.analysis import EfficientFrontier
from allocation_station.data import MarketDataProvider
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define assets
assets = ["SPY", "TLT", "GLD", "VNQ", "DBC"]
asset_names = {
    "SPY": "US Stocks",
    "TLT": "Long-Term Bonds",
    "GLD": "Gold",
    "VNQ": "Real Estate",
    "DBC": "Commodities"
}

print("Investment Universe:")
for symbol, name in asset_names.items():
    print(f"  {symbol}: {name}")
```

### Step 2: Fetch Historical Data

```python
# Get 5 years of historical data
provider = MarketDataProvider(source="yahoo")

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print(f"\nFetching data from {start_date.date()} to {end_date.date()}...")

data = provider.get_historical_data(
    symbols=assets,
    start_date=start_date,
    end_date=end_date,
    frequency="daily"
)

print(f"Retrieved {len(data)} days of data")
```

### Step 3: Calculate Expected Returns and Risk

```python
# Calculate daily returns
returns = data['Close'].pct_change().dropna()

# Annualize returns (252 trading days)
expected_returns = returns.mean() * 252
volatilities = returns.std() * np.sqrt(252)

# Calculate covariance matrix
cov_matrix = returns.cov() * 252

print("\nAsset Statistics:")
print(f"{'Asset':<10} {'Return':<12} {'Volatility':<12}")
print("-" * 40)
for symbol in assets:
    print(f"{symbol:<10} {expected_returns[symbol]:>10.2%} {volatilities[symbol]:>10.2%}")
```

### Step 4: Create Efficient Frontier

```python
# Create efficient frontier
ef = EfficientFrontier(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    risk_free_rate=0.04  # 4% risk-free rate
)

print("\nCalculating efficient frontier...")
```

### Step 5: Find Optimal Portfolios

```python
# 1. Maximum Sharpe Ratio (optimal risk-return tradeoff)
max_sharpe = ef.optimize("max_sharpe")

print("\n" + "="*60)
print("MAXIMUM SHARPE RATIO PORTFOLIO")
print("="*60)
print("\nWeights:")
for symbol, weight in max_sharpe.weights.items():
    if weight > 0.01:  # Only show meaningful allocations
        print(f"  {asset_names[symbol]:<20} {weight:>6.1%}")

print(f"\nExpected Return: {max_sharpe.expected_return:.2%}")
print(f"Volatility: {max_sharpe.volatility:.2%}")
print(f"Sharpe Ratio: {max_sharpe.sharpe_ratio:.2f}")

# 2. Minimum Variance (lowest risk)
min_var = ef.optimize("min_variance")

print("\n" + "="*60)
print("MINIMUM VARIANCE PORTFOLIO")
print("="*60)
print("\nWeights:")
for symbol, weight in min_var.weights.items():
    if weight > 0.01:
        print(f"  {asset_names[symbol]:<20} {weight:>6.1%}")

print(f"\nExpected Return: {min_var.expected_return:.2%}")
print(f"Volatility: {min_var.volatility:.2%}")
print(f"Sharpe Ratio: {min_var.sharpe_ratio:.2f}")

# 3. Risk Parity (equal risk contribution)
risk_parity = ef.optimize("risk_parity")

print("\n" + "="*60)
print("RISK PARITY PORTFOLIO")
print("="*60)
print("\nWeights:")
for symbol, weight in risk_parity.weights.items():
    if weight > 0.01:
        print(f"  {asset_names[symbol]:<20} {weight:>6.1%}")

print(f"\nExpected Return: {risk_parity.expected_return:.2%}")
print(f"Volatility: {risk_parity.volatility:.2%}")
print(f"Sharpe Ratio: {risk_parity.sharpe_ratio:.2f}")
```

### Step 6: Target Return/Risk Optimization

```python
# Find portfolio with target 8% return
target_return_portfolio = ef.optimize(
    "target_return",
    target_return=0.08
)

print("\n" + "="*60)
print("TARGET 8% RETURN PORTFOLIO")
print("="*60)
print("\nWeights:")
for symbol, weight in target_return_portfolio.weights.items():
    if weight > 0.01:
        print(f"  {asset_names[symbol]:<20} {weight:>6.1%}")

print(f"\nExpected Return: {target_return_portfolio.expected_return:.2%}")
print(f"Volatility: {target_return_portfolio.volatility:.2%}")
print(f"Sharpe Ratio: {target_return_portfolio.sharpe_ratio:.2f}")

# Find portfolio with target 10% volatility
target_risk_portfolio = ef.optimize(
    "target_volatility",
    target_volatility=0.10
)

print("\n" + "="*60)
print("TARGET 10% VOLATILITY PORTFOLIO")
print("="*60)
print("\nWeights:")
for symbol, weight in target_risk_portfolio.weights.items():
    if weight > 0.01:
        print(f"  {asset_names[symbol]:<20} {weight:>6.1%}")

print(f"\nExpected Return: {target_risk_portfolio.expected_return:.2%}")
print(f"Volatility: {target_risk_portfolio.volatility:.2%}")
print(f"Sharpe Ratio: {target_risk_portfolio.sharpe_ratio:.2f}")
```

### Step 7: Visualize Efficient Frontier

```python
from allocation_station.visualization import plot_efficient_frontier

# Plot efficient frontier with optimal portfolios
fig = plot_efficient_frontier(
    efficient_frontier=ef,
    portfolios={
        "Max Sharpe": max_sharpe.weights,
        "Min Variance": min_var.weights,
        "Risk Parity": risk_parity.weights,
        "Equal Weight": {s: 0.2 for s in assets}
    },
    show_assets=True
)

fig.write_html("efficient_frontier.html")
print("\nEfficient frontier saved to efficient_frontier.html")
```

### Step 8: Add Constraints

```python
from allocation_station.optimization import ConstrainedOptimizer

# Define constraints
constraints = {
    "min_weights": {"SPY": 0.20, "TLT": 0.10},  # Minimums
    "max_weights": {"GLD": 0.15, "DBC": 0.10},  # Maximums
    "asset_class_limits": {
        "equity": {"min": 0.40, "max": 0.70}
    }
}

# Optimize with constraints
constrained_optimizer = ConstrainedOptimizer(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    constraints=constraints
)

constrained_portfolio = constrained_optimizer.optimize("max_sharpe")

print("\n" + "="*60)
print("CONSTRAINED OPTIMIZATION (MAX SHARPE)")
print("="*60)
print("\nConstraints:")
print("  • Minimum 20% US Stocks")
print("  • Minimum 10% Bonds")
print("  • Maximum 15% Gold")
print("  • Maximum 10% Commodities")
print("  • Total equity: 40-70%")

print("\nWeights:")
for symbol, weight in constrained_portfolio.weights.items():
    if weight > 0.01:
        print(f"  {asset_names[symbol]:<20} {weight:>6.1%}")
```

### Key Takeaways

- Efficient frontier shows optimal risk-return combinations
- Max Sharpe ratio portfolio is typically most attractive
- Constraints reflect real-world investment restrictions
- Historical data drives optimization (garbage in, garbage out)
- Consider multiple optimization objectives

---

## Tutorial 5: Retirement Planning

**Goal**: Plan retirement withdrawals and test portfolio sustainability.

**Duration**: 25 minutes

### Step 1: Define Retirement Scenario

```python
from allocation_station import Portfolio, Asset
from allocation_station.portfolio import WithdrawalStrategy, WithdrawalMethod
from allocation_station.simulation import MonteCarloSimulator, SimulationConfig
from allocation_station.withdrawal import SocialSecurityOptimizer

# Retirement parameters
current_age = 55
retirement_age = 65
life_expectancy = 95
current_savings = 1000000
annual_expenses = 60000

print("Retirement Scenario:")
print(f"  Current Age: {current_age}")
print(f"  Retirement Age: {retirement_age}")
print(f"  Life Expectancy: {life_expectancy}")
print(f"  Current Savings: ${current_savings:,.0f}")
print(f"  Annual Expenses: ${annual_expenses:,.0f}")
```

### Step 2: Create Retirement Portfolio

```python
# Age-appropriate allocation (55 years old)
# Rule of thumb: (age) in bonds = 55% bonds, 45% stocks

portfolio = Portfolio(
    name="Retirement Portfolio",
    initial_value=current_savings
)

# Diversified allocation
portfolio.add_asset(Asset(symbol="VTI", name="Total US Stock"), weight=0.35)
portfolio.add_asset(Asset(symbol="VXUS", name="International Stock"), weight=0.10)
portfolio.add_asset(Asset(symbol="BND", name="Total Bond"), weight=0.45)
portfolio.add_asset(Asset(symbol="VNQ", name="REITs"), weight=0.05)
portfolio.add_asset(Asset(symbol="GLD", name="Gold"), weight=0.05)

print("\nPortfolio Allocation:")
for symbol, weight in portfolio.get_allocation().items():
    print(f"  {symbol}: {weight:.0%}")
```

### Step 3: Test 4% Rule

```python
# Classic 4% rule
withdrawal_4pct = WithdrawalStrategy(
    name="4% Rule",
    method=WithdrawalMethod.FOUR_PERCENT_RULE,
    initial_withdrawal_rate=0.04,
    inflation_adjustment=True
)

# First year withdrawal
initial_withdrawal = current_savings * 0.04
print(f"\n4% Rule:")
print(f"  Initial Withdrawal: ${initial_withdrawal:,.0f}/year")
print(f"  Monthly Income: ${initial_withdrawal/12:,.0f}")

# Simulate
config = SimulationConfig(
    n_simulations=10000,
    time_horizon=retirement_age - current_age + life_expectancy - retirement_age,
    expected_returns={
        "VTI": 0.09, "VXUS": 0.08, "BND": 0.04,
        "VNQ": 0.07, "GLD": 0.05
    },
    volatilities={
        "VTI": 0.18, "VXUS": 0.20, "BND": 0.05,
        "VNQ": 0.22, "GLD": 0.16
    },
    inflation_rate=0.025,
    random_seed=42
)

simulator = MonteCarloSimulator(config)
strategy = StrategicAllocation(
    name="Fixed Allocation",
    target_allocation=portfolio.get_allocation(),
    rebalance_frequency="annual"
)

print("\nRunning 4% Rule simulation...")
results_4pct = simulator.simulate(portfolio, strategy, withdrawal_4pct)

print(f"\nResults:")
print(f"  Success Rate: {results_4pct.success_rate:.1%}")
print(f"  Median Final Value: ${results_4pct.median_final_value:,.0f}")
print(f"  10th Percentile: ${results_4pct.percentile(10):,.0f}")
```

### Step 4: Test Guyton-Klinger Strategy

```python
# Guyton-Klinger with guardrails
withdrawal_gk = WithdrawalStrategy(
    name="Guyton-Klinger",
    method=WithdrawalMethod.GUYTON_KLINGER,
    initial_withdrawal_rate=0.05,  # Start with 5%
    upper_guardrail=0.20,  # Increase if portfolio up 20%
    lower_guardrail=0.15,  # Decrease if portfolio down 15%
    inflation_adjustment=True
)

print("\nGuyton-Klinger Strategy:")
print(f"  Initial Withdrawal Rate: 5.0%")
print(f"  Initial Amount: ${current_savings * 0.05:,.0f}")
print(f"  Guardrails: +20% / -15%")

print("\nRunning Guyton-Klinger simulation...")
results_gk = simulator.simulate(portfolio, strategy, withdrawal_gk)

print(f"\nResults:")
print(f"  Success Rate: {results_gk.success_rate:.1%}")
print(f"  Median Final Value: ${results_gk.median_final_value:,.0f}")
print(f"  Average Withdrawal: ${results_gk.average_withdrawal:,.0f}")
```

### Step 5: Variable Percentage Withdrawal

```python
# Variable percentage (recalculate each year)
withdrawal_vp = WithdrawalStrategy(
    name="Variable Percentage",
    method=WithdrawalMethod.VARIABLE_PERCENTAGE,
    withdrawal_percentage=0.045,  # 4.5% of current balance
    floor=40000,  # Minimum $40k/year
    ceiling=100000  # Maximum $100k/year
)

print("\nVariable Percentage Strategy:")
print(f"  Withdrawal Rate: 4.5% of current balance")
print(f"  Floor: $40,000")
print(f"  Ceiling: $100,000")

results_vp = simulator.simulate(portfolio, strategy, withdrawal_vp)

print(f"\nResults:")
print(f"  Success Rate: {results_vp.success_rate:.1%}")
print(f"  Median Final Value: ${results_vp.median_final_value:,.0f}")
```

### Step 6: Compare Strategies

```python
print("\n" + "="*70)
print("WITHDRAWAL STRATEGY COMPARISON")
print("="*70)

comparison = {
    "4% Rule": results_4pct,
    "Guyton-Klinger": results_gk,
    "Variable Percentage": results_vp
}

print(f"\n{'Strategy':<25} {'Success':<12} {'Median Final':<18} {'10th %ile':<15}")
print("-" * 70)

for name, result in comparison.items():
    print(f"{name:<25} {result.success_rate:>10.1%} "
          f"${result.median_final_value:>15,.0f} "
          f"${result.percentile(10):>13,.0f}")
```

### Step 7: Social Security Integration

```python
# Optimize Social Security claiming
ss_optimizer = SocialSecurityOptimizer(
    birth_date="1970-06-15",
    full_retirement_benefit=2800,  # Monthly benefit at FRA (67)
    spousal_benefit=1400,  # Spouse's benefit
    spouse_birth_date="1972-03-20"
)

optimal_strategy = ss_optimizer.optimize(
    portfolio=portfolio,
    life_expectancy=life_expectancy,
    discount_rate=0.03
)

print("\n" + "="*60)
print("SOCIAL SECURITY OPTIMIZATION")
print("="*60)

print(f"\nOptimal Claiming Ages:")
print(f"  You: {optimal_strategy['your_age']}")
print(f"  Spouse: {optimal_strategy['spouse_age']}")

print(f"\nExpected Benefits:")
print(f"  Your Monthly Benefit: ${optimal_strategy['your_benefit']:,.0f}")
print(f"  Spouse Monthly Benefit: ${optimal_strategy['spouse_benefit']:,.0f}")
print(f"  Total Monthly: ${optimal_strategy['total_monthly']:,.0f}")

print(f"\nLifetime Value:")
print(f"  Total Lifetime Benefits: ${optimal_strategy['lifetime_value']:,.0f}")
print(f"  Break-even Age: {optimal_strategy['breakeven_age']}")
```

### Step 8: Create Retirement Plan

```python
# Combine everything into comprehensive plan
print("\n" + "="*70)
print("COMPREHENSIVE RETIREMENT PLAN")
print("="*70)

print(f"\nPhase 1: Pre-Retirement ({current_age}-{retirement_age})")
print(f"  • Continue saving ${(retirement_age - current_age) * 25000:,.0f}")
print(f"  • Maintain current allocation")
print(f"  • Target savings at retirement: ${current_savings * 1.5:,.0f}")

print(f"\nPhase 2: Early Retirement ({retirement_age}-{retirement_age + 10})")
print(f"  • Allocation: 45% stocks / 55% bonds")
print(f"  • Withdrawal: 4% rule or Guyton-Klinger")
print(f"  • Delay Social Security to age {optimal_strategy['your_age']}")
print(f"  • Annual withdrawals: ~$60,000")

print(f"\nPhase 3: Late Retirement ({retirement_age + 10}+)")
print(f"  • Allocation: 35% stocks / 65% bonds")
print(f"  • Social Security: ${optimal_strategy['total_monthly']:,.0f}/month")
print(f"  • Portfolio withdrawals: ${annual_expenses - optimal_strategy['total_monthly'] * 12:,.0f}/year")

print(f"\nSuccess Probability: {results_gk.success_rate:.1%}")
print(f"Recommended Strategy: Guyton-Klinger with SS at {optimal_strategy['your_age']}")
```

### Key Takeaways

- Multiple withdrawal strategies have different tradeoffs
- Guardrails (Guyton-Klinger) provide flexibility
- Social Security timing significantly impacts outcomes
- Success rate should be 85%+ for comfortable retirement
- Adjust allocation as you age (glide path)

---

## Tutorial 6: Risk Analysis

**Goal**: Perform comprehensive risk analysis on a portfolio.

**Duration**: 20 minutes

### Step 1: Create Portfolio for Analysis

```python
from allocation_station import Portfolio, Asset
from allocation_station.analysis import (
    RiskAnalyzer, StressTester, TailRiskAnalyzer, CorrelationAnalyzer
)

# Create aggressive growth portfolio
portfolio = Portfolio(name="Growth Portfolio", initial_value=100000)
portfolio.add_asset(Asset(symbol="QQQ", name="Nasdaq 100"), weight=0.50)
portfolio.add_asset(Asset(symbol="SPY", name="S&P 500"), weight=0.30)
portfolio.add_asset(Asset(symbol="IWM", name="Russell 2000"), weight=0.15)
portfolio.add_asset(Asset(symbol="AGG", name="Bonds"), weight=0.05)

print("Portfolio:", portfolio.name)
for symbol, weight in portfolio.get_allocation().items():
    print(f"  {symbol}: {weight:.0%}")
```

### Step 2: Basic Risk Metrics

```python
analyzer = RiskAnalyzer(portfolio)

# Value at Risk
var_95 = analyzer.calculate_var(confidence_level=0.95, time_horizon=1)
var_99 = analyzer.calculate_var(confidence_level=0.99, time_horizon=1)

print("\nValue at Risk (1-day):")
print(f"  95% VaR: ${var_95:,.0f}")
print(f"  99% VaR: ${var_99:,.0f}")
print(f"  Interpretation: 95% confident daily loss won't exceed ${abs(var_95):,.0f}")

# Conditional VaR (Expected Shortfall)
cvar_95 = analyzer.calculate_cvar(confidence_level=0.95)
cvar_99 = analyzer.calculate_cvar(confidence_level=0.99)

print("\nConditional VaR (Expected Shortfall):")
print(f"  95% CVaR: ${cvar_95:,.0f}")
print(f"  99% CVaR: ${cvar_99:,.0f}")
print(f"  Interpretation: Average loss in worst 5% of days is ${abs(cvar_95):,.0f}")
```

### Step 3: Drawdown Analysis

```python
# Maximum drawdown
max_dd = analyzer.calculate_max_drawdown()
dd_stats = analyzer.drawdown_statistics()

print("\nDrawdown Analysis:")
print(f"  Maximum Drawdown: {max_dd:.2%}")
print(f"  Average Drawdown: {dd_stats['avg_drawdown']:.2%}")
print(f"  Drawdown Duration:")
print(f"    • Average: {dd_stats['avg_duration']} days")
print(f"    • Longest: {dd_stats['max_duration']} days")
print(f"  Recovery Time:")
print(f"    • Average: {dd_stats['avg_recovery_time']} days")
print(f"    • Longest: {dd_stats['max_recovery_time']} days")
```

### Step 4: Stress Testing

```python
tester = StressTester(portfolio)

# Define historical stress scenarios
scenarios = {
    "2008 Financial Crisis": {
        "QQQ": -0.42, "SPY": -0.37, "IWM": -0.34, "AGG": 0.05
    },
    "2020 COVID Crash": {
        "QQQ": -0.27, "SPY": -0.34, "IWM": -0.42, "AGG": 0.03
    },
    "2000 Dot-Com Bubble": {
        "QQQ": -0.83, "SPY": -0.49, "IWM": -0.30, "AGG": 0.12
    },
    "1987 Black Monday": {
        "QQQ": -0.30, "SPY": -0.20, "IWM": -0.25, "AGG": 0.02
    },
    "Rising Rates Scenario": {
        "QQQ": -0.15, "SPY": -0.10, "IWM": -0.12, "AGG": -0.08
    }
}

print("\n" + "="*70)
print("STRESS TEST RESULTS")
print("="*70)

results = tester.run_scenarios(scenarios)

print(f"\n{'Scenario':<30} {'Portfolio Loss':<18} {'Recovery Time':<15}")
print("-" * 70)

for scenario, outcome in results.items():
    print(f"{scenario:<30} {outcome['portfolio_loss']:>16.2%} "
          f"{outcome['recovery_time']:>12} months")

# Worst case analysis
worst_scenario = min(results.items(), key=lambda x: x[1]['portfolio_loss'])
print(f"\nWorst Case Scenario: {worst_scenario[0]}")
print(f"  Portfolio Loss: {worst_scenario[1]['portfolio_loss']:.2%}")
print(f"  Dollar Loss: ${portfolio.value * abs(worst_scenario[1]['portfolio_loss']):,.0f}")
```

### Step 5: Tail Risk Analysis

```python
tail_analyzer = TailRiskAnalyzer(portfolio)

tail_metrics = tail_analyzer.analyze_tails()

print("\n" + "="*60)
print("TAIL RISK ANALYSIS")
print("="*60)

print(f"\nLeft Tail (Downside Risk):")
print(f"  5th Percentile: {tail_metrics['left_tail_5']:.2%}")
print(f"  1st Percentile: {tail_metrics['left_tail_1']:.2%}")

print(f"\nRight Tail (Upside):")
print(f"  95th Percentile: {tail_metrics['right_tail_95']:.2%}")
print(f"  99th Percentile: {tail_metrics['right_tail_99']:.2%}")

print(f"\nDistribution Shape:")
print(f"  Skewness: {tail_metrics['skewness']:.2f}")
if tail_metrics['skewness'] < 0:
    print(f"    → Negative skew: More downside tail risk")
else:
    print(f"    → Positive skew: More upside potential")

print(f"  Kurtosis: {tail_metrics['kurtosis']:.2f}")
if tail_metrics['kurtosis'] > 3:
    print(f"    → Fat tails: Higher probability of extreme events")

print(f"\n  Tail Ratio (95th/5th): {tail_metrics['tail_ratio']:.2f}")
```

### Step 6: Correlation Breakdown Analysis

```python
corr_analyzer = CorrelationAnalyzer(portfolio)

# Check correlation stability
breakdown_risk = corr_analyzer.check_correlation_breakdown()

print("\n" + "="*60)
print("CORRELATION BREAKDOWN ANALYSIS")
print("="*60)

print(f"\nCorrelation Stability:")
print(f"  Risk Score: {breakdown_risk['risk_score']:.2f} / 10")
if breakdown_risk['risk_score'] > 7:
    print(f"  Status: ⚠️ HIGH RISK - Correlations may converge in crisis")
elif breakdown_risk['risk_score'] > 4:
    print(f"  Status: ⚠️ MODERATE RISK - Some correlation instability")
else:
    print(f"  Status: ✓ LOW RISK - Correlations relatively stable")

print(f"\nNormal Market Correlations:")
print(f"  Average Pairwise Correlation: {breakdown_risk['stable_corr']:.2f}")

print(f"\nCrisis Period Correlations:")
print(f"  Average Pairwise Correlation: {breakdown_risk['crisis_corr']:.2f}")
print(f"  Change: {breakdown_risk['crisis_corr'] - breakdown_risk['stable_corr']:+.2f}")

if breakdown_risk['crisis_corr'] > 0.8:
    print(f"\n⚠️ Warning: Assets highly correlated in crisis (diversification breaks down)")
```

### Step 7: Comprehensive Risk Report

```python
print("\n" + "="*70)
print("COMPREHENSIVE RISK REPORT")
print("="*70)

print(f"\nPortfolio: {portfolio.name}")
print(f"Value: ${portfolio.value:,.0f}")

print(f"\nRisk Metrics Summary:")
print(f"  • 1-Day 95% VaR: ${abs(var_95):,.0f}")
print(f"  • 1-Day 99% CVaR: ${abs(cvar_99):,.0f}")
print(f"  • Maximum Drawdown: {max_dd:.2%}")
print(f"  • Tail Risk Score: {tail_metrics['kurtosis']:.1f}")
print(f"  • Correlation Risk: {breakdown_risk['risk_score']:.1f}/10")

print(f"\nWorst Case Scenarios:")
print(f"  • 2008-style Crisis Loss: {worst_scenario[1]['portfolio_loss']:.2%}")
print(f"  • Dollar Amount: ${portfolio.value * abs(worst_scenario[1]['portfolio_loss']):,.0f}")

print(f"\nRisk Recommendations:")
if max_dd < -0.30:
    print(f"  • ⚠️ High drawdown risk - consider more bonds")
if breakdown_risk['risk_score'] > 7:
    print(f"  • ⚠️ Add uncorrelated assets (gold, managed futures)")
if tail_metrics['kurtosis'] > 5:
    print(f"  • ⚠️ Fat tails detected - size positions accordingly")

print(f"\n✓ Risk analysis complete")
```

### Key Takeaways

- VaR and CVaR quantify downside risk
- Stress testing shows performance in extreme scenarios
- Tail risk analysis reveals distribution characteristics
- Correlations can break down in crises
- Multiple risk metrics provide comprehensive view

---

*Continue with Tutorials 7-10 in similar detailed format...*

---

## Summary

These tutorials cover the essential workflows for using Allocation Station:

1. **Portfolio Construction**: Build and manage multi-asset portfolios
2. **Monte Carlo Simulation**: Project future outcomes under uncertainty
3. **Backtesting**: Validate strategies with historical data
4. **Optimization**: Find optimal allocations using MPT
5. **Retirement Planning**: Plan sustainable withdrawals
6. **Risk Analysis**: Comprehensive risk assessment

### Next Steps

- Review the [User Guide](USER_GUIDE.md) for comprehensive documentation
- Check the [API Reference](API_REFERENCE.md) for detailed function documentation
- Explore the [Cookbook](COOKBOOK.md) for quick code recipes
- Read the [Theoretical Background](THEORY.md) for mathematical foundations

### Getting Help

- GitHub Issues: Report bugs and request features
- GitHub Discussions: Ask questions and share ideas
- Examples folder: Review example scripts
- Documentation: [docs.allocation-station.dev](https://docs.allocation-station.dev)

---

**Last Updated**: January 2025
**Version**: 0.1.0
