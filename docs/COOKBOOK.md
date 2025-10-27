# Allocation Station Cookbook

Quick code recipes for common portfolio management tasks.

## Table of Contents

1. [Portfolio Construction Recipes](#portfolio-construction-recipes)
2. [Data Management Recipes](#data-management-recipes)
3. [Analysis Recipes](#analysis-recipes)
4. [Optimization Recipes](#optimization-recipes)
5. [Simulation Recipes](#simulation-recipes)
6. [Visualization Recipes](#visualization-recipes)
7. [Risk Management Recipes](#risk-management-recipes)
8. [Integration Recipes](#integration-recipes)

---

## Portfolio Construction Recipes

### Recipe 1: Create Classic 60/40 Portfolio

```python
from allocation_station import Portfolio, Asset, AssetClass

portfolio = Portfolio(name="60/40 Portfolio", initial_value=100000)
portfolio.add_asset(Asset(symbol="SPY", asset_class=AssetClass.ETF), weight=0.6)
portfolio.add_asset(Asset(symbol="AGG", asset_class=AssetClass.ETF), weight=0.4)
```

### Recipe 2: Three-Fund Lazy Portfolio

```python
# Bogleheads three-fund portfolio
portfolio = Portfolio(name="Three-Fund Portfolio", initial_value=100000)
portfolio.add_asset(Asset(symbol="VTI", name="Total US Stock"), weight=0.60)
portfolio.add_asset(Asset(symbol="VXUS", name="Total International"), weight=0.20)
portfolio.add_asset(Asset(symbol="BND", name="Total Bond"), weight=0.20)
```

### Recipe 3: All-Weather Portfolio (Ray Dalio)

```python
# Risk parity inspired allocation
portfolio = Portfolio(name="All-Weather Portfolio", initial_value=100000)
portfolio.add_asset(Asset(symbol="SPY", name="US Stocks"), weight=0.30)
portfolio.add_asset(Asset(symbol="TLT", name="Long-Term Bonds"), weight=0.40)
portfolio.add_asset(Asset(symbol="IEF", name="Intermediate Bonds"), weight=0.15)
portfolio.add_asset(Asset(symbol="DBC", name="Commodities"), weight=0.075)
portfolio.add_asset(Asset(symbol="GLD", name="Gold"), weight=0.075)
```

### Recipe 4: Permanent Portfolio (Harry Browne)

```python
# Equal weight across asset classes
portfolio = Portfolio(name="Permanent Portfolio", initial_value=100000)
portfolio.add_asset(Asset(symbol="VTI", name="Stocks"), weight=0.25)
portfolio.add_asset(Asset(symbol="TLT", name="Long Bonds"), weight=0.25)
portfolio.add_asset(Asset(symbol="GLD", name="Gold"), weight=0.25)
portfolio.add_asset(Asset(symbol="BIL", name="Cash/T-Bills"), weight=0.25)
```

### Recipe 5: Aggressive Growth Portfolio

```python
# High equity concentration for growth
portfolio = Portfolio(name="Aggressive Growth", initial_value=100000)
portfolio.add_asset(Asset(symbol="QQQ", name="Nasdaq 100"), weight=0.40)
portfolio.add_asset(Asset(symbol="VUG", name="Growth Stocks"), weight=0.30)
portfolio.add_asset(Asset(symbol="VWO", name="Emerging Markets"), weight=0.20)
portfolio.add_asset(Asset(symbol="BND", name="Bonds"), weight=0.10)
```

### Recipe 6: Conservative Income Portfolio

```python
# Focus on income and capital preservation
portfolio = Portfolio(name="Conservative Income", initial_value=100000)
portfolio.add_asset(Asset(symbol="BND", name="Total Bond"), weight=0.40)
portfolio.add_asset(Asset(symbol="VYMI", name="High Div Yield"), weight=0.20)
portfolio.add_asset(Asset(symbol="VNQ", name="REITs"), weight=0.15)
portfolio.add_asset(Asset(symbol="VTI", name="US Stocks"), weight=0.15)
portfolio.add_asset(Asset(symbol="SHY", name="Short-Term Bonds"), weight=0.10)
```

### Recipe 7: Factor-Tilted Portfolio

```python
# Tilt toward value and small-cap factors
portfolio = Portfolio(name="Factor Portfolio", initial_value=100000)
portfolio.add_asset(Asset(symbol="VTV", name="Value"), weight=0.30)
portfolio.add_asset(Asset(symbol="VBR", name="Small-Cap Value"), weight=0.20)
portfolio.add_asset(Asset(symbol="MTUM", name="Momentum"), weight=0.15)
portfolio.add_asset(Asset(symbol="QUAL", name="Quality"), weight=0.15)
portfolio.add_asset(Asset(symbol="AGG", name="Bonds"), weight=0.20)
```

### Recipe 8: Global Diversification

```python
# True global diversification
portfolio = Portfolio(name="Global Portfolio", initial_value=100000)
portfolio.add_asset(Asset(symbol="VTI", name="US Stocks"), weight=0.30)
portfolio.add_asset(Asset(symbol="VXUS", name="International"), weight=0.25)
portfolio.add_asset(Asset(symbol="VWO", name="Emerging Markets"), weight=0.10)
portfolio.add_asset(Asset(symbol="BND", name="US Bonds"), weight=0.20)
portfolio.add_asset(Asset(symbol="BNDX", name="Intl Bonds"), weight=0.10)
portfolio.add_asset(Asset(symbol="VNQ", name="US REITs"), weight=0.03)
portfolio.add_asset(Asset(symbol="VNQI", name="Intl REITs"), weight=0.02)
```

---

## Data Management Recipes

### Recipe 9: Fetch Historical Data

```python
from allocation_station.data import MarketDataProvider
from datetime import datetime, timedelta

provider = MarketDataProvider(source="yahoo")

# Get 5 years of data
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

data = provider.get_historical_data(
    symbols=["SPY", "AGG", "GLD"],
    start_date=start_date,
    end_date=end_date,
    frequency="daily"
)
```

### Recipe 10: Cache Data Locally

```python
provider = MarketDataProvider(
    source="yahoo",
    cache_dir="./data/cache",
    cache_enabled=True
)

# Data will be cached for faster access
data = provider.get_historical_data(["SPY"], start_date="2020-01-01")
```

### Recipe 11: Use Multiple Data Sources

```python
from allocation_station.data import (
    YahooFinanceProvider,
    AlphaVantageProvider,
    FREDDataProvider
)

# Primary source
yahoo = YahooFinanceProvider()
stock_data = yahoo.get_data(["SPY", "AGG"])

# Alpha Vantage for fundamentals
av = AlphaVantageProvider(api_key="YOUR_KEY")
fundamentals = av.get_fundamentals("AAPL")

# FRED for economic data
fred = FREDDataProvider(api_key="YOUR_KEY")
gdp = fred.get_series("GDP")
inflation = fred.get_series("CPIAUCSL")
```

### Recipe 12: Export Portfolio Data

```python
from allocation_station.data import DataExporter

exporter = DataExporter(portfolio)

# Export to Excel
exporter.to_excel("portfolio_data.xlsx")

# Export to CSV
exporter.to_csv("portfolio_data.csv")

# Export to Parquet
exporter.to_parquet("portfolio_data.parquet")
```

---

## Analysis Recipes

### Recipe 13: Calculate Basic Statistics

```python
import pandas as pd
import numpy as np

# Calculate returns
returns = data['Close'].pct_change().dropna()

# Annual statistics
annual_return = returns.mean() * 252
annual_vol = returns.std() * np.sqrt(252)
sharpe_ratio = (annual_return - 0.04) / annual_vol  # Assuming 4% risk-free

print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {annual_vol:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
```

### Recipe 14: Calculate Correlation Matrix

```python
# Calculate correlation between assets
returns = data['Close'].pct_change().dropna()
correlation_matrix = returns.corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Visualize
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Asset Correlation Matrix")
plt.show()
```

### Recipe 15: Calculate Rolling Sharpe Ratio

```python
# 1-year rolling Sharpe ratio
window = 252  # 1 year
risk_free_rate = 0.04

rolling_return = returns.rolling(window).mean() * 252
rolling_vol = returns.rolling(window).std() * np.sqrt(252)
rolling_sharpe = (rolling_return - risk_free_rate) / rolling_vol

# Plot
rolling_sharpe.plot(title="1-Year Rolling Sharpe Ratio")
plt.show()
```

### Recipe 16: Identify Drawdown Periods

```python
# Calculate drawdowns
cumulative = (1 + returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max

# Find significant drawdowns (>10%)
significant_dd = drawdown[drawdown < -0.10]

print(f"Number of significant drawdowns: {len(significant_dd)}")
print(f"Worst drawdown: {drawdown.min():.2%}")
```

### Recipe 17: Performance Attribution

```python
from allocation_station.analysis import PerformanceAttribution

attributor = PerformanceAttribution(portfolio)

# Brinson attribution
attribution = attributor.brinson_attribution(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    portfolio_weights=portfolio.get_allocation()
)

print("Performance Attribution:")
print(f"  Allocation Effect: {attribution['allocation_effect']:.2%}")
print(f"  Selection Effect: {attribution['selection_effect']:.2%}")
print(f"  Interaction Effect: {attribution['interaction_effect']:.2%}")
```

---

## Optimization Recipes

### Recipe 18: Maximum Sharpe Ratio Portfolio

```python
from allocation_station.analysis import EfficientFrontier

ef = EfficientFrontier(expected_returns, cov_matrix, risk_free_rate=0.04)
max_sharpe = ef.optimize("max_sharpe")

print("Maximum Sharpe Ratio Portfolio:")
for asset, weight in max_sharpe.weights.items():
    if weight > 0.01:
        print(f"  {asset}: {weight:.1%}")
```

### Recipe 19: Minimum Variance Portfolio

```python
min_var = ef.optimize("min_variance")

print("Minimum Variance Portfolio:")
for asset, weight in min_var.weights.items():
    if weight > 0.01:
        print(f"  {asset}: {weight:.1%}")
print(f"Expected Volatility: {min_var.volatility:.2%}")
```

### Recipe 20: Target Return Optimization

```python
# Find minimum risk portfolio with 8% target return
target_portfolio = ef.optimize("target_return", target_return=0.08)

print("Portfolio with 8% Target Return:")
for asset, weight in target_portfolio.weights.items():
    if weight > 0.01:
        print(f"  {asset}: {weight:.1%}")
print(f"Expected Volatility: {target_portfolio.volatility:.2%}")
```

### Recipe 21: Risk Parity Allocation

```python
risk_parity = ef.optimize("risk_parity")

print("Risk Parity Portfolio:")
for asset, weight in risk_parity.weights.items():
    if weight > 0.01:
        print(f"  {asset}: {weight:.1%}")

# Verify equal risk contribution
risk_contributions = ef.get_risk_contributions(risk_parity.weights)
print("\nRisk Contributions:")
for asset, contrib in risk_contributions.items():
    print(f"  {asset}: {contrib:.2%}")
```

### Recipe 22: Black-Litterman Optimization

```python
from allocation_station.optimization import BlackLittermanOptimizer

# Market equilibrium
market_weights = {"SPY": 0.6, "AGG": 0.4}

# Your views
views = [
    {"asset": "SPY", "view": 0.12, "confidence": 0.7},  # Bullish on stocks
]

bl = BlackLittermanOptimizer(
    market_weights=market_weights,
    risk_aversion=2.5,
    cov_matrix=cov_matrix
)

optimal = bl.optimize(views)
print("Black-Litterman Optimal Weights:")
for asset, weight in optimal.items():
    print(f"  {asset}: {weight:.1%}")
```

### Recipe 23: Constrained Optimization

```python
from allocation_station.optimization import ConstrainedOptimizer

constraints = {
    "min_weights": {"SPY": 0.20},  # At least 20% in SPY
    "max_weights": {"GLD": 0.15},  # At most 15% in gold
}

optimizer = ConstrainedOptimizer(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    constraints=constraints
)

constrained_portfolio = optimizer.optimize("max_sharpe")
```

---

## Simulation Recipes

### Recipe 24: Quick Monte Carlo Simulation

```python
from allocation_station.simulation import MonteCarloSimulator, SimulationConfig

config = SimulationConfig(
    n_simulations=10000,
    time_horizon=30,
    expected_returns={"SPY": 0.08, "AGG": 0.04},
    volatilities={"SPY": 0.16, "AGG": 0.05},
    random_seed=42
)

simulator = MonteCarloSimulator(config)
results = simulator.simulate(portfolio, strategy)

print(f"Success Rate: {results.success_rate:.1%}")
print(f"Median Final Value: ${results.median_final_value:,.0f}")
```

### Recipe 25: Parallel Monte Carlo for Speed

```python
config = SimulationConfig(
    n_simulations=100000,  # Large number
    time_horizon=30,
    n_jobs=-1  # Use all CPU cores
)

simulator = MonteCarloSimulator(config)
results = simulator.simulate(portfolio, strategy)
```

### Recipe 26: Regime-Switching Simulation

```python
from allocation_station.simulation import RegimeSwitchingSimulator

simulator = RegimeSwitchingSimulator(
    regimes={
        "bull": {"returns": 0.12, "volatility": 0.12},
        "bear": {"returns": -0.05, "volatility": 0.25},
        "normal": {"returns": 0.08, "volatility": 0.16}
    },
    transition_matrix=[
        [0.7, 0.2, 0.1],  # From bull
        [0.3, 0.4, 0.3],  # From bear
        [0.3, 0.2, 0.5]   # From normal
    ]
)

results = simulator.simulate(portfolio, n_simulations=10000, time_horizon=30)
```

### Recipe 27: GARCH Volatility Simulation

```python
from allocation_station.simulation import GARCHSimulator

simulator = GARCHSimulator(
    omega=0.00001,
    alpha=0.1,
    beta=0.85,
    mean_return=0.08
)

results = simulator.simulate(portfolio, n_simulations=5000, time_horizon=30)
```

### Recipe 28: Bootstrap Historical Returns

```python
from allocation_station.simulation import BootstrapSimulator

# Use actual historical returns distribution
simulator = BootstrapSimulator(
    historical_returns=returns,
    block_size=21  # 1-month blocks
)

results = simulator.simulate(portfolio, n_simulations=10000, time_horizon=30)
```

---

## Visualization Recipes

### Recipe 29: Portfolio Allocation Pie Chart

```python
from allocation_station.visualization import plot_allocation_pie

fig = plot_allocation_pie(
    allocation=portfolio.get_allocation(),
    title="Portfolio Allocation"
)
fig.show()
```

### Recipe 30: Performance Line Chart

```python
from allocation_station.visualization import plot_portfolio_performance

fig = plot_portfolio_performance(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    title="Portfolio vs Benchmark"
)
fig.show()
```

### Recipe 31: Efficient Frontier with Portfolios

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

### Recipe 32: Monte Carlo Simulation Paths

```python
from allocation_station.visualization import plot_monte_carlo_paths

fig = plot_monte_carlo_paths(
    simulation_results=results,
    n_paths=100,
    show_percentiles=[10, 25, 50, 75, 90],
    title="30-Year Portfolio Projections"
)
fig.show()
```

### Recipe 33: Drawdown Chart

```python
from allocation_station.visualization import plot_drawdown

fig = plot_drawdown(
    returns=returns,
    title="Portfolio Drawdown Analysis"
)
fig.show()
```

### Recipe 34: Correlation Heatmap

```python
from allocation_station.visualization import plot_correlation_heatmap

fig = plot_correlation_heatmap(
    correlation_matrix=returns.corr(),
    title="Asset Correlation Matrix"
)
fig.show()
```

### Recipe 35: Rolling Metrics Dashboard

```python
from allocation_station.visualization import plot_rolling_metrics

fig = plot_rolling_metrics(
    returns=returns,
    window=252,  # 1 year
    metrics=["return", "volatility", "sharpe", "max_drawdown"]
)
fig.show()
```

---

## Risk Management Recipes

### Recipe 36: Calculate VaR and CVaR

```python
from allocation_station.analysis import RiskAnalyzer

analyzer = RiskAnalyzer(portfolio)

var_95 = analyzer.calculate_var(confidence_level=0.95, time_horizon=1)
cvar_95 = analyzer.calculate_cvar(confidence_level=0.95)

print(f"1-Day 95% VaR: ${var_95:,.0f}")
print(f"1-Day 95% CVaR: ${cvar_95:,.0f}")
```

### Recipe 37: Run Stress Tests

```python
from allocation_station.analysis import StressTester

tester = StressTester(portfolio)

scenarios = {
    "2008 Crisis": {"SPY": -0.37, "AGG": 0.05},
    "COVID Crash": {"SPY": -0.34, "AGG": 0.03}
}

results = tester.run_scenarios(scenarios)

for scenario, outcome in results.items():
    print(f"{scenario}: {outcome['portfolio_loss']:.2%} loss")
```

### Recipe 38: Check Portfolio Diversification

```python
from allocation_station.analysis import DiversificationAnalyzer

analyzer = DiversificationAnalyzer(portfolio)

metrics = analyzer.calculate_diversification_metrics()

print(f"Number of Holdings: {metrics['n_holdings']}")
print(f"Effective N: {metrics['effective_n']:.1f}")
print(f"HHI Index: {metrics['hhi']:.3f}")
print(f"Diversification Ratio: {metrics['diversification_ratio']:.2f}")
```

### Recipe 39: Rebalancing Alert

```python
# Check if rebalancing is needed
current_allocation = portfolio.get_current_allocation()
target_allocation = portfolio.get_target_allocation()

threshold = 0.05  # 5% drift threshold

needs_rebalance = False
for asset, target in target_allocation.items():
    current = current_allocation.get(asset, 0)
    drift = abs(current - target)

    if drift > threshold:
        needs_rebalance = True
        print(f"⚠️ {asset}: {drift:.1%} drift (Current: {current:.1%}, Target: {target:.1%})")

if needs_rebalance:
    print("\n🔄 Rebalancing recommended")
else:
    print("\n✓ Portfolio within threshold")
```

### Recipe 40: Position Sizing with Kelly Criterion

```python
from allocation_station.portfolio import KellyCriterion

kelly = KellyCriterion(
    win_rate=0.55,  # 55% win rate
    avg_win=0.08,   # 8% average win
    avg_loss=0.05   # 5% average loss
)

optimal_size = kelly.calculate_position_size()
print(f"Optimal position size: {optimal_size:.1%}")

# Half-Kelly for more conservative approach
print(f"Half-Kelly position size: {optimal_size/2:.1%}")
```

---

## Integration Recipes

### Recipe 41: Connect to Interactive Brokers

```python
from allocation_station.integrations import IBKRConnector

# Connect to IBKR
ibkr = IBKRConnector(
    host="127.0.0.1",
    port=7497,  # TWS paper trading port
    client_id=1
)

ibkr.connect()

# Get current positions
positions = ibkr.get_positions()
for pos in positions:
    print(f"{pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_cost']:.2f}")
```

### Recipe 42: Import Portfolio from Broker

```python
from allocation_station.integrations import PortfolioImporter

importer = PortfolioImporter()

# Import from CSV (exported from broker)
portfolio = importer.from_csv("broker_positions.csv")

# Or import from IBKR directly
portfolio = importer.from_ibkr(ibkr_connection)

print(f"Imported portfolio: {portfolio.name}")
print(f"Total value: ${portfolio.value:,.0f}")
```

### Recipe 43: Setup Automated Rebalancing

```python
from allocation_station.automation import AutoRebalancer

rebalancer = AutoRebalancer(
    portfolio=portfolio,
    strategy=strategy,
    broker=ibkr,
    schedule="quarterly",  # or "monthly", "annual"
    threshold=0.05  # 5% drift threshold
)

# Start automated rebalancing
rebalancer.start()
print("Automated rebalancing enabled")
```

### Recipe 44: Send Portfolio Report via Email

```python
from allocation_station.ui import ReportGenerator
from allocation_station.integrations import EmailSender

# Generate PDF report
generator = ReportGenerator(portfolio)
report_pdf = generator.create_pdf_report(
    include_performance=True,
    include_risk_metrics=True,
    include_charts=True
)

# Send via email
sender = EmailSender(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="your_email@gmail.com",
    password="your_password"
)

sender.send_report(
    to="client@example.com",
    subject="Monthly Portfolio Report",
    report_file=report_pdf
)
```

### Recipe 45: Setup Webhook Notifications

```python
from allocation_station.integrations import WebhookNotifier

notifier = WebhookNotifier(webhook_url="https://hooks.slack.com/services/...")

# Notify on rebalancing
@portfolio.on_rebalance
def notify_rebalance(event):
    notifier.send({
        "text": f"Portfolio rebalanced: {event['n_trades']} trades executed"
    })

# Notify on threshold breach
@portfolio.on_threshold_breach
def notify_breach(event):
    notifier.send({
        "text": f"⚠️ Alert: {event['asset']} breached {event['threshold']:.1%} threshold"
    })
```

### Recipe 46: Store Results in Database

```python
from allocation_station.integrations import DatabaseConnector

db = DatabaseConnector(
    connection_string="postgresql://user:pass@localhost/portfolio_db"
)

# Store portfolio snapshot
db.save_portfolio_snapshot(portfolio)

# Store simulation results
db.save_simulation_results(results)

# Store backtest results
db.save_backtest_results(backtest_results)

# Query historical data
history = db.get_portfolio_history(portfolio.name, start_date="2024-01-01")
```

### Recipe 47: REST API Server

```python
from allocation_station.integrations import create_api_server

# Create REST API
app = create_api_server(portfolio_manager)

# Run server
app.run(host="0.0.0.0", port=8000)

# API Endpoints:
# GET  /portfolios
# GET  /portfolios/{id}
# POST /portfolios/{id}/optimize
# POST /portfolios/{id}/simulate
# POST /portfolios/{id}/backtest
```

### Recipe 48: Export to R

```python
from allocation_station.export import RExporter

exporter = RExporter(portfolio)

# Create R-compatible data structures
exporter.export_to_rdata("portfolio.RData")

# Generate R script
exporter.generate_r_script(
    "analyze_portfolio.R",
    include_plots=True,
    include_optimization=True
)
```

### Recipe 49: Launch Interactive Dashboard

```python
from allocation_station.ui import create_dashboard

# Create Dash application
app = create_dashboard(
    portfolio=portfolio,
    simulation_results=sim_results,
    backtest_results=backtest_results,
    enable_optimization=True,
    enable_rebalancing=True
)

# Run dashboard
app.run_server(debug=False, host="0.0.0.0", port=8050)
print("Dashboard running at http://localhost:8050")
```

### Recipe 50: Cloud Storage Backup

```python
from allocation_station.integrations import CloudStorage

# AWS S3
s3 = CloudStorage(
    provider="s3",
    bucket="my-portfolio-backups",
    access_key="YOUR_ACCESS_KEY",
    secret_key="YOUR_SECRET_KEY"
)

# Backup portfolio data
s3.upload_portfolio(portfolio, key="portfolios/retirement.json")

# Backup simulation results
s3.upload_results(results, key="simulations/retirement_2024.json")

# Restore from backup
restored_portfolio = s3.download_portfolio(key="portfolios/retirement.json")
```

---

## Quick Reference

### Common Patterns

**Create and Populate Portfolio**
```python
portfolio = Portfolio(name="My Portfolio", initial_value=100000)
portfolio.add_asset(Asset(symbol="SPY"), weight=0.6)
portfolio.add_asset(Asset(symbol="AGG"), weight=0.4)
```

**Fetch Data**
```python
provider = MarketDataProvider()
data = provider.get_historical_data(["SPY", "AGG"], start_date="2020-01-01")
```

**Optimize**
```python
ef = EfficientFrontier(expected_returns, cov_matrix)
optimal = ef.optimize("max_sharpe")
```

**Simulate**
```python
config = SimulationConfig(n_simulations=10000, time_horizon=30)
simulator = MonteCarloSimulator(config)
results = simulator.simulate(portfolio, strategy)
```

**Backtest**
```python
config = BacktestConfig(start_date="2010-01-01", end_date="2024-01-01")
engine = BacktestEngine(config)
results = engine.run(portfolio, strategy)
```

**Visualize**
```python
from allocation_station.visualization import plot_*
fig = plot_portfolio_performance(returns)
fig.show()
```

---

## Tips and Tricks

1. **Use caching** for data to speed up repeated analysis
2. **Parallel processing** (`n_jobs=-1`) for large simulations
3. **Start with fewer simulations** (1000) for testing, scale up for production
4. **Always set random_seed** for reproducible results
5. **Include transaction costs** for realistic backtesting
6. **Compare to benchmarks** to validate performance
7. **Test multiple time periods** to check robustness
8. **Use constraints** to reflect real-world limitations
9. **Monitor correlations** as they can change in crises
10. **Regular rebalancing** maintains target allocation

---

**Last Updated**: January 2025
**Version**: 0.1.0
