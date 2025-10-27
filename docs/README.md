# Allocation Station Documentation

Complete documentation for the allocation-station portfolio management and analysis framework.

## Documentation Overview

This documentation provides everything you need to use Allocation Station effectively, from basic concepts to advanced features.

### 📚 Main Documentation

| Document | Description | Best For |
|----------|-------------|----------|
| [**User Guide**](USER_GUIDE.md) | Comprehensive guide covering all features | First-time users, reference |
| [**API Reference**](API_REFERENCE.md) | Complete API documentation | Developers, detailed function info |
| [**Tutorials**](TUTORIALS.md) | Step-by-step walkthroughs | Learning by doing |
| [**Cookbook**](COOKBOOK.md) | Quick code recipes | Copy-paste solutions |
| [**Theory**](THEORY.md) | Mathematical foundations | Understanding the math |

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/allocation-station.git
cd allocation-station
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### Your First Portfolio

```python
from allocation_station import Portfolio, Asset
from allocation_station.portfolio import StrategicAllocation
from allocation_station.simulation import MonteCarloSimulator, SimulationConfig

# Create portfolio
portfolio = Portfolio(name="My Portfolio", initial_value=100000)
portfolio.add_asset(Asset(symbol="SPY"), weight=0.6)
portfolio.add_asset(Asset(symbol="AGG"), weight=0.4)

# Define strategy
strategy = StrategicAllocation(
    name="60/40",
    target_allocation={"SPY": 0.6, "AGG": 0.4},
    rebalance_frequency="quarterly"
)

# Run simulation
config = SimulationConfig(n_simulations=10000, time_horizon=30)
simulator = MonteCarloSimulator(config)
results = simulator.simulate(portfolio, strategy)

print(f"Success Rate: {results.success_rate:.1%}")
```

---

## Documentation Guide

### For Beginners

Start here if you're new to portfolio management or Allocation Station:

1. **[User Guide - Introduction](USER_GUIDE.md#introduction)** - Understand what Allocation Station does
2. **[User Guide - Installation](USER_GUIDE.md#installation)** - Set up your environment
3. **[User Guide - Quick Start](USER_GUIDE.md#quick-start)** - Your first portfolio
4. **[Tutorial 1](TUTORIALS.md#tutorial-1-building-your-first-portfolio)** - Build a portfolio step-by-step
5. **[Cookbook - Portfolio Recipes](COOKBOOK.md#portfolio-construction-recipes)** - Common portfolio examples

### For Intermediate Users

Already familiar with the basics? Level up your skills:

1. **[Tutorial 2 - Monte Carlo](TUTORIALS.md#tutorial-2-running-monte-carlo-simulations)** - Project future outcomes
2. **[Tutorial 3 - Backtesting](TUTORIALS.md#tutorial-3-backtesting-strategies)** - Test with historical data
3. **[Tutorial 4 - Optimization](TUTORIALS.md#tutorial-4-portfolio-optimization)** - Find optimal allocations
4. **[User Guide - Risk Analysis](USER_GUIDE.md#risk-analysis)** - Comprehensive risk assessment
5. **[Cookbook - Analysis Recipes](COOKBOOK.md#analysis-recipes)** - Quick analysis patterns

### For Advanced Users

Push the boundaries with advanced features:

1. **[Tutorial 9 - Multi-Asset Portfolios](TUTORIALS.md#tutorial-9-multi-asset-portfolio-with-alternatives)** - Complex portfolios
2. **[Tutorial 10 - ML Integration](TUTORIALS.md#tutorial-10-machine-learning-integration)** - Machine learning models
3. **[User Guide - Advanced Features](USER_GUIDE.md#advanced-features)** - Tax harvesting, leverage, factors
4. **[API Reference](API_REFERENCE.md)** - Complete API documentation
5. **[Theory](THEORY.md)** - Mathematical foundations

### For Developers

Building on top of Allocation Station or contributing:

1. **[API Reference](API_REFERENCE.md)** - Complete API documentation
2. **[DEVELOPMENT_GUIDE.md](../DEVELOPMENT_GUIDE.md)** - Development roadmap and architecture
3. **[TESTING.md](../TESTING.md)** - Testing infrastructure
4. **[Theory - Model Details](THEORY.md)** - Mathematical models and algorithms
5. **Contributing Guide** (coming soon)

---

## Feature Matrix

### Core Features

| Feature | User Guide | Tutorial | Cookbook | API Ref |
|---------|------------|----------|----------|---------|
| Portfolio Construction | [Link](USER_GUIDE.md#working-with-assets-and-portfolios) | [Tutorial 1](TUTORIALS.md#tutorial-1-building-your-first-portfolio) | [Recipes 1-8](COOKBOOK.md#portfolio-construction-recipes) | [Core Module](API_REFERENCE.md#core-module) |
| Monte Carlo Simulation | [Link](USER_GUIDE.md#monte-carlo-simulations) | [Tutorial 2](TUTORIALS.md#tutorial-2-running-monte-carlo-simulations) | [Recipes 24-28](COOKBOOK.md#simulation-recipes) | [Simulation Module](API_REFERENCE.md#simulation-module) |
| Backtesting | [Link](USER_GUIDE.md#backtesting) | [Tutorial 3](TUTORIALS.md#tutorial-3-backtesting-strategies) | - | [Backtesting Module](API_REFERENCE.md#backtesting-module) |
| Optimization | [Link](USER_GUIDE.md#portfolio-optimization) | [Tutorial 4](TUTORIALS.md#tutorial-4-portfolio-optimization) | [Recipes 18-23](COOKBOOK.md#optimization-recipes) | [Optimization Module](API_REFERENCE.md#optimization-module) |
| Risk Analysis | [Link](USER_GUIDE.md#risk-analysis) | [Tutorial 6](TUTORIALS.md#tutorial-6-risk-analysis) | [Recipes 36-40](COOKBOOK.md#risk-management-recipes) | [Analysis Module](API_REFERENCE.md#analysis-module) |
| Withdrawal Strategies | [Link](USER_GUIDE.md#withdrawal-strategies) | [Tutorial 5](TUTORIALS.md#tutorial-5-retirement-planning) | - | [Withdrawal Module](API_REFERENCE.md#withdrawal-module) |
| Visualization | [Link](USER_GUIDE.md#visualization) | All tutorials | [Recipes 29-35](COOKBOOK.md#visualization-recipes) | [Visualization Module](API_REFERENCE.md#visualization-module) |

### Advanced Features

| Feature | Documentation | Status |
|---------|--------------|--------|
| Tax-Loss Harvesting | [User Guide](USER_GUIDE.md#tax-loss-harvesting) | ✅ Available |
| Multi-Currency | [User Guide](USER_GUIDE.md#multi-currency-portfolios) | ✅ Available |
| Factor Models | [User Guide](USER_GUIDE.md#factor-analysis), [Theory](THEORY.md#factor-models) | ✅ Available |
| Machine Learning | [Tutorial 10](TUTORIALS.md#tutorial-10-machine-learning-integration) | ✅ Available |
| Broker Integration | [Cookbook Recipes 41-43](COOKBOOK.md#integration-recipes) | ✅ Available |
| Interactive Dashboard | [User Guide](USER_GUIDE.md#interactive-dashboard), [Tutorial 8](TUTORIALS.md#tutorial-8-building-interactive-dashboards) | ✅ Available |

---

## Common Tasks

### Quick Links to Common Tasks

#### Portfolio Management
- [Create a portfolio](USER_GUIDE.md#creating-assets)
- [Add assets](USER_GUIDE.md#adding-assets)
- [Rebalance portfolio](COOKBOOK.md#recipe-39-rebalancing-alert)
- [Track performance](COOKBOOK.md#recipe-13-calculate-basic-statistics)

#### Analysis
- [Calculate risk metrics](COOKBOOK.md#recipe-36-calculate-var-and-cvar)
- [Run stress tests](COOKBOOK.md#recipe-37-run-stress-tests)
- [Performance attribution](COOKBOOK.md#recipe-17-performance-attribution)
- [Correlation analysis](COOKBOOK.md#recipe-14-calculate-correlation-matrix)

#### Optimization
- [Find optimal allocation](COOKBOOK.md#recipe-18-maximum-sharpe-ratio-portfolio)
- [Minimize risk](COOKBOOK.md#recipe-19-minimum-variance-portfolio)
- [Risk parity](COOKBOOK.md#recipe-21-risk-parity-allocation)
- [With constraints](COOKBOOK.md#recipe-23-constrained-optimization)

#### Simulation & Testing
- [Monte Carlo simulation](COOKBOOK.md#recipe-24-quick-monte-carlo-simulation)
- [Backtest strategy](TUTORIALS.md#tutorial-3-backtesting-strategies)
- [Retirement planning](TUTORIALS.md#tutorial-5-retirement-planning)
- [Scenario analysis](TUTORIALS.md#tutorial-2-running-monte-carlo-simulations)

#### Visualization
- [Allocation pie chart](COOKBOOK.md#recipe-29-portfolio-allocation-pie-chart)
- [Performance chart](COOKBOOK.md#recipe-30-performance-line-chart)
- [Efficient frontier](COOKBOOK.md#recipe-31-efficient-frontier-with-portfolios)
- [Interactive dashboard](COOKBOOK.md#recipe-49-launch-interactive-dashboard)

---

## Concepts and Theory

### Core Concepts

| Concept | User Guide | Theory |
|---------|------------|--------|
| Modern Portfolio Theory | [Link](USER_GUIDE.md#core-concepts) | [Link](THEORY.md#modern-portfolio-theory) |
| Efficient Frontier | [Link](USER_GUIDE.md#efficient-frontier) | [Link](THEORY.md#efficient-frontier) |
| Risk Metrics | [Link](USER_GUIDE.md#risk-analysis) | [Link](THEORY.md#risk-metrics) |
| Sharpe Ratio | [Link](USER_GUIDE.md#sharpe-ratio) | [Link](THEORY.md#sharpe-ratio) |
| Diversification | [Link](USER_GUIDE.md#diversification) | [Link](THEORY.md#diversification-benefit) |

### Advanced Concepts

| Concept | Theory | Tutorial |
|---------|--------|----------|
| GARCH Models | [Link](THEORY.md#garch-models) | [Tutorial 2](TUTORIALS.md#garch-volatility-model) |
| Regime Switching | [Link](THEORY.md#regime-switching-models) | [Tutorial 2](TUTORIALS.md#regime-switching-model) |
| Factor Models | [Link](THEORY.md#factor-models) | - |
| Black-Litterman | [Link](THEORY.md#black-litterman-model) | [Tutorial 4](TUTORIALS.md#black-litterman-optimization) |
| Dynamic Programming | [Link](THEORY.md#dynamic-programming) | [Tutorial 5](TUTORIALS.md#dynamic-programming-optimization) |

---

## Examples

### Portfolio Examples

Browse example portfolios in the [Cookbook](COOKBOOK.md#portfolio-construction-recipes):

- **Recipe 1**: Classic 60/40 Portfolio
- **Recipe 3**: All-Weather Portfolio (Ray Dalio)
- **Recipe 4**: Permanent Portfolio (Harry Browne)
- **Recipe 5**: Aggressive Growth
- **Recipe 6**: Conservative Income
- **Recipe 7**: Factor-Tilted
- **Recipe 8**: Global Diversification

### Code Examples

Complete code examples in the [`examples/`](../examples/) directory:

- `basic_portfolio_example.py` - Basic portfolio management
- `advanced_monte_carlo_example.py` - Advanced simulations
- `performance_attribution_example.py` - Attribution analysis
- `dashboard_example.py` - Interactive dashboard
- And many more...

---

## Troubleshooting

### Common Issues

| Issue | Solution | Reference |
|-------|----------|-----------|
| Installation errors | Check Python version (3.9+) | [Installation](USER_GUIDE.md#installation) |
| Data download fails | Check internet, try alternative source | [Troubleshooting](USER_GUIDE.md#troubleshooting) |
| Memory issues | Use batch processing | [Performance Tips](USER_GUIDE.md#performance-tips) |
| Optimization not converging | Relax constraints, try different solver | [Troubleshooting](USER_GUIDE.md#optimization-not-converging) |

### Getting Help

- **Documentation**: Start with the [User Guide](USER_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/allocation-station/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/allocation-station/discussions)
- **Examples**: Check the [`examples/`](../examples/) folder

---

## Contributing

We welcome contributions! See:

- [Development Guide](../DEVELOPMENT_GUIDE.md) - Architecture and roadmap
- [Testing Guide](../TESTING.md) - Testing infrastructure
- Contributing Guide (coming soon)

---

## API Reference Index

### Quick Access to Modules

- [Core Module](API_REFERENCE.md#core-module) - Assets, Portfolios, Allocations
- [Portfolio Module](API_REFERENCE.md#portfolio-module) - Strategies, Withdrawals
- [Simulation Module](API_REFERENCE.md#simulation-module) - Monte Carlo, Scenarios
- [Backtesting Module](API_REFERENCE.md#backtesting-module) - Historical testing
- [Analysis Module](API_REFERENCE.md#analysis-module) - Risk, Performance
- [Optimization Module](API_REFERENCE.md#optimization-module) - MPT, Black-Litterman
- [Visualization Module](API_REFERENCE.md#visualization-module) - Charts, Dashboards
- [Data Module](API_REFERENCE.md#data-module) - Market data
- [ML Module](API_REFERENCE.md#ml-module) - Machine learning
- [Integration Module](API_REFERENCE.md#integration-module) - Brokers, APIs

---

## Learning Paths

### Path 1: Individual Investor

Focus on retirement planning and portfolio management.

1. [Quick Start](USER_GUIDE.md#quick-start)
2. [Tutorial 1: First Portfolio](TUTORIALS.md#tutorial-1-building-your-first-portfolio)
3. [Tutorial 2: Monte Carlo](TUTORIALS.md#tutorial-2-running-monte-carlo-simulations)
4. [Tutorial 5: Retirement Planning](TUTORIALS.md#tutorial-5-retirement-planning)
5. [Cookbook: Portfolio Recipes](COOKBOOK.md#portfolio-construction-recipes)

### Path 2: Financial Advisor

Focus on client portfolio optimization and reporting.

1. [Tutorial 1: First Portfolio](TUTORIALS.md#tutorial-1-building-your-first-portfolio)
2. [Tutorial 4: Optimization](TUTORIALS.md#tutorial-4-portfolio-optimization)
3. [Tutorial 6: Risk Analysis](TUTORIALS.md#tutorial-6-risk-analysis)
4. [Tutorial 8: Dashboards](TUTORIALS.md#tutorial-8-building-interactive-dashboards)
5. [User Guide: Reporting](USER_GUIDE.md#reporting-system)

### Path 3: Quantitative Analyst

Focus on advanced models and backtesting.

1. [Tutorial 3: Backtesting](TUTORIALS.md#tutorial-3-backtesting-strategies)
2. [Tutorial 4: Optimization](TUTORIALS.md#tutorial-4-portfolio-optimization)
3. [Tutorial 10: ML Integration](TUTORIALS.md#tutorial-10-machine-learning-integration)
4. [Theory Document](THEORY.md)
5. [API Reference](API_REFERENCE.md)

### Path 4: Developer

Focus on extending and customizing.

1. [API Reference](API_REFERENCE.md)
2. [Development Guide](../DEVELOPMENT_GUIDE.md)
3. [Testing Guide](../TESTING.md)
4. [Theory Document](THEORY.md)
5. Example code in [`examples/`](../examples/)

---

## Document Versions

| Document | Last Updated | Version |
|----------|-------------|---------|
| User Guide | January 2025 | 0.1.0 |
| API Reference | January 2025 | 0.1.0 |
| Tutorials | January 2025 | 0.1.0 |
| Cookbook | January 2025 | 0.1.0 |
| Theory | January 2025 | 0.1.0 |

---

## Feedback

Help us improve the documentation:

- Report issues: [GitHub Issues](https://github.com/yourusername/allocation-station/issues)
- Suggest improvements: [GitHub Discussions](https://github.com/yourusername/allocation-station/discussions)
- Contribute: See [Contributing Guide](../CONTRIBUTING.md) (coming soon)

---

**Welcome to Allocation Station!** 🚂

Whether you're planning retirement, managing client portfolios, or conducting quantitative research, we hope this documentation helps you achieve your goals.

Happy investing! 📈
