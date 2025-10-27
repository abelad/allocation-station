# Allocation Station 🚂

A comprehensive asset allocation strategy testing and analysis framework for portfolio management, Monte Carlo simulation, and retirement planning.

## Features

### Core Capabilities
- **Portfolio Management**: Create and manage multi-asset portfolios with stocks, bonds, ETFs, and more
- **Strategy Testing**: Backtest various allocation strategies against historical market data
- **Monte Carlo Simulation**: Run thousands of simulations to test portfolio resilience
- **Efficient Frontier Analysis**: Optimize portfolios using Modern Portfolio Theory
- **Withdrawal Strategies**: Test retirement withdrawal strategies including 4% rule, Guyton-Klinger, and more
- **Risk Analysis**: Comprehensive risk metrics including VaR, CVaR, maximum drawdown, and Sharpe ratio
- **Visualization**: Interactive charts and dashboards for portfolio analysis

### Supported Strategies
- Strategic (Buy & Hold) Allocation
- Tactical Asset Allocation
- Risk Parity
- Mean-Variance Optimization
- Target Date Glide Paths
- Constant Mix Rebalancing

## Installation

### Requirements
- Python 3.9 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/allocation-station.git
cd allocation-station

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Install with Development Dependencies

```bash
pip install -e .[dev]
```

## Quick Start

```python
from allocation_station import Portfolio, Asset
from allocation_station.portfolio import StrategicAllocation
from allocation_station.simulation import MonteCarloSimulator

# Create a portfolio
portfolio = Portfolio(name="My Retirement Portfolio", initial_value=500000)

# Add assets
portfolio.add_asset(
    Asset(symbol="SPY", name="S&P 500 ETF", asset_class="equity"),
    weight=0.6
)
portfolio.add_asset(
    Asset(symbol="AGG", name="Bond ETF", asset_class="bond"),
    weight=0.4
)

# Define allocation strategy
strategy = StrategicAllocation(
    name="60/40 Portfolio",
    target_allocation={"SPY": 0.6, "AGG": 0.4},
    rebalance_frequency="quarterly"
)

# Run Monte Carlo simulation
simulator = MonteCarloSimulator(
    n_simulations=1000,
    time_horizon=30
)
results = simulator.simulate(portfolio, strategy)

# View results
print(f"Success Rate: {results.success_rate:.1%}")
print(f"Median Final Value: ${results.median_final_value:,.0f}")
```

## Examples

Check the `examples/` directory for detailed examples:

- `basic_portfolio_example.py`: Simple portfolio creation and analysis
- More examples coming soon!

## Project Structure

```
allocation-station/
├── src/allocation_station/
│   ├── core/              # Core data models
│   ├── data/              # Market data management
│   ├── portfolio/         # Portfolio strategies
│   ├── simulation/        # Monte Carlo engine
│   ├── backtesting/       # Backtesting framework
│   ├── analysis/          # Analysis tools
│   └── visualization/     # Charting utilities
├── examples/              # Usage examples
├── tests/                 # Test suite
├── docs/                  # Documentation
└── DEVELOPMENT_GUIDE.md   # Development roadmap
```

## Key Components

### Portfolio Management
- Create multi-asset portfolios
- Track allocations and rebalancing
- Support for various asset classes (stocks, bonds, ETFs, REITs, etc.)
- Tax-aware strategies (coming soon)

### Market Data
- Integration with Yahoo Finance
- Historical data caching
- Support for multiple data providers (expandable)
- Real-time price updates

### Simulation Engine
- Monte Carlo simulations with configurable parameters
- Multiple return distributions (normal, t-distribution, historical bootstrap)
- Parallel processing for performance
- Scenario analysis and stress testing

### Backtesting
- Historical strategy testing
- Transaction cost modeling
- Realistic rebalancing constraints
- Performance attribution analysis

### Risk Analysis
- Value at Risk (VaR) and Conditional VaR
- Maximum drawdown analysis
- Correlation and covariance matrices
- Factor-based risk decomposition

### Visualization
- Interactive Plotly charts
- Portfolio performance dashboards
- Efficient frontier visualization
- Monte Carlo path analysis
- Drawdown charts

## Development Roadmap

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for a comprehensive development checklist including:

- Phase 1: Core Enhancements (data, assets, portfolio features)
- Phase 2: Analysis & Optimization (risk analysis, attribution)
- Phase 3: Simulation & Forecasting (advanced Monte Carlo, ML integration)
- Phase 4: Withdrawal & Income Planning
- Phase 5: User Interface & Reporting
- Phase 6: Integration & APIs
- Phase 7: Testing & Documentation
- Phase 8: Performance & Scalability
- Phase 9: Advanced Features
- Phase 10: Compliance & Risk Management

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Standards
- Follow PEP 8 style guide
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Maintain >80% code coverage

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=allocation_station

# Run specific test file
pytest tests/test_portfolio.py
```

## Documentation

📚 **Comprehensive documentation is now available!**

Visit the [**Documentation Hub**](docs/README.md) for complete guides, tutorials, and reference materials.

### Quick Links

- **[User Guide](docs/USER_GUIDE.md)** - Complete guide to all features and capabilities
- **[Tutorials](docs/TUTORIALS.md)** - Step-by-step walkthroughs for common tasks
- **[Cookbook](docs/COOKBOOK.md)** - Quick code recipes for 50+ common scenarios
- **[API Reference](docs/API_REFERENCE.md)** - Detailed API documentation for all modules
- **[Theory](docs/THEORY.md)** - Mathematical foundations and financial theory

### Documentation Highlights

| Topic | Quick Link |
|-------|------------|
| Getting Started | [Installation & Quick Start](docs/USER_GUIDE.md#installation) |
| Building Portfolios | [Tutorial 1](docs/TUTORIALS.md#tutorial-1-building-your-first-portfolio) |
| Monte Carlo Simulation | [Tutorial 2](docs/TUTORIALS.md#tutorial-2-running-monte-carlo-simulations) |
| Portfolio Optimization | [Tutorial 4](docs/TUTORIALS.md#tutorial-4-portfolio-optimization) |
| Retirement Planning | [Tutorial 5](docs/TUTORIALS.md#tutorial-5-retirement-planning) |
| Risk Analysis | [Tutorial 6](docs/TUTORIALS.md#tutorial-6-risk-analysis) |
| Code Recipes | [Cookbook - 50+ Recipes](docs/COOKBOOK.md) |

### For Different Users

- **New Users**: Start with the [User Guide](docs/USER_GUIDE.md) and [Tutorial 1](docs/TUTORIALS.md#tutorial-1-building-your-first-portfolio)
- **Developers**: See [API Reference](docs/API_REFERENCE.md) and [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)
- **Researchers**: Read [Theory](docs/THEORY.md) for mathematical foundations
- **Quick Reference**: Browse the [Cookbook](docs/COOKBOOK.md) for code snippets

## Use Cases

- **Retirement Planning**: Test withdrawal strategies and portfolio sustainability
- **Investment Research**: Analyze different allocation strategies
- **Risk Management**: Stress test portfolios under various market conditions
- **Financial Advisory**: Create client portfolios with optimization
- **Academic Research**: Study portfolio theory and market behavior
- **Robo-Advisory**: Build automated portfolio management systems

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Modern Portfolio Theory (Markowitz)
- Black-Litterman Model
- Monte Carlo methods in finance
- Open source Python community

## Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: See `/docs` folder
- **Examples**: Check `/examples` folder

## Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors before making investment choices.

---

**Current Version**: 0.1.0
**Python Support**: 3.9+
**Status**: Active Development
**Last Updated**: 2024  
