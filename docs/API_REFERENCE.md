# Allocation Station API Reference

Complete API documentation for the allocation-station package.

## Table of Contents

1. [Core Module](#core-module)
2. [Portfolio Module](#portfolio-module)
3. [Simulation Module](#simulation-module)
4. [Backtesting Module](#backtesting-module)
5. [Analysis Module](#analysis-module)
6. [Optimization Module](#optimization-module)
7. [Visualization Module](#visualization-module)
8. [Data Module](#data-module)
9. [Withdrawal Module](#withdrawal-module)
10. [ML Module](#ml-module)
11. [Integration Module](#integration-module)

---

## Core Module

### `allocation_station.core`

Core data models and structures.

#### `Asset`

Represents a financial asset.

**Constructor:**
```python
Asset(
    symbol: str,
    name: str = None,
    asset_class: AssetClass = AssetClass.EQUITY,
    sector: str = None,
    expense_ratio: float = 0.0,
    tax_treatment: str = "qualified",
    currency: str = "USD"
)
```

**Parameters:**
- `symbol` (str): Ticker symbol
- `name` (str, optional): Full name of the asset
- `asset_class` (AssetClass): Type of asset (EQUITY, BOND, ETF, etc.)
- `sector` (str, optional): Sector classification
- `expense_ratio` (float): Annual expense ratio (default: 0.0)
- `tax_treatment` (str): Tax treatment type
- `currency` (str): Currency denomination

**Methods:**

##### `get_info() -> Dict`
Returns asset information as dictionary.

**Returns:** Dict with asset details

**Example:**
```python
asset = Asset(symbol="SPY", name="S&P 500 ETF", asset_class=AssetClass.ETF)
info = asset.get_info()
```

---

#### `AssetClass`

Enumeration of asset classes.

**Values:**
- `EQUITY`: Stocks and equity funds
- `BOND`: Bonds and fixed income
- `ETF`: Exchange-traded funds
- `MUTUAL_FUND`: Mutual funds
- `REIT`: Real estate investment trusts
- `COMMODITY`: Commodities and futures
- `CRYPTO`: Cryptocurrencies
- `CASH`: Cash and money market
- `OPTION`: Options contracts
- `ALTERNATIVE`: Alternative investments

---

#### `Portfolio`

Portfolio management class.

**Constructor:**
```python
Portfolio(
    name: str,
    description: str = "",
    initial_value: float = 0.0,
    currency: str = "USD"
)
```

**Parameters:**
- `name` (str): Portfolio name
- `description` (str, optional): Portfolio description
- `initial_value` (float): Initial portfolio value
- `currency` (str): Base currency

**Properties:**
- `value` (float): Current portfolio value
- `assets` (List[Asset]): List of assets in portfolio
- `weights` (Dict[str, float]): Asset weights dictionary

**Methods:**

##### `add_asset(asset: Asset, weight: float) -> None`
Add asset to portfolio with specified weight.

**Parameters:**
- `asset` (Asset): Asset to add
- `weight` (float): Portfolio weight (0.0 to 1.0)

**Raises:**
- `ValueError`: If weight is invalid or total weights exceed 1.0

**Example:**
```python
portfolio = Portfolio(name="My Portfolio", initial_value=100000)
portfolio.add_asset(Asset(symbol="SPY"), weight=0.6)
portfolio.add_asset(Asset(symbol="AGG"), weight=0.4)
```

##### `remove_asset(symbol: str) -> None`
Remove asset from portfolio.

**Parameters:**
- `symbol` (str): Asset symbol to remove

##### `get_allocation() -> Dict[str, float]`
Get current portfolio allocation.

**Returns:** Dictionary mapping symbols to weights

##### `update_allocation(allocation: Dict[str, float]) -> None`
Update portfolio allocation.

**Parameters:**
- `allocation` (Dict[str, float]): New allocation weights

##### `rebalance() -> List[Trade]`
Rebalance portfolio to target allocation.

**Returns:** List of trades executed

##### `calculate_metrics() -> Dict[str, float]`
Calculate portfolio metrics.

**Returns:** Dictionary with metrics:
- `expected_return` (float): Expected annual return
- `volatility` (float): Annual volatility
- `sharpe_ratio` (float): Sharpe ratio
- `value_at_risk` (float): 95% VaR
- `max_drawdown` (float): Maximum drawdown

**Example:**
```python
metrics = portfolio.calculate_metrics()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

##### `get_current_values() -> Dict[str, float]`
Get current dollar value of each position.

**Returns:** Dictionary mapping symbols to dollar values

---

#### `Allocation`

Represents a portfolio allocation.

**Constructor:**
```python
Allocation(
    weights: Dict[str, float],
    name: str = "Allocation",
    description: str = ""
)
```

**Parameters:**
- `weights` (Dict[str, float]): Asset weights
- `name` (str): Allocation name
- `description` (str): Description

**Methods:**

##### `normalize() -> None`
Normalize weights to sum to 1.0.

##### `validate() -> bool`
Validate allocation (weights between 0 and 1, sum to 1).

**Returns:** True if valid, raises ValueError if invalid

---

### Enhanced Assets

#### `OptionAsset`

Represents an option contract.

**Constructor:**
```python
OptionAsset(
    symbol: str,
    option_type: OptionType,
    strike_price: float,
    expiration_date: str,
    premium: float,
    underlying_symbol: str = None
)
```

**Parameters:**
- `symbol` (str): Option symbol
- `option_type` (OptionType): CALL or PUT
- `strike_price` (float): Strike price
- `expiration_date` (str): Expiration date (YYYY-MM-DD)
- `premium` (float): Option premium
- `underlying_symbol` (str): Underlying asset symbol

**Methods:**

##### `calculate_greeks() -> Dict[str, float]`
Calculate option Greeks.

**Returns:** Dictionary with delta, gamma, theta, vega, rho

##### `calculate_intrinsic_value(spot_price: float) -> float`
Calculate intrinsic value.

**Parameters:**
- `spot_price` (float): Current spot price

**Returns:** Intrinsic value

---

#### `REITAsset`

Real Estate Investment Trust asset.

**Constructor:**
```python
REITAsset(
    symbol: str,
    name: str,
    property_type: PropertyType,
    ffo: float = None,
    affo: float = None,
    occupancy_rate: float = None
)
```

**Parameters:**
- `symbol` (str): REIT symbol
- `name` (str): REIT name
- `property_type` (PropertyType): Type of property
- `ffo` (float): Funds From Operations per share
- `affo` (float): Adjusted FFO per share
- `occupancy_rate` (float): Occupancy rate (0.0 to 1.0)

---

#### `CryptoAsset`

Cryptocurrency asset.

**Constructor:**
```python
CryptoAsset(
    symbol: str,
    name: str,
    network: str = "Ethereum",
    volatility_adjustment: float = 1.5
)
```

**Parameters:**
- `symbol` (str): Crypto symbol (e.g., "BTC-USD")
- `name` (str): Cryptocurrency name
- `network` (str): Blockchain network
- `volatility_adjustment` (float): Risk adjustment factor

---

#### `CommodityAsset`

Commodity asset.

**Constructor:**
```python
CommodityAsset(
    symbol: str,
    name: str,
    commodity_type: CommodityType,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0
)
```

**Parameters:**
- `symbol` (str): Commodity symbol
- `name` (str): Commodity name
- `commodity_type` (CommodityType): Type of commodity
- `storage_cost` (float): Annual storage cost
- `convenience_yield` (float): Convenience yield

---

## Portfolio Module

### `allocation_station.portfolio`

Portfolio strategies and management.

#### `StrategicAllocation`

Buy-and-hold strategy with periodic rebalancing.

**Constructor:**
```python
StrategicAllocation(
    name: str,
    target_allocation: Dict[str, float],
    rebalance_frequency: str = "quarterly",
    rebalance_threshold: float = 0.05,
    rebalance_method: str = "to_target"
)
```

**Parameters:**
- `name` (str): Strategy name
- `target_allocation` (Dict[str, float]): Target weights
- `rebalance_frequency` (str): "monthly", "quarterly", "annual"
- `rebalance_threshold` (float): Drift threshold for rebalancing
- `rebalance_method` (str): "to_target" or "cash_flow"

**Methods:**

##### `should_rebalance(portfolio: Portfolio, current_date: datetime) -> bool`
Check if rebalancing is needed.

**Returns:** True if rebalancing required

##### `execute_rebalance(portfolio: Portfolio) -> List[Trade]`
Execute rebalancing trades.

**Returns:** List of trades

---

#### `TacticalAllocation`

Dynamic allocation based on market signals.

**Constructor:**
```python
TacticalAllocation(
    name: str,
    base_allocation: Dict[str, float],
    signals: Dict[str, Dict],
    adjustment_limits: float = 0.2
)
```

**Parameters:**
- `name` (str): Strategy name
- `base_allocation` (Dict[str, float]): Base allocation
- `signals` (Dict): Signal definitions
- `adjustment_limits` (float): Max deviation from base

**Methods:**

##### `calculate_adjustments(market_data: pd.DataFrame) -> Dict[str, float]`
Calculate allocation adjustments based on signals.

**Returns:** Adjusted allocation weights

---

#### `RiskParityAllocation`

Equal risk contribution strategy.

**Constructor:**
```python
RiskParityAllocation(
    name: str,
    assets: List[str],
    target_volatility: float = 0.12,
    rebalance_frequency: str = "monthly"
)
```

**Parameters:**
- `name` (str): Strategy name
- `assets` (List[str]): Asset symbols
- `target_volatility` (float): Target portfolio volatility
- `rebalance_frequency` (str): Rebalancing frequency

**Methods:**

##### `calculate_risk_parity_weights(cov_matrix: pd.DataFrame) -> Dict[str, float]`
Calculate risk parity weights.

**Parameters:**
- `cov_matrix` (pd.DataFrame): Covariance matrix

**Returns:** Risk parity weights

##### `get_risk_contributions(portfolio: Portfolio) -> Dict[str, float]`
Get risk contribution of each asset.

**Returns:** Risk contributions

---

#### `WithdrawalStrategy`

Retirement withdrawal strategy.

**Constructor:**
```python
WithdrawalStrategy(
    name: str,
    method: WithdrawalMethod,
    initial_withdrawal_rate: float = 0.04,
    inflation_adjustment: bool = True,
    withdrawal_floor: float = None,
    withdrawal_ceiling: float = None
)
```

**Parameters:**
- `name` (str): Strategy name
- `method` (WithdrawalMethod): Withdrawal method
- `initial_withdrawal_rate` (float): Initial withdrawal rate
- `inflation_adjustment` (bool): Adjust for inflation
- `withdrawal_floor` (float): Minimum withdrawal amount
- `withdrawal_ceiling` (float): Maximum withdrawal amount

**Methods:**

##### `calculate_withdrawal(portfolio_value: float, year: int, inflation_rate: float) -> float`
Calculate withdrawal amount for given year.

**Parameters:**
- `portfolio_value` (float): Current portfolio value
- `year` (int): Year number
- `inflation_rate` (float): Inflation rate

**Returns:** Withdrawal amount

---

### Advanced Portfolio Features

#### `TaxLossHarvester`

Tax-loss harvesting implementation.

**Constructor:**
```python
TaxLossHarvester(
    portfolio: Portfolio,
    tax_rate: float,
    wash_sale_period: int = 30,
    minimum_loss: float = 1000
)
```

**Methods:**

##### `find_opportunities() -> List[HarvestOpportunity]`
Find tax-loss harvesting opportunities.

**Returns:** List of opportunities

##### `execute_harvest(opportunity: HarvestOpportunity) -> List[Trade]`
Execute tax-loss harvest.

**Returns:** List of trades executed

---

#### `MultiCurrencyPortfolio`

Multi-currency portfolio management.

**Constructor:**
```python
MultiCurrencyPortfolio(
    name: str,
    base_currency: str,
    initial_value: float
)
```

**Methods:**

##### `add_asset(asset: Asset, weight: float, currency: str) -> None`
Add asset with currency specification.

##### `get_fx_exposure() -> Dict[str, float]`
Get foreign exchange exposure.

**Returns:** Currency exposures

##### `set_fx_manager(manager: FXRiskManager) -> None`
Set FX risk manager for hedging.

---

## Simulation Module

### `allocation_station.simulation`

Monte Carlo simulation engine.

#### `MonteCarloSimulator`

Standard Monte Carlo simulator.

**Constructor:**
```python
MonteCarloSimulator(config: SimulationConfig)
```

**Parameters:**
- `config` (SimulationConfig): Simulation configuration

**Methods:**

##### `simulate(portfolio: Portfolio, strategy: AllocationStrategy, withdrawal_strategy: WithdrawalStrategy = None) -> SimulationResults`
Run Monte Carlo simulation.

**Parameters:**
- `portfolio` (Portfolio): Portfolio to simulate
- `strategy` (AllocationStrategy): Allocation strategy
- `withdrawal_strategy` (WithdrawalStrategy, optional): Withdrawal strategy

**Returns:** SimulationResults object

**Example:**
```python
config = SimulationConfig(n_simulations=10000, time_horizon=30)
simulator = MonteCarloSimulator(config)
results = simulator.simulate(portfolio, strategy)
```

---

#### `SimulationConfig`

Configuration for Monte Carlo simulation.

**Constructor:**
```python
SimulationConfig(
    n_simulations: int = 1000,
    time_horizon: int = 30,
    time_steps: int = 252,
    expected_returns: Dict[str, float] = None,
    volatilities: Dict[str, float] = None,
    correlation_matrix: pd.DataFrame = None,
    inflation_rate: float = 0.025,
    random_seed: int = None,
    n_jobs: int = 1
)
```

**Parameters:**
- `n_simulations` (int): Number of simulation runs
- `time_horizon` (int): Simulation horizon in years
- `time_steps` (int): Steps per year (252 for daily)
- `expected_returns` (Dict[str, float]): Expected returns by asset
- `volatilities` (Dict[str, float]): Volatilities by asset
- `correlation_matrix` (pd.DataFrame): Correlation matrix
- `inflation_rate` (float): Annual inflation rate
- `random_seed` (int): Random seed for reproducibility
- `n_jobs` (int): Number of parallel jobs (-1 for all cores)

---

#### `SimulationResults`

Results from Monte Carlo simulation.

**Properties:**
- `success_rate` (float): Success rate (portfolio survives)
- `median_final_value` (float): Median ending value
- `mean_final_value` (float): Mean ending value
- `ruin_probability` (float): Probability of ruin
- `paths` (np.ndarray): All simulation paths

**Methods:**

##### `percentile(p: float) -> float`
Get value at percentile p.

**Parameters:**
- `p` (float): Percentile (0-100)

**Returns:** Value at percentile

##### `summary() -> Dict`
Get summary statistics.

**Returns:** Dictionary of summary metrics

##### `plot_paths(n_paths: int = 100) -> Figure`
Plot simulation paths.

**Parameters:**
- `n_paths` (int): Number of paths to plot

**Returns:** Plotly figure

---

#### `RegimeSwitchingSimulator`

Regime-switching Monte Carlo simulator.

**Constructor:**
```python
RegimeSwitchingSimulator(
    regimes: Dict[str, Dict],
    transition_matrix: np.ndarray
)
```

**Parameters:**
- `regimes` (Dict): Regime definitions with returns, volatility, probability
- `transition_matrix` (np.ndarray): Regime transition probabilities

---

#### `GARCHSimulator`

GARCH volatility model simulator.

**Constructor:**
```python
GARCHSimulator(
    omega: float,
    alpha: float,
    beta: float,
    mean_return: float
)
```

**Parameters:**
- `omega` (float): Base variance
- `alpha` (float): ARCH coefficient
- `beta` (float): GARCH coefficient
- `mean_return` (float): Mean return

---

## Backtesting Module

### `allocation_station.backtesting`

Historical backtesting framework.

#### `BacktestEngine`

Backtesting engine.

**Constructor:**
```python
BacktestEngine(config: BacktestConfig)
```

**Methods:**

##### `run(portfolio: Portfolio, strategy: AllocationStrategy) -> BacktestResults`
Run backtest.

**Parameters:**
- `portfolio` (Portfolio): Portfolio to backtest
- `strategy` (AllocationStrategy): Strategy to test

**Returns:** BacktestResults object

---

#### `BacktestConfig`

Backtest configuration.

**Constructor:**
```python
BacktestConfig(
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    rebalance_frequency: str = "quarterly",
    benchmark_symbol: str = "SPY"
)
```

---

#### `BacktestResults`

Backtest results.

**Properties:**
- `total_return` (float): Total return
- `cagr` (float): Compound annual growth rate
- `sharpe_ratio` (float): Sharpe ratio
- `max_drawdown` (float): Maximum drawdown
- `win_rate` (float): Winning periods percentage

**Methods:**

##### `get_metrics() -> Dict[str, float]`
Get all performance metrics.

##### `get_transaction_costs() -> Dict`
Get transaction cost analysis.

##### `plot_performance() -> Figure`
Plot backtest performance.

---

## Analysis Module

### `allocation_station.analysis`

Portfolio analysis tools.

#### `EfficientFrontier`

Efficient frontier calculation.

**Constructor:**
```python
EfficientFrontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02
)
```

**Methods:**

##### `optimize(objective: str, **kwargs) -> OptimalPortfolio`
Find optimal portfolio.

**Parameters:**
- `objective` (str): "max_sharpe", "min_variance", "risk_parity", "max_return"
- `**kwargs`: Additional objective-specific parameters

**Returns:** OptimalPortfolio object

**Example:**
```python
ef = EfficientFrontier(returns, cov_matrix)
max_sharpe = ef.optimize("max_sharpe")
print(max_sharpe.weights)
```

##### `efficient_frontier_curve(n_points: int = 100) -> pd.DataFrame`
Calculate efficient frontier curve.

**Returns:** DataFrame with returns and volatilities

---

#### `RiskAnalyzer`

Risk analysis tools.

**Constructor:**
```python
RiskAnalyzer(portfolio: Portfolio)
```

**Methods:**

##### `calculate_var(confidence_level: float = 0.95, time_horizon: int = 1) -> float`
Calculate Value at Risk.

**Parameters:**
- `confidence_level` (float): Confidence level
- `time_horizon` (int): Time horizon in days

**Returns:** VaR value

##### `calculate_cvar(confidence_level: float = 0.95) -> float`
Calculate Conditional VaR.

**Returns:** CVaR value

##### `calculate_max_drawdown() -> float`
Calculate maximum drawdown.

**Returns:** Maximum drawdown as decimal

##### `drawdown_statistics() -> Dict`
Get drawdown statistics.

**Returns:** Dictionary with drawdown metrics

---

#### `StressTester`

Stress testing framework.

**Constructor:**
```python
StressTester(portfolio: Portfolio)
```

**Methods:**

##### `run_scenarios(scenarios: Dict[str, Dict]) -> Dict`
Run stress test scenarios.

**Parameters:**
- `scenarios` (Dict): Scenario definitions

**Returns:** Results for each scenario

---

## Optimization Module

### `allocation_station.optimization`

Portfolio optimization algorithms.

#### `BlackLittermanOptimizer`

Black-Litterman model.

**Constructor:**
```python
BlackLittermanOptimizer(
    market_weights: Dict[str, float],
    risk_aversion: float,
    cov_matrix: pd.DataFrame
)
```

**Methods:**

##### `optimize(views: List[Dict]) -> Dict[str, float]`
Optimize with investor views.

**Parameters:**
- `views` (List[Dict]): List of views

**Returns:** Optimal weights

---

#### `RobustOptimizer`

Robust optimization under uncertainty.

**Constructor:**
```python
RobustOptimizer(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    return_uncertainty: float,
    covariance_uncertainty: float
)
```

---

## Visualization Module

### `allocation_station.visualization`

Visualization tools.

#### Functions

##### `plot_portfolio_performance(portfolio_returns, benchmark_returns=None, title="Portfolio Performance") -> Figure`
Plot portfolio performance over time.

##### `plot_efficient_frontier(efficient_frontier, portfolios=None, show_assets=True) -> Figure`
Plot efficient frontier.

##### `plot_monte_carlo_paths(simulation_results, n_paths=100, show_percentiles=None) -> Figure`
Plot Monte Carlo simulation paths.

##### `plot_allocation_pie(allocation, title="Portfolio Allocation") -> Figure`
Plot allocation pie chart.

##### `plot_drawdown(returns, title="Drawdown Analysis") -> Figure`
Plot drawdown chart.

---

## Data Module

### `allocation_station.data`

Market data management.

#### `MarketDataProvider`

Market data provider.

**Constructor:**
```python
MarketDataProvider(
    source: str = "yahoo",
    api_key: str = None,
    cache_dir: str = "data/cache"
)
```

**Methods:**

##### `get_historical_data(symbols, start_date, end_date, frequency="daily") -> pd.DataFrame`
Fetch historical market data.

**Parameters:**
- `symbols` (List[str] or str): Ticker symbols
- `start_date` (str or datetime): Start date
- `end_date` (str or datetime): End date
- `frequency` (str): "daily", "weekly", "monthly"

**Returns:** DataFrame with OHLCV data

---

## Withdrawal Module

### `allocation_station.withdrawal`

Advanced withdrawal strategies.

#### `OptimalWithdrawal`

Optimal withdrawal using dynamic programming.

**Methods:**

##### `compute_optimal_withdrawals() -> pd.Series`
Compute optimal withdrawal path.

---

#### `SocialSecurityOptimizer`

Social Security claiming optimization.

**Methods:**

##### `optimize(portfolio, life_expectancy, discount_rate) -> Dict`
Find optimal claiming strategy.

---

## ML Module

### `allocation_station.ml`

Machine learning integration.

#### `ReturnPredictor`

Return prediction model.

**Methods:**

##### `train(data) -> None`
Train prediction model.

##### `predict(features) -> np.ndarray`
Predict returns.

---

## Integration Module

### `allocation_station.integrations`

External integrations.

#### `BrokerAPI`

Broker API integration.

**Methods:**

##### `connect() -> None`
Connect to broker.

##### `get_positions() -> List[Position]`
Get current positions.

##### `place_order(order) -> OrderResponse`
Place order.

---

**Last Updated**: January 2025
**Version**: 0.1.0
