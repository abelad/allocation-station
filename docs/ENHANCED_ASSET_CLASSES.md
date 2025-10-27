

 # Enhanced Asset Classes Documentation

This document describes the enhanced asset class features implemented in Allocation Station, including support for options, REITs, cryptocurrencies, commodities, alternatives, and structured products.

## Overview

The enhanced asset class system extends the base `Asset` class with specialized implementations for:

1. **Options & Derivatives**: Full options contract modeling with Greeks
2. **REITs**: Real estate investment trusts with property-specific metrics
3. **Cryptocurrencies**: Digital assets with volatility adjustments and risk scoring
4. **Commodities & Futures**: Physical commodities and futures contracts
5. **Alternative Investments**: Private equity, hedge funds, and other alternatives
6. **Structured Products**: Complex products combining derivatives and securities
7. **Correlation Models**: Advanced correlation analysis and regime detection

## Options and Derivatives

### OptionAsset Class

Models options contracts with full Greek support, intrinsic/time value calculation, and moneyness analysis.

```python
from allocation_station.core import OptionAsset, OptionType, OptionStyle
from datetime import date

# Create a call option
call = OptionAsset(
    symbol="AAPL250117C00150000",
    name="AAPL Jan 2025 150 Call",
    option_type=OptionType.CALL,
    option_style=OptionStyle.AMERICAN,
    underlying_symbol="AAPL",
    strike_price=150.0,
    expiration_date=date(2025, 1, 17),
    delta=0.65,
    gamma=0.05,
    theta=-0.15,
    vega=0.25,
    implied_volatility=0.30
)

# Calculate values
underlying_price = 155.00
intrinsic = call.calculate_intrinsic_value(underlying_price)
time_value = call.calculate_time_value(12.50, underlying_price)

# Check moneyness
moneyness = call.moneyness(underlying_price)  # Returns "ITM", "ATM", or "OTM"
is_itm = call.is_in_the_money(underlying_price)
days_left = call.days_to_expiration()
```

**Key Features:**
- Call and Put options
- American, European, and Bermudan styles
- Complete Greeks (delta, gamma, theta, vega, rho)
- Intrinsic and time value calculation
- Moneyness determination
- Days to expiration calculation

**Option Types:**
- `OptionType.CALL`: Call option
- `OptionType.PUT`: Put option

**Exercise Styles:**
- `OptionStyle.AMERICAN`: Can exercise anytime
- `OptionStyle.EUROPEAN`: Exercise only at expiration
- `OptionStyle.BERMUDAN`: Exercise on specific dates

## Real Estate Investment Trusts (REITs)

### REITAsset Class

Models REITs with property-type tracking, FFO metrics, and REIT-specific ratios.

```python
from allocation_station.core import REITAsset, REITType

reit = REITAsset(
    symbol="VNQ",
    name="Vanguard Real Estate ETF",
    reit_type=REITType.EQUITY,
    property_types=["Office", "Retail", "Residential", "Industrial"],
    geographic_focus=["United States"],
    dividend_yield=0.042,  # 4.2%
    ffo_per_share=3.50,
    affo_per_share=3.30,
    price_to_ffo=25.0,
    debt_to_equity=0.45,
    occupancy_rate=0.92,  # 92%
    total_properties=180
)

# Calculate REIT-specific metrics
metrics = reit.calculate_reit_metrics()
```

**Key Features:**
- Property type classification
- Geographic focus tracking
- FFO (Funds From Operations) metrics
- AFFO (Adjusted FFO) calculation
- Occupancy rates
- Debt ratios
- Price-to-FFO valuation

**REIT Types:**
- `REITType.EQUITY`: Own and operate properties
- `REITType.MORTGAGE`: Finance income-producing real estate
- `REITType.HYBRID`: Combination of equity and mortgage

## Cryptocurrencies

### CryptoAsset Class

Models cryptocurrencies with blockchain-specific attributes, volatility adjustments, and risk scoring.

```python
from allocation_station.core import CryptoAsset, CryptoType

btc = CryptoAsset(
    symbol="BTC-USD",
    name="Bitcoin",
    crypto_type=CryptoType.COIN,
    blockchain="Bitcoin",
    consensus_mechanism="Proof of Work",
    circulating_supply=19.5e6,
    max_supply=21e6,
    market_cap=650e9,
    market_dominance=0.45,
    trading_volume_24h=25e9,
    volatility_30d=0.55,
    hash_rate=350e18,
    volatility_multiplier=1.5
)

# Calculate adjusted volatility
adjusted_vol = btc.calculate_adjusted_volatility()

# Calculate risk score (0-100)
risk_score = btc.calculate_risk_score()
```

**Key Features:**
- Blockchain and consensus mechanism tracking
- Supply metrics (circulating, total, max)
- Market dominance calculation
- Volatility adjustments for crypto-specific risk
- Risk scoring system
- DeFi metrics (TVL, staking APY)
- Network metrics (hash rate, active addresses)

**Crypto Types:**
- `CryptoType.COIN`: Native blockchain currency (BTC, ETH)
- `CryptoType.TOKEN`: Built on another blockchain (ERC-20)
- `CryptoType.STABLECOIN`: Pegged to fiat currency
- `CryptoType.DEFI`: Decentralized finance tokens

**Volatility Adjustment:**
- Base volatility multiplied by `volatility_multiplier` (default 1.5x)
- Additional 20% adjustment for market cap < $1B
- Accounts for crypto-specific risk factors

## Commodities and Futures

### CommodityAsset Class

Models physical commodities with storage costs, convenience yield, and contango/backwardation analysis.

```python
from allocation_station.core import CommodityAsset, CommodityType

gold = CommodityAsset(
    symbol="GC=F",
    name="Gold",
    commodity_type=CommodityType.METALS,
    unit="troy ounce",
    spot_price=2050.00,
    futures_price=2065.00,
    storage_costs=0.005,  # 0.5% annual
    convenience_yield=0.001,
    contango_backwardation="contango"
)

# Calculate cost of carry
carry_cost = gold.calculate_carry_cost(risk_free_rate=0.04)

# Check market structure
is_contango = gold.is_contango()  # True if futures > spot
```

**Commodity Types:**
- `CommodityType.ENERGY`: Oil, natural gas
- `CommodityType.METALS`: Gold, silver, copper
- `CommodityType.AGRICULTURE`: Corn, wheat, soybeans
- `CommodityType.LIVESTOCK`: Cattle, hogs

### FutureContract Class

Models futures contracts with margin requirements and leverage calculation.

```python
from allocation_station.core import FutureContract, FutureType
from datetime import date

oil_future = FutureContract(
    symbol="CL=F",
    name="WTI Crude Oil Future",
    future_type=FutureType.COMMODITY,
    underlying_symbol="WTI",
    contract_month="Dec2024",
    contract_size=1000.0,  # 1,000 barrels
    initial_margin=5000.0,
    maintenance_margin=4000.0,
    settlement_type="physical",
    last_trading_day=date(2024, 11, 20)
)

# Calculate leverage
contract_value = 78.50 * 1000  # Price * size
leverage = oil_future.calculate_leverage(contract_value)
```

**Future Types:**
- `FutureType.COMMODITY`: Physical commodities
- `FutureType.FINANCIAL`: Financial instruments
- `FutureType.CURRENCY`: Currency pairs
- `FutureType.INDEX`: Stock indices

## Alternative Investments

### AlternativeAsset Class

Models private equity, hedge funds, and other alternative investments with performance metrics and fee structures.

```python
from allocation_station.core import AlternativeAsset, AlternativeType

pe_fund = AlternativeAsset(
    symbol="PE-TECH-2020",
    name="Tech Growth PE Fund 2020",
    alternative_type=AlternativeType.PRIVATE_EQUITY,
    vintage_year=2020,
    investment_period=5,
    lock_up_period=10,
    management_fee=0.02,  # 2%
    performance_fee=0.20,  # 20% carry
    hurdle_rate=0.08,  # 8%
    irr=0.25,  # 25% IRR
    moic=2.8,  # 2.8x multiple
    tvpi=2.8,
    dpi=1.2,  # Distributed 1.2x
    rvpi=1.6,  # Remaining 1.6x
    liquidity_tier="Illiquid",
    strategy="Growth equity in technology sector"
)

# Calculate net return after fees
gross_return = 0.30
net_return = pe_fund.calculate_net_return(gross_return)
```

**Alternative Types:**
- `AlternativeType.PRIVATE_EQUITY`: Private equity funds
- `AlternativeType.VENTURE_CAPITAL`: VC funds
- `AlternativeType.HEDGE_FUND`: Hedge funds
- `AlternativeType.REAL_ASSETS`: Real assets
- `AlternativeType.INFRASTRUCTURE`: Infrastructure investments
- `AlternativeType.ART_COLLECTIBLES`: Art and collectibles

**Key Metrics:**
- **IRR**: Internal Rate of Return
- **MOIC**: Multiple on Invested Capital
- **TVPI**: Total Value to Paid-In
- **DPI**: Distributions to Paid-In
- **RVPI**: Residual Value to Paid-In

## Structured Products

### StructuredProduct Class

Models structured products with embedded options, protection levels, and participation rates.

```python
from allocation_station.core import StructuredProduct, StructuredProductType
from datetime import date

structured = StructuredProduct(
    symbol="SPX-PP-2027",
    name="S&P 500 Principal Protected Note 2027",
    product_type=StructuredProductType.PRINCIPAL_PROTECTED,
    underlying_assets=["SPX"],
    protection_level=1.0,  # 100% principal protection
    participation_rate=0.75,  # 75% participation
    cap_level=0.40,  # 40% cap
    maturity_date=date(2027, 12, 31),
    issue_price=100.0,
    issuer="Major Bank",
    issuer_credit_rating="AA-"
)

# Calculate payoff for different scenarios
market_return = 0.30  # 30% market return
payoff = structured.calculate_payoff(market_return, at_maturity=True)
```

**Product Types:**
- `StructuredProductType.PRINCIPAL_PROTECTED`: Capital protection
- `StructuredProductType.YIELD_ENHANCEMENT`: Enhanced yield
- `StructuredProductType.PARTICIPATION`: Market participation
- `StructuredProductType.LEVERAGED`: Leveraged exposure

**Key Components:**
- **Protection Level**: Downside protection (0-1)
- **Participation Rate**: Upside participation
- **Cap Level**: Maximum return
- **Barrier Level**: Knock-in/knock-out threshold

## Correlation Models

### AssetClassCorrelationModel

Provides default correlations between asset classes for normal and stress scenarios.

```python
from allocation_station.core import AssetClassCorrelationModel

corr_model = AssetClassCorrelationModel()

# Get correlation between two asset classes
corr = corr_model.get_correlation("equity", "bond")  # Returns -0.15

# Stress scenario
stress_corr = corr_model.get_correlation("equity", "bond", stress_mode=True)

# Get full matrix
asset_classes = ["equity", "bond", "commodity", "real_estate", "cryptocurrency"]
matrix = corr_model.get_correlation_matrix(asset_classes)
```

**Default Correlations:**
- Equity-Bond: -0.15 (diversification benefit)
- Equity-Real Estate: 0.60 (positive correlation)
- Equity-Crypto: 0.30 (moderate correlation)
- Bond-Commodity: -0.10 (slight negative)

**Stress Correlations:**
- Correlations typically increase during stress
- Diversification benefits may decrease
- Flight-to-quality increases bond-equity negative correlation

### DynamicCorrelationModel

Estimates time-varying correlations and detects regime changes.

```python
from allocation_station.core import DynamicCorrelationModel
import pandas as pd

# Create model with return data
dcc_model = DynamicCorrelationModel(returns)

# Calculate rolling correlations
rolling_corr = dcc_model.calculate_rolling_correlation(window=60)

# Detect correlation regimes
regimes = dcc_model.detect_correlation_regimes(window=60)

# EWMA correlation
ewma_corr = dcc_model.calculate_ewma_correlation(halflife=30)
```

**Correlation Regimes:**
- `CorrelationRegime.LOW`: < 0.3
- `CorrelationRegime.MODERATE`: 0.3 - 0.6
- `CorrelationRegime.HIGH`: 0.6 - 0.8
- `CorrelationRegime.EXTREME`: > 0.8

### HierarchicalCorrelationModel

Hierarchical clustering of assets based on correlations.

```python
from allocation_station.core import HierarchicalCorrelationModel

hier_model = HierarchicalCorrelationModel(returns)

# Get asset clusters
clusters = hier_model.get_clusters(n_clusters=3)

# Calculate diversification score
div_score = hier_model.get_diversification_score()  # 0-1, higher is better
```

### CopulaCorrelationModel

Tail dependency analysis for extreme events.

```python
from allocation_station.core import CopulaCorrelationModel

copula_model = CopulaCorrelationModel(returns)

# Calculate tail dependencies
lower_tail, upper_tail = copula_model.calculate_tail_dependence(
    "asset1", "asset2", quantile=0.05
)

# Identify contagion risks
contagion_pairs = copula_model.identify_contagion_risk(threshold=0.5)
```

**Tail Dependencies:**
- **Lower Tail**: Co-movement during market crashes
- **Upper Tail**: Co-movement during rallies
- **Contagion Risk**: High tail dependency indicates crisis contagion

## Utility Functions

### Correlation Matrix Operations

```python
from allocation_station.core.correlation_models import (
    estimate_correlation_from_data,
    shrink_correlation_matrix,
    ensure_positive_definite,
    calculate_rolling_correlation_stability
)

# Estimate from data
corr = estimate_correlation_from_data(returns, method="pearson")

# Apply Ledoit-Wolf shrinkage
shrunk_corr = shrink_correlation_matrix(corr, shrinkage_factor=0.2)

# Ensure positive definite
fixed_corr = ensure_positive_definite(corr)

# Calculate stability
stability = calculate_rolling_correlation_stability(returns, window=60)
```

## Best Practices

### Options Trading

1. **Greeks Management**: Monitor delta for directional exposure, theta for time decay
2. **Implied Volatility**: Compare to historical volatility for relative value
3. **Moneyness**: Understand ITM/OTM probabilities for strategy selection
4. **Time Decay**: Short options benefit from theta, long options lose value

### REIT Investment

1. **FFO Analysis**: Use FFO instead of earnings for valuation
2. **Occupancy Rates**: High occupancy indicates strong demand
3. **Debt Levels**: Monitor debt-to-equity for financial stability
4. **Property Diversification**: Multiple property types reduce risk

### Cryptocurrency

1. **Volatility Adjustment**: Use crypto-specific volatility multipliers
2. **Market Cap**: Larger coins generally less risky
3. **Supply Dynamics**: Understand inflation/deflation mechanisms
4. **Risk Scoring**: Use risk scores for portfolio allocation limits

### Commodities

1. **Contango/Backwardation**: Understand roll yield impact
2. **Storage Costs**: Factor into total return expectations
3. **Convenience Yield**: Consider benefits of physical ownership
4. **Leverage**: Futures provide significant leverage - manage carefully

### Alternative Investments

1. **Fee Impact**: Calculate net returns after all fees
2. **Illiquidity**: Account for lock-up periods in portfolio planning
3. **Vintage Year**: Different vintages have different return profiles
4. **Due Diligence**: Verify IRR and MOIC calculations

### Structured Products

1. **Issuer Credit**: Structured products have issuer credit risk
2. **Hidden Costs**: Embedded options have implicit costs
3. **Complexity**: Fully understand payoff structure
4. **Alternatives**: Compare to simpler strategies with similar payoffs

### Correlation Analysis

1. **Time-Varying**: Correlations change over time - use dynamic models
2. **Stress Testing**: Test portfolios under stress correlations
3. **Tail Risk**: Use copula models for tail dependency
4. **Diversification**: Monitor diversification score regularly

## Examples

See [examples/enhanced_assets_example.py](../examples/enhanced_assets_example.py) for comprehensive examples demonstrating:
- Options valuation and Greeks
- REIT analysis with FFO metrics
- Cryptocurrency risk assessment
- Commodity carry cost calculation
- Alternative investment fee analysis
- Structured product payoff scenarios
- Correlation regime detection
- Tail dependency analysis

## API Reference

For detailed API documentation, see:
- [enhanced_assets.py](../src/allocation_station/core/enhanced_assets.py) - Asset class implementations
- [correlation_models.py](../src/allocation_station/core/correlation_models.py) - Correlation models

## Integration with Portfolio Management

All enhanced asset classes inherit from the base `Asset` class and can be used seamlessly with existing portfolio management features:

```python
from allocation_station import Portfolio
from allocation_station.core import OptionAsset, CryptoAsset, REITAsset

portfolio = Portfolio(name="Diversified Portfolio", initial_value=100000)

# Add various asset types
portfolio.add_asset(option_asset, weight=0.05)
portfolio.add_asset(crypto_asset, weight=0.03)
portfolio.add_asset(reit_asset, weight=0.15)

# Portfolio analytics work with all asset types
metrics = portfolio.calculate_metrics()
```

## Performance Considerations

- **Options**: Greeks calculations are fast, can support large option portfolios
- **Correlation Models**: Dynamic models can be computationally intensive for large universes
- **Tail Dependencies**: Copula models require sufficient data for accuracy
- **Hierarchical Clustering**: Scales well up to ~100 assets

## Future Enhancements

Potential future additions:
- Exotic options (Asian, barrier, etc.)
- CDS and credit derivatives
- Weather derivatives
- Carbon credits
- NFTs and digital assets
- More sophisticated option pricing models (Black-Scholes, binomial trees)
