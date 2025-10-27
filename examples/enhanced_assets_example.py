"""
Example demonstrating enhanced asset classes.

This example shows how to:
1. Create and use options contracts
2. Model REITs with property-specific metrics
3. Handle cryptocurrencies with volatility adjustments
4. Work with commodities and futures
5. Model alternative investments (private equity, hedge funds)
6. Create structured products
7. Analyze asset class correlations
"""

from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd

from allocation_station.core import (
    OptionAsset, OptionType, OptionStyle,
    REITAsset, REITType,
    CryptoAsset, CryptoType,
    CommodityAsset, CommodityType,
    FutureContract, FutureType,
    AlternativeAsset, AlternativeType,
    StructuredProduct, StructuredProductType,
    AssetClassCorrelationModel,
    DynamicCorrelationModel,
    HierarchicalCorrelationModel,
    CopulaCorrelationModel,
    CorrelationMethod
)


def example_1_options():
    """Example 1: Options and derivatives."""
    print("\n" + "="*60)
    print("Example 1: Options and Derivatives")
    print("="*60)

    # Create a call option
    call_option = OptionAsset(
        symbol="AAPL250117C00150000",
        name="AAPL Jan 2025 150 Call",
        option_type=OptionType.CALL,
        option_style=OptionStyle.AMERICAN,
        underlying_symbol="AAPL",
        strike_price=150.0,
        expiration_date=date(2025, 1, 17),
        contract_size=100,
        delta=0.65,
        gamma=0.05,
        theta=-0.15,
        vega=0.25,
        implied_volatility=0.30,
        bid=12.50,
        ask=12.75
    )

    print(f"\n1. Call Option Details:")
    print(f"   Symbol: {call_option.symbol}")
    print(f"   Strike: ${call_option.strike_price}")
    print(f"   Expiration: {call_option.expiration_date}")
    print(f"   Days to expiration: {call_option.days_to_expiration()}")

    # Calculate intrinsic and time value
    underlying_price = 155.00
    intrinsic = call_option.calculate_intrinsic_value(underlying_price)
    time_value = call_option.calculate_time_value(12.625, underlying_price)

    print(f"\n2. Option Valuation (underlying at ${underlying_price}):")
    print(f"   Intrinsic Value: ${intrinsic:.2f}")
    print(f"   Time Value: ${time_value:.2f}")
    print(f"   Moneyness: {call_option.moneyness(underlying_price)}")
    print(f"   In-the-money: {call_option.is_in_the_money(underlying_price)}")

    # Greeks
    print(f"\n3. Greeks:")
    print(f"   Delta: {call_option.delta:.2f} (1% move in stock = ${call_option.delta:.2f} move in option)")
    print(f"   Gamma: {call_option.gamma:.2f}")
    print(f"   Theta: {call_option.theta:.2f} (daily time decay)")
    print(f"   Vega: {call_option.vega:.2f} (1% vol change impact)")

    # Create a put option
    put_option = OptionAsset(
        symbol="AAPL250117P00150000",
        name="AAPL Jan 2025 150 Put",
        option_type=OptionType.PUT,
        option_style=OptionStyle.AMERICAN,
        underlying_symbol="AAPL",
        strike_price=150.0,
        expiration_date=date(2025, 1, 17),
        delta=-0.35,
        implied_volatility=0.32
    )

    print(f"\n4. Put Option:")
    print(f"   Strike: ${put_option.strike_price}")
    print(f"   Delta: {put_option.delta:.2f}")
    print(f"   Moneyness: {put_option.moneyness(underlying_price)}")


def example_2_reits():
    """Example 2: Real Estate Investment Trusts."""
    print("\n" + "="*60)
    print("Example 2: REITs")
    print("="*60)

    # Create an equity REIT
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
        total_properties=180,
        market_cap=75e9  # $75B
    )

    print(f"\n1. REIT Details:")
    print(f"   Name: {reit.name}")
    print(f"   Type: {reit.reit_type.value}")
    print(f"   Property Types: {', '.join(reit.property_types)}")
    print(f"   Total Properties: {reit.total_properties}")

    print(f"\n2. Financial Metrics:")
    print(f"   Dividend Yield: {reit.dividend_yield*100:.2f}%")
    print(f"   FFO per Share: ${reit.ffo_per_share:.2f}")
    print(f"   AFFO per Share: ${reit.affo_per_share:.2f}")
    print(f"   Price/FFO: {reit.price_to_ffo:.1f}x")

    print(f"\n3. Operational Metrics:")
    print(f"   Occupancy Rate: {reit.occupancy_rate*100:.1f}%")
    print(f"   Debt/Equity: {reit.debt_to_equity:.2f}")

    # Calculate REIT-specific metrics
    metrics = reit.calculate_reit_metrics()
    print(f"\n4. Calculated Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")


def example_3_crypto():
    """Example 3: Cryptocurrencies."""
    print("\n" + "="*60)
    print("Example 3: Cryptocurrencies")
    print("="*60)

    # Bitcoin
    btc = CryptoAsset(
        symbol="BTC-USD",
        name="Bitcoin",
        crypto_type=CryptoType.COIN,
        blockchain="Bitcoin",
        consensus_mechanism="Proof of Work",
        circulating_supply=19.5e6,
        max_supply=21e6,
        market_cap=650e9,  # $650B
        market_dominance=0.45,  # 45%
        trading_volume_24h=25e9,
        volatility_30d=0.55,  # 55% annualized
        hash_rate=350e18,  # 350 EH/s
        volatility_multiplier=1.5
    )

    # Create some mock price history for metrics
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    prices = 40000 * (1 + np.random.randn(365) * 0.03).cumprod()
    btc.price_history = pd.DataFrame({'close': prices}, index=dates)
    btc.calculate_metrics(risk_free_rate=0.04)

    print(f"\n1. Bitcoin Details:")
    print(f"   Blockchain: {btc.blockchain}")
    print(f"   Consensus: {btc.consensus_mechanism}")
    print(f"   Market Cap: ${btc.market_cap/1e9:.1f}B")
    print(f"   Market Dominance: {btc.market_dominance*100:.1f}%")

    print(f"\n2. Supply Metrics:")
    print(f"   Circulating: {btc.circulating_supply/1e6:.2f}M")
    print(f"   Max Supply: {btc.max_supply/1e6:.0f}M")
    print(f"   % of Max: {(btc.circulating_supply/btc.max_supply)*100:.1f}%")

    print(f"\n3. Volatility Analysis:")
    print(f"   30-day Volatility: {btc.volatility_30d*100:.1f}%")
    print(f"   Volatility Multiplier: {btc.volatility_multiplier}x")
    adjusted_vol = btc.calculate_adjusted_volatility()
    print(f"   Adjusted Volatility: {adjusted_vol*100:.1f}%")

    print(f"\n4. Risk Assessment:")
    risk_score = btc.calculate_risk_score()
    print(f"   Risk Score: {risk_score:.0f}/100")

    # Ethereum (DeFi token)
    eth = CryptoAsset(
        symbol="ETH-USD",
        name="Ethereum",
        crypto_type=CryptoType.COIN,
        blockchain="Ethereum",
        consensus_mechanism="Proof of Stake",
        circulating_supply=120e6,
        total_supply=120e6,
        max_supply=None,  # No max supply
        market_cap=250e9,
        staking_apy=0.045,  # 4.5%
        volatility_30d=0.65,
        volatility_multiplier=1.6
    )

    print(f"\n5. Ethereum (DeFi Platform):")
    print(f"   Consensus: {eth.consensus_mechanism}")
    print(f"   Staking APY: {eth.staking_apy*100:.2f}%")
    print(f"   Volatility Multiplier: {eth.volatility_multiplier}x")


def example_4_commodities():
    """Example 4: Commodities and Futures."""
    print("\n" + "="*60)
    print("Example 4: Commodities and Futures")
    print("="*60)

    # Gold commodity
    gold = CommodityAsset(
        symbol="GC=F",
        name="Gold",
        commodity_type=CommodityType.METALS,
        unit="troy ounce",
        spot_price=2050.00,
        futures_price=2065.00,
        storage_costs=0.005,  # 0.5% annual
        convenience_yield=0.001,  # 0.1%
        contango_backwardation="contango"
    )

    print(f"\n1. Gold Commodity:")
    print(f"   Spot Price: ${gold.spot_price:.2f}/oz")
    print(f"   Futures Price: ${gold.futures_price:.2f}/oz")
    print(f"   Market Structure: {gold.contango_backwardation}")
    print(f"   Is Contango: {gold.is_contango()}")

    # Calculate carry cost
    carry_cost = gold.calculate_carry_cost(risk_free_rate=0.04)
    print(f"\n2. Cost of Carry Analysis:")
    print(f"   Risk-free Rate: 4.0%")
    print(f"   Storage Costs: {gold.storage_costs*100:.2f}%")
    print(f"   Convenience Yield: {gold.convenience_yield*100:.2f}%")
    print(f"   Total Carry Cost: {carry_cost*100:.2f}%")

    # Oil futures contract
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

    print(f"\n3. Oil Futures Contract:")
    print(f"   Contract: {oil_future.contract_month}")
    print(f"   Size: {oil_future.contract_size:.0f} barrels")
    print(f"   Settlement: {oil_future.settlement_type}")

    # Calculate leverage
    current_price = 78.50
    contract_value = current_price * oil_future.contract_size
    leverage = oil_future.calculate_leverage(contract_value)

    print(f"\n4. Leverage Analysis:")
    print(f"   Current Price: ${current_price:.2f}/barrel")
    print(f"   Contract Value: ${contract_value:,.0f}")
    print(f"   Initial Margin: ${oil_future.initial_margin:,.0f}")
    print(f"   Leverage: {leverage:.1f}x")


def example_5_alternatives():
    """Example 5: Alternative investments."""
    print("\n" + "="*60)
    print("Example 5: Alternative Investments")
    print("="*60)

    # Private equity fund
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
        strategy="Growth equity in technology sector",
        sector_focus=["Software", "Cloud", "AI"]
    )

    print(f"\n1. Private Equity Fund:")
    print(f"   Name: {pe_fund.name}")
    print(f"   Vintage Year: {pe_fund.vintage_year}")
    print(f"   Strategy: {pe_fund.strategy}")
    print(f"   Sector Focus: {', '.join(pe_fund.sector_focus)}")

    print(f"\n2. Fee Structure:")
    print(f"   Management Fee: {pe_fund.management_fee*100:.1f}% annually")
    print(f"   Performance Fee: {pe_fund.performance_fee*100:.0f}% (carried interest)")
    print(f"   Hurdle Rate: {pe_fund.hurdle_rate*100:.0f}%")

    print(f"\n3. Performance Metrics:")
    print(f"   IRR: {pe_fund.irr*100:.1f}%")
    print(f"   MOIC: {pe_fund.moic:.1f}x")
    print(f"   TVPI: {pe_fund.tvpi:.1f}x")
    print(f"   DPI: {pe_fund.dpi:.1f}x (distributed)")
    print(f"   RVPI: {pe_fund.rvpi:.1f}x (remaining)")

    print(f"\n4. Liquidity:")
    print(f"   Lock-up Period: {pe_fund.lock_up_period} years")
    print(f"   Liquidity Tier: {pe_fund.liquidity_tier}")

    # Calculate net return
    gross_return = 0.30  # 30% gross return
    net_return = pe_fund.calculate_net_return(gross_return)
    print(f"\n5. Net Return Calculation:")
    print(f"   Gross Return: {gross_return*100:.1f}%")
    print(f"   Net Return (after fees): {net_return*100:.1f}%")
    print(f"   Fee Impact: {(gross_return - net_return)*100:.1f}%")


def example_6_structured_products():
    """Example 6: Structured products."""
    print("\n" + "="*60)
    print("Example 6: Structured Products")
    print("="*60)

    # Principal protected note
    structured = StructuredProduct(
        symbol="SPX-PP-2027",
        name="S&P 500 Principal Protected Note 2027",
        product_type=StructuredProductType.PRINCIPAL_PROTECTED,
        underlying_assets=["SPX"],
        protection_level=1.0,  # 100% principal protection
        participation_rate=0.75,  # 75% participation
        cap_level=0.40,  # 40% cap on returns
        maturity_date=date(2027, 12, 31),
        observation_frequency="at_maturity",
        issue_price=100.0,
        issuer="Major Bank",
        issuer_credit_rating="AA-"
    )

    print(f"\n1. Structured Product Details:")
    print(f"   Type: {structured.product_type.value}")
    print(f"   Underlying: {', '.join(structured.underlying_assets)}")
    print(f"   Maturity: {structured.maturity_date}")
    print(f"   Issuer: {structured.issuer} ({structured.issuer_credit_rating})")

    print(f"\n2. Terms:")
    print(f"   Principal Protection: {structured.protection_level*100:.0f}%")
    print(f"   Participation Rate: {structured.participation_rate*100:.0f}%")
    print(f"   Cap Level: {structured.cap_level*100:.0f}%")

    # Calculate payoffs for different scenarios
    print(f"\n3. Payoff Scenarios (at maturity):")

    scenarios = [
        ("Market down 30%", -0.30),
        ("Market flat", 0.00),
        ("Market up 20%", 0.20),
        ("Market up 40%", 0.40),
        ("Market up 60%", 0.60)
    ]

    for scenario_name, market_return in scenarios:
        payoff = structured.calculate_payoff(market_return, at_maturity=True)
        effective_return = (payoff / structured.issue_price) - 1

        print(f"   {scenario_name}:")
        print(f"     Payoff: ${payoff:.2f}")
        print(f"     Effective Return: {effective_return*100:+.1f}%")


def example_7_correlation_analysis():
    """Example 7: Asset class correlation models."""
    print("\n" + "="*60)
    print("Example 7: Asset Class Correlation Analysis")
    print("="*60)

    # Create correlation model
    corr_model = AssetClassCorrelationModel()

    # Get correlation between asset classes
    print(f"\n1. Default Asset Class Correlations:")
    pairs = [
        ("equity", "bond"),
        ("equity", "real_estate"),
        ("equity", "cryptocurrency"),
        ("bond", "commodity"),
        ("real_estate", "commodity")
    ]

    for ac1, ac2 in pairs:
        corr = corr_model.get_correlation(ac1, ac2)
        print(f"   {ac1.capitalize()} vs {ac2.capitalize()}: {corr:+.2f}")

    # Stress scenario correlations
    print(f"\n2. Stress Scenario Correlations:")
    for ac1, ac2 in pairs:
        normal_corr = corr_model.get_correlation(ac1, ac2, stress_mode=False)
        stress_corr = corr_model.get_correlation(ac1, ac2, stress_mode=True)
        change = stress_corr - normal_corr

        print(f"   {ac1.capitalize()} vs {ac2.capitalize()}:")
        print(f"     Normal: {normal_corr:+.2f}, Stress: {stress_corr:+.2f}, Change: {change:+.2f}")

    # Get full correlation matrix
    asset_classes = ["equity", "bond", "cash", "commodity", "real_estate", "cryptocurrency"]
    corr_matrix = corr_model.get_correlation_matrix(asset_classes)

    print(f"\n3. Full Correlation Matrix:")
    print(corr_matrix.round(2))

    # Simulate return data for advanced analysis
    print(f"\n4. Dynamic Correlation Analysis:")
    print("   (Using simulated data)")

    # Create simulated returns
    np.random.seed(42)
    n_days = 252 * 3  # 3 years
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    returns_data = {
        'Stocks': np.random.randn(n_days) * 0.01,
        'Bonds': np.random.randn(n_days) * 0.005,
        'Commodities': np.random.randn(n_days) * 0.015,
        'REITs': np.random.randn(n_days) * 0.012
    }
    returns = pd.DataFrame(returns_data, index=dates)

    # Dynamic correlation model
    dcc_model = DynamicCorrelationModel(returns)

    # Detect correlation regimes
    regimes = dcc_model.detect_correlation_regimes(window=60)
    print(f"\n   Correlation Regimes over time:")
    regime_counts = regimes['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = (count / len(regimes)) * 100
        print(f"     {regime.capitalize()}: {pct:.1f}% of the time")

    # Hierarchical clustering
    print(f"\n5. Asset Clustering (Hierarchical):")
    hier_model = HierarchicalCorrelationModel(returns)
    clusters = hier_model.get_clusters(n_clusters=2)

    print(f"   Clusters (2 groups):")
    for cluster_id in range(1, 3):
        assets_in_cluster = [asset for asset, cid in clusters.items() if cid == cluster_id]
        print(f"     Cluster {cluster_id}: {', '.join(assets_in_cluster)}")

    # Diversification score
    div_score = hier_model.get_diversification_score()
    print(f"\n   Portfolio Diversification Score: {div_score:.2f} (0-1, higher is better)")

    # Tail dependency analysis
    print(f"\n6. Tail Dependency Analysis:")
    copula_model = CopulaCorrelationModel(returns)

    # Identify contagion risks
    contagion_pairs = copula_model.identify_contagion_risk(threshold=0.4)
    if contagion_pairs:
        print(f"   High contagion risk pairs:")
        for asset1, asset2 in contagion_pairs:
            print(f"     {asset1} <-> {asset2}")
    else:
        print(f"   No high contagion risk pairs detected (threshold: 0.4)")


if __name__ == "__main__":
    """Run all examples."""

    print("\n" + "="*70)
    print(" Allocation Station - Enhanced Asset Classes Examples")
    print("="*70)

    try:
        example_1_options()
        example_2_reits()
        example_3_crypto()
        example_4_commodities()
        example_5_alternatives()
        example_6_structured_products()
        example_7_correlation_analysis()

        print("\n" + "="*70)
        print(" All examples completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
