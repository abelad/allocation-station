"""
Example demonstrating advanced portfolio features.

This example shows how to:
1. Implement tax-loss harvesting
2. Manage multi-currency portfolios
3. Analyze portfolio transitions
4. Use factor-based allocation
5. Manage leveraged portfolios
6. Implement CPPI portfolio insurance
"""

from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd

from allocation_station.portfolio.advanced_features import (
    TaxLot, TaxLossHarvestingStrategy,
    Currency, CurrencyExposure, MultiCurrencyPortfolio,
    TransitionCost, PortfolioTransition,
    Factor, FactorExposure, FactorAllocationStrategy,
    LeverageType, LeveragedPosition, LeveragedPortfolio,
    CPPIStrategy
)


def example_1_tax_loss_harvesting():
    """Example 1: Tax-loss harvesting."""
    print("\n" + "="*60)
    print("Example 1: Tax-Loss Harvesting")
    print("="*60)

    # Create tax lots
    lots = [
        TaxLot(
            asset_symbol="AAPL",
            purchase_date=date(2023, 1, 15),
            quantity=100,
            cost_basis=150.00,
            current_price=155.00  # Gain
        ),
        TaxLot(
            asset_symbol="MSFT",
            purchase_date=date(2023, 6, 1),
            quantity=50,
            cost_basis=350.00,
            current_price=330.00  # Loss
        ),
        TaxLot(
            asset_symbol="GOOGL",
            purchase_date=date(2024, 3, 10),
            quantity=75,
            cost_basis=140.00,
            current_price=125.00  # Loss
        ),
        TaxLot(
            asset_symbol="TSLA",
            purchase_date=date(2024, 8, 20),
            quantity=30,
            cost_basis=250.00,
            current_price=200.00  # Short-term loss
        )
    ]

    print("\n1. Tax Lot Summary:")
    for i, lot in enumerate(lots, 1):
        gain_loss = lot.calculate_gain_loss()
        holding_days = lot.get_holding_period_days()
        is_st = lot.is_short_term()

        print(f"\n   Lot {i}: {lot.asset_symbol}")
        print(f"     Purchase Date: {lot.purchase_date}")
        print(f"     Quantity: {lot.quantity}")
        print(f"     Cost Basis: ${lot.cost_basis:.2f}/share")
        print(f"     Current Price: ${lot.current_price:.2f}/share")
        print(f"     Gain/Loss: ${gain_loss:.2f}")
        print(f"     Holding Period: {holding_days} days ({'Short-term' if is_st else 'Long-term'})")

    # Initialize tax-loss harvesting strategy
    tlh_strategy = TaxLossHarvestingStrategy(
        short_term_rate=0.37,
        long_term_rate=0.20,
        wash_sale_days=30,
        min_loss_threshold=100.0
    )

    # Identify harvest opportunities
    print("\n2. Tax-Loss Harvesting Opportunities:")
    opportunities = tlh_strategy.identify_harvest_opportunities(lots)

    if opportunities:
        print(f"\n   Found {len(opportunities)} harvesting opportunities:")

        for opp in opportunities:
            print(f"\n   {opp['asset_symbol']}:")
            print(f"     Unrealized Loss: ${opp['unrealized_loss']:.2f}")
            print(f"     Tax Rate: {opp['tax_rate']*100:.0f}%")
            print(f"     Tax Benefit: ${opp['tax_benefit']:.2f}")
            print(f"     Term: {'Short' if opp['is_short_term'] else 'Long'}")

        total_tax_benefit = sum(opp['tax_benefit'] for opp in opportunities)
        print(f"\n   Total Potential Tax Benefit: ${total_tax_benefit:.2f}")
    else:
        print("   No harvesting opportunities found")

    # Suggest replacements
    print("\n3. Replacement Asset Suggestions:")
    for opp in opportunities[:2]:  # Show first 2
        replacements = tlh_strategy.suggest_replacement_assets(
            opp['asset_symbol'],
            ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO']
        )
        print(f"   For {opp['asset_symbol']}: {', '.join(replacements[:3])}")


def example_2_multi_currency():
    """Example 2: Multi-currency portfolio."""
    print("\n" + "="*60)
    print("Example 2: Multi-Currency Portfolio")
    print("="*60)

    # Create multi-currency portfolio
    fx_rates = {
        (Currency.EUR, Currency.USD): 1.10,
        (Currency.GBP, Currency.USD): 1.27,
        (Currency.JPY, Currency.USD): 0.0067,
        (Currency.CHF, Currency.USD): 1.12
    }

    mc_portfolio = MultiCurrencyPortfolio(
        base_currency=Currency.USD,
        fx_rates=fx_rates
    )

    # Add currency exposures
    exposures = [
        CurrencyExposure(
            base_currency=Currency.USD,
            position_currency=Currency.EUR,
            exposure_amount=100000,  # €100,000
            hedge_ratio=0.5  # 50% hedged
        ),
        CurrencyExposure(
            base_currency=Currency.USD,
            position_currency=Currency.GBP,
            exposure_amount=50000,  # £50,000
            hedge_ratio=0.0  # Unhedged
        ),
        CurrencyExposure(
            base_currency=Currency.USD,
            position_currency=Currency.JPY,
            exposure_amount=10000000,  # ¥10,000,000
            hedge_ratio=1.0  # Fully hedged
        )
    ]

    for exp in exposures:
        mc_portfolio.add_currency_exposure(exp)

    print("\n1. Currency Exposures:")
    for exp in exposures:
        base_value = mc_portfolio.convert_to_base(exp.exposure_amount, exp.position_currency)
        unhedged_pct = (1 - exp.hedge_ratio) * 100

        print(f"\n   {exp.position_currency.value}:")
        print(f"     Exposure: {exp.exposure_amount:,.0f} {exp.position_currency.value}")
        print(f"     USD Equivalent: ${base_value:,.2f}")
        print(f"     Hedge Ratio: {exp.hedge_ratio*100:.0f}%")
        print(f"     Unhedged: {unhedged_pct:.0f}%")

    # Calculate FX exposure
    print("\n2. Total FX Exposure by Currency (Unhedged Portion):")
    fx_exposure = mc_portfolio.calculate_fx_exposure()

    for currency, exposure in fx_exposure.items():
        print(f"   {currency.value}: ${exposure:,.2f}")

    total_fx_exposure = sum(fx_exposure.values())
    print(f"\n   Total FX Exposure: ${total_fx_exposure:,.2f}")

    # Calculate FX VaR
    print("\n3. FX Risk Assessment:")
    fx_var = mc_portfolio.calculate_fx_var(confidence_level=0.95, time_horizon_days=1)
    print(f"   1-Day VaR (95%): ${fx_var:,.2f}")

    fx_var_monthly = mc_portfolio.calculate_fx_var(confidence_level=0.95, time_horizon_days=21)
    print(f"   Monthly VaR (95%): ${fx_var_monthly:,.2f}")


def example_3_portfolio_transition():
    """Example 3: Portfolio transition analysis."""
    print("\n" + "="*60)
    print("Example 3: Portfolio Transition Analysis")
    print("="*60)

    # Current portfolio
    current_holdings = {
        'AAPL': 100,
        'MSFT': 50,
        'GOOGL': 75,
        'AMZN': 20
    }

    # Target portfolio
    target_holdings = {
        'AAPL': 80,
        'MSFT': 60,
        'GOOGL': 50,
        'AMZN': 30,
        'NVDA': 40  # New position
    }

    # Current prices
    current_prices = {
        'AAPL': 155.00,
        'MSFT': 330.00,
        'GOOGL': 125.00,
        'AMZN': 145.00,
        'NVDA': 450.00
    }

    print("\n1. Current vs Target Holdings:")
    all_symbols = set(list(current_holdings.keys()) + list(target_holdings.keys()))

    for symbol in sorted(all_symbols):
        current = current_holdings.get(symbol, 0)
        target = target_holdings.get(symbol, 0)
        change = target - current
        price = current_prices.get(symbol, 0)

        print(f"   {symbol}:")
        print(f"     Current: {current:,.0f} shares")
        print(f"     Target: {target:,.0f} shares")
        print(f"     Change: {change:+,.0f} shares (${change * price:+,.2f})")

    # Analyze transition
    transition_costs = TransitionCost(
        commission_rate=0.001,
        bid_ask_spread=0.002,
        market_impact=0.001,
        tax_cost=0.0
    )

    transition = PortfolioTransition(transition_costs=transition_costs)

    plan = transition.calculate_transition_plan(
        current_holdings=current_holdings,
        target_holdings=target_holdings,
        current_prices=current_prices
    )

    print("\n2. Transition Plan:")
    print(f"   Total Trade Value: ${plan['total_trade_value']:,.2f}")

    print("\n3. Transaction Costs:")
    for cost_type, amount in plan['costs'].items():
        print(f"   {cost_type.replace('_', ' ').title()}: ${amount:,.2f}")

    print(f"\n   Total Cost: ${plan['net_transition_cost']:,.2f}")
    print(f"   Cost as % of Trade Value: {plan['cost_as_pct']*100:.2f}%")

    # Optimize schedule
    print("\n4. Recommended Execution Schedule (5 days):")
    schedule = transition.optimize_transition_schedule(plan['trades'], n_days=5)

    for day_plan in schedule[:2]:  # Show first 2 days
        print(f"\n   Day {day_plan['day']}:")
        for symbol, trade_info in day_plan['trades'].items():
            if abs(trade_info['quantity']) > 0.01:
                print(f"     {symbol}: {trade_info['direction'].upper()} {abs(trade_info['quantity']):.1f} shares")


def example_4_factor_allocation():
    """Example 4: Factor-based allocation."""
    print("\n" + "="*60)
    print("Example 4: Factor-Based Allocation")
    print("="*60)

    # Define factor exposures for assets
    factor_exposures = {
        'VTV': FactorExposure(  # Value ETF
            asset_symbol='VTV',
            factor_loadings={
                Factor.VALUE: 0.90,
                Factor.MOMENTUM: -0.20,
                Factor.QUALITY: 0.30,
                Factor.SIZE: 0.50,
                Factor.LOW_VOLATILITY: 0.40
            }
        ),
        'MTUM': FactorExposure(  # Momentum ETF
            asset_symbol='MTUM',
            factor_loadings={
                Factor.VALUE: -0.30,
                Factor.MOMENTUM: 0.95,
                Factor.QUALITY: 0.50,
                Factor.SIZE: 0.20,
                Factor.LOW_VOLATILITY: -0.10
            }
        ),
        'QUAL': FactorExposure(  # Quality ETF
            asset_symbol='QUAL',
            factor_loadings={
                Factor.VALUE: 0.10,
                Factor.MOMENTUM: 0.30,
                Factor.QUALITY: 0.90,
                Factor.SIZE: 0.00,
                Factor.LOW_VOLATILITY: 0.50
            }
        ),
        'IWM': FactorExposure(  # Small Cap ETF
            asset_symbol='IWM',
            factor_loadings={
                Factor.VALUE: 0.20,
                Factor.MOMENTUM: 0.10,
                Factor.QUALITY: -0.20,
                Factor.SIZE: 0.90,
                Factor.LOW_VOLATILITY: -0.30
            }
        )
    }

    # Define target factors
    target_factors = {
        Factor.VALUE: 0.50,
        Factor.MOMENTUM: 0.40,
        Factor.QUALITY: 0.60,
        Factor.SIZE: 0.20,
        Factor.LOW_VOLATILITY: 0.30
    }

    print("\n1. Target Factor Exposures:")
    for factor, target in target_factors.items():
        print(f"   {factor.value.replace('_', ' ').title()}: {target:.2f}")

    # Create strategy
    factor_strategy = FactorAllocationStrategy(
        target_factors=target_factors,
        factor_exposures=factor_exposures
    )

    # Example portfolio
    weights = {
        'VTV': 0.30,
        'MTUM': 0.25,
        'QUAL': 0.25,
        'IWM': 0.20
    }

    # Calculate portfolio factors
    portfolio_factors = factor_strategy.calculate_portfolio_factors(weights)

    print("\n2. Current Portfolio Factor Exposures:")
    for factor, exposure in portfolio_factors.items():
        target = target_factors.get(factor, 0)
        diff = exposure - target

        print(f"   {factor.value.replace('_', ' ').title()}:")
        print(f"     Current: {exposure:.2f}")
        print(f"     Target: {target:.2f}")
        print(f"     Difference: {diff:+.2f}")


def example_5_leveraged_portfolio():
    """Example 5: Leveraged portfolio management."""
    print("\n" + "="*60)
    print("Example 5: Leveraged Portfolio")
    print("="*60)

    # Create leveraged portfolio
    lev_portfolio = LeveragedPortfolio(
        max_leverage=2.0,
        margin_call_threshold=0.25
    )

    # Add leveraged positions
    positions = [
        LeveragedPosition(
            asset_symbol='SPY',
            position_value=100000,
            leverage_ratio=1.5,
            leverage_type=LeverageType.MARGIN,
            margin_requirement=50000,
            interest_rate=0.05  # 5% margin interest
        ),
        LeveragedPosition(
            asset_symbol='ES',  # S&P 500 Futures
            position_value=50000,
            leverage_ratio=10.0,
            leverage_type=LeverageType.FUTURES,
            margin_requirement=5000
        )
    ]

    for pos in positions:
        lev_portfolio.add_leveraged_position(pos)

    print("\n1. Leveraged Positions:")
    for i, pos in enumerate(lev_portfolio.leveraged_positions, 1):
        exposure = pos.calculate_exposure()
        margin_call_level = pos.calculate_margin_call_level()

        print(f"\n   Position {i}: {pos.asset_symbol}")
        print(f"     Position Value: ${pos.position_value:,.2f}")
        print(f"     Leverage Ratio: {pos.leverage_ratio:.1f}x")
        print(f"     Total Exposure: ${exposure:,.2f}")
        print(f"     Leverage Type: {pos.leverage_type.value}")
        print(f"     Margin Requirement: ${pos.margin_requirement:,.2f}")
        if pos.interest_rate:
            print(f"     Interest Rate: {pos.interest_rate*100:.1f}%")
        print(f"     Margin Call Level: ${margin_call_level:,.2f}")

    # Portfolio-level metrics
    print("\n2. Portfolio-Level Leverage:")
    total_leverage = lev_portfolio.calculate_total_leverage()
    total_margin = lev_portfolio.calculate_margin_requirements()
    leverage_costs = lev_portfolio.calculate_leverage_costs()

    print(f"   Total Leverage Ratio: {total_leverage:.2f}x")
    print(f"   Total Margin Requirements: ${total_margin:,.2f}")
    print(f"   Annual Leverage Costs: ${leverage_costs:,.2f}")

    # Risk analysis
    print("\n3. Leverage Risk Analysis:")
    max_drawdown_pct = 0.20  # 20% market decline
    total_exposure = sum(pos.calculate_exposure() for pos in lev_portfolio.leveraged_positions)
    potential_loss = total_exposure * max_drawdown_pct

    print(f"   Total Exposure: ${total_exposure:,.2f}")
    print(f"   Potential Loss (20% decline): ${potential_loss:,.2f}")
    print(f"   Loss vs Equity: {(potential_loss / sum(pos.position_value for pos in lev_portfolio.leveraged_positions))*100:.1f}%")


def example_6_cppi_insurance():
    """Example 6: CPPI portfolio insurance."""
    print("\n" + "="*60)
    print("Example 6: CPPI Portfolio Insurance")
    print("="*60)

    # Initialize CPPI strategy
    initial_value = 1000000  # $1M
    floor_value = 800000      # $800K floor (80% protection)

    cppi = CPPIStrategy(
        floor_value=floor_value,
        multiplier=3.0,
        risky_asset_return=0.08,
        safe_asset_return=0.02
    )

    print(f"\n1. CPPI Strategy Parameters:")
    print(f"   Initial Value: ${initial_value:,.2f}")
    print(f"   Floor Value: ${floor_value:,.2f} ({(floor_value/initial_value)*100:.0f}% protection)")
    print(f"   Multiplier: {cppi.multiplier:.1f}x")
    print(f"   Risky Asset Return: {cppi.risky_asset_return*100:.1f}%")
    print(f"   Safe Asset Return: {cppi.safe_asset_return*100:.1f}%")

    # Calculate allocations for different portfolio values
    print("\n2. Dynamic Allocation Examples:")

    scenarios = [
        ("Current (at start)", initial_value),
        ("After 10% gain", initial_value * 1.10),
        ("After 10% loss", initial_value * 0.90),
        ("After 20% loss", initial_value * 0.80),
        ("Near floor", floor_value * 1.05)
    ]

    for scenario_name, value in scenarios:
        risky_amt, safe_amt = cppi.calculate_allocation(value)
        risky_pct, safe_pct = cppi.calculate_allocation_pct(value)

        print(f"\n   {scenario_name} (${value:,.0f}):")
        print(f"     Risky: ${risky_amt:,.0f} ({risky_pct*100:.1f}%)")
        print(f"     Safe: ${safe_amt:,.0f} ({safe_pct*100:.1f}%)")
        print(f"     Cushion: ${(value - floor_value):,.0f}")

    # Simulate CPPI path
    print("\n3. Monte Carlo Simulation (100 paths, 1 year):")
    results_list = []

    for sim in range(100):
        simulation = cppi.simulate_path(
            initial_value=initial_value,
            n_periods=252,  # 1 year
            risky_vol=0.20
        )
        final_value = simulation['value'].iloc[-1]
        results_list.append(final_value)

    results_array = np.array(results_list)

    print(f"   Mean Final Value: ${results_array.mean():,.2f}")
    print(f"   Median Final Value: ${np.median(results_array):,.2f}")
    print(f"   5th Percentile: ${np.percentile(results_array, 5):,.2f}")
    print(f"   95th Percentile: ${np.percentile(results_array, 95):,.2f}")
    print(f"   % Above Floor: {(results_array > floor_value).sum() / len(results_array) * 100:.1f}%")
    print(f"   % Above Initial: {(results_array > initial_value).sum() / len(results_array) * 100:.1f}%")


if __name__ == "__main__":
    """Run all examples."""

    print("\n" + "="*70)
    print(" Allocation Station - Advanced Portfolio Features Examples")
    print("="*70)

    try:
        example_1_tax_loss_harvesting()
        example_2_multi_currency()
        example_3_portfolio_transition()
        example_4_factor_allocation()
        example_5_leveraged_portfolio()
        example_6_cppi_insurance()

        print("\n" + "="*70)
        print(" All examples completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
