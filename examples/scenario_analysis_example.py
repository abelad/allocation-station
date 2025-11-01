"""
Scenario Analysis Examples

This script demonstrates the scenario analysis capabilities of Allocation Station:
1. Historical scenario replay (2008 crisis, COVID-19, etc.)
2. Custom scenario builder with user-defined shocks
3. Economic scenario generators (recession, inflation, recovery)
4. What-if analysis tools for portfolio testing
5. Sensitivity analysis with tornado charts
6. Parametric scenario testing across parameter spaces
7. Scenario comparison framework for robust portfolios

Run this script to see each scenario analysis tool in action.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from allocation_station.simulation.scenario_analysis import (
    HistoricalScenarioReplay,
    HistoricalEvent,
    CustomScenarioBuilder,
    ScenarioParameter,
    EconomicScenarioGenerator,
    EconomicScenarioType,
    WhatIfAnalyzer,
    SensitivityAnalyzer,
    ParametricScenarioTester,
    ScenarioComparisonFramework,
)


def create_sample_portfolio():
    """Create a sample diversified portfolio for testing."""
    portfolio_weights = pd.Series({
        'US_Equity': 0.30,
        'International_Equity': 0.15,
        'Emerging_Markets': 0.05,
        'Government_Bonds': 0.20,
        'Corporate_Bonds': 0.10,
        'Real_Estate': 0.10,
        'Commodities': 0.05,
        'Gold': 0.03,
        'Cash': 0.02,
    })

    expected_returns = pd.Series({
        'US_Equity': 0.08,
        'International_Equity': 0.07,
        'Emerging_Markets': 0.10,
        'Government_Bonds': 0.03,
        'Corporate_Bonds': 0.04,
        'Real_Estate': 0.06,
        'Commodities': 0.05,
        'Gold': 0.02,
        'Cash': 0.01,
    })

    return portfolio_weights, expected_returns


def example_1_historical_scenarios():
    """Example 1: Replay historical market scenarios."""
    print("=" * 80)
    print("EXAMPLE 1: Historical Scenario Replay")
    print("=" * 80)

    portfolio_weights, _ = create_sample_portfolio()

    print("\nPortfolio Allocation:")
    for asset, weight in portfolio_weights.items():
        print(f"  {asset:20s}: {weight:6.1%}")

    # Initialize historical scenario replay
    historical = HistoricalScenarioReplay()

    # Test multiple historical events
    scenarios_to_test = [
        (HistoricalEvent.FINANCIAL_CRISIS_2008, 1.0),
        (HistoricalEvent.COVID_PANDEMIC, 1.0),
        (HistoricalEvent.BLACK_MONDAY_1987, 1.0),
        (HistoricalEvent.DOT_COM_CRASH, 0.5),  # 50% severity
    ]

    print("\n--- Testing Historical Scenarios ---")

    for event, scale in scenarios_to_test:
        # Get scenario details
        details = historical.get_scenario_details(event)

        print(f"\n{event.value.upper()}:")
        print(f"  Description: {details['description']}")
        print(f"  Duration: {details['duration_days']} days")
        print(f"  Scale Factor: {scale:.0%}")

        # Apply scenario
        impact = historical.replay_scenario(portfolio_weights, event, scale)

        print(f"\n  Portfolio Impact:")
        print(f"    Initial Value: ${impact.initial_value:,.0f}")
        print(f"    Final Value:   ${impact.final_value:,.0f}")
        print(f"    Change:        {impact.percentage_change:7.2%}")

        # Show worst affected assets
        print(f"\n  Asset Impacts (worst 3):")
        asset_impacts_sorted = sorted(impact.asset_impacts.items(), key=lambda x: x[1])
        for asset, return_impact in asset_impacts_sorted[:3]:
            print(f"    {asset:20s}: {return_impact:7.2%}")

        if impact.risk_metrics:
            print(f"\n  Risk Metrics:")
            print(f"    Volatility Multiple: {impact.risk_metrics.get('volatility_multiplier', 'N/A'):.1f}x")
            print(f"    Correlation Increase: {impact.risk_metrics.get('correlation_increase', 0):.0%}")


def example_2_custom_scenarios():
    """Example 2: Build custom scenarios."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Scenario Builder")
    print("=" * 80)

    portfolio_weights, _ = create_sample_portfolio()

    # Initialize custom scenario builder
    builder = CustomScenarioBuilder()

    # Create custom scenarios
    print("\n--- Creating Custom Scenarios ---")

    # Scenario 1: Interest rate spike
    print("\n1. Interest Rate Spike Scenario")
    builder.create_scenario(
        name="rate_spike",
        description="Central bank raises rates aggressively",
        asset_shocks={
            'US_Equity': -0.15,
            'International_Equity': -0.12,
            'Emerging_Markets': -0.20,
            'Government_Bonds': -0.10,
            'Corporate_Bonds': -0.15,
            'Real_Estate': -0.25,
            'Commodities': -0.05,
            'Gold': 0.05,
            'Cash': 0.02,
        },
        duration_periods=3,
        parameters=[
            ScenarioParameter(
                name="interest_rate",
                base_value=0.02,
                scenario_value=0.05,
                transition_type="linear",
                transition_periods=3,
            ),
        ],
    )

    impact = builder.apply_scenario(portfolio_weights, "rate_spike", 1000000)

    print(f"  Initial Value: ${impact.initial_value:,.0f}")
    print(f"  Final Value:   ${impact.final_value:,.0f}")
    print(f"  Impact:        {impact.percentage_change:.2%}")

    # Scenario 2: Technology bubble burst
    print("\n2. Technology Bubble Burst")
    builder.create_scenario(
        name="tech_bubble",
        description="Technology sector correction",
        asset_shocks={
            'US_Equity': -0.35,  # Heavy tech weight in US
            'International_Equity': -0.20,
            'Emerging_Markets': -0.25,
            'Government_Bonds': 0.08,
            'Corporate_Bonds': -0.05,
            'Real_Estate': -0.10,
            'Commodities': 0.00,
            'Gold': 0.10,
            'Cash': 0.00,
        },
        duration_periods=6,
    )

    impact = builder.apply_scenario(portfolio_weights, "tech_bubble", 1000000)
    print(f"  Initial Value: ${impact.initial_value:,.0f}")
    print(f"  Final Value:   ${impact.final_value:,.0f}")
    print(f"  Impact:        {impact.percentage_change:.2%}")

    # Combine scenarios
    print("\n3. Combined Scenario (50% rate spike + 50% tech bubble)")
    combined = builder.combine_scenarios(
        ["rate_spike", "tech_bubble"],
        weights=[0.5, 0.5],
        name="combined_stress"
    )

    impact = builder.apply_scenario(portfolio_weights, "combined_stress", 1000000)
    print(f"  Initial Value: ${impact.initial_value:,.0f}")
    print(f"  Final Value:   ${impact.final_value:,.0f}")
    print(f"  Impact:        {impact.percentage_change:.2%}")


def example_3_economic_scenarios():
    """Example 3: Economic scenario generation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Economic Scenario Generators")
    print("=" * 80)

    portfolio_weights, _ = create_sample_portfolio()

    # Initialize economic scenario generator
    econ_gen = EconomicScenarioGenerator()

    # Test different economic scenarios
    scenarios = [
        (EconomicScenarioType.RECESSION, 1.0),
        (EconomicScenarioType.INFLATION, 1.5),
        (EconomicScenarioType.STAGFLATION, 1.0),
        (EconomicScenarioType.RECOVERY, 1.0),
        (EconomicScenarioType.DEFLATION, 0.5),
    ]

    print("\n--- Economic Scenarios ---")

    for scenario_type, severity in scenarios:
        print(f"\n{scenario_type.value.upper()} (Severity: {severity})")

        # Generate scenario
        scenario = econ_gen.generate_scenario(scenario_type, severity, duration_quarters=4)

        # Show macro indicators
        print(f"  Macro Indicators:")
        print(f"    GDP Growth:        {scenario['macro_indicators']['gdp_growth']:7.2%}")
        print(f"    Inflation:         {scenario['macro_indicators']['inflation']:7.2%}")
        print(f"    Unemployment Delta:    {scenario['macro_indicators']['unemployment_change']:+7.2%}")
        print(f"    Interest Rate Delta:   {scenario['macro_indicators']['interest_rate_change']:+7.2%}")

        # Apply to portfolio
        impact = econ_gen.apply_to_portfolio(portfolio_weights, scenario_type, severity)

        print(f"\n  Portfolio Impact:")
        print(f"    Expected Change: {impact.percentage_change:7.2%}")

        # Show asset class impacts
        print(f"\n  Asset Class Impacts:")
        for asset_class, impact_pct in scenario['asset_impacts'].items():
            print(f"    {asset_class:15s}: {impact_pct:7.2%}")


def example_4_what_if_analysis():
    """Example 4: What-if analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: What-If Analysis")
    print("=" * 80)

    portfolio_weights, expected_returns = create_sample_portfolio()

    # Initialize what-if analyzer
    what_if = WhatIfAnalyzer()

    print("\nBase Portfolio Return: {:.2%}".format(
        (portfolio_weights * expected_returns).sum()
    ))

    # 1. Single parameter change
    print("\n--- Single Parameter Changes ---")

    changes_to_test = [
        ('US_Equity', -0.05, 'absolute', "US equity drops 5%"),
        ('Government_Bonds', 0.02, 'absolute', "Bond yields rise 2%"),
        ('Emerging_Markets', 0.50, 'relative', "EM rallies 50%"),
    ]

    for param, change, change_type, description in changes_to_test:
        result = what_if.analyze_single_change(
            portfolio_weights,
            expected_returns,
            param,
            change,
            change_type,
        )

        print(f"\nWhat if: {description}")
        print(f"  Parameter: {param}")
        print(f"  Change: {change:.2%} ({change_type})")
        print(f"  New Portfolio Return: {result['new_return']:.2%}")
        print(f"  Impact: {result['impact']:.2%}")

    # 2. Multiple simultaneous changes
    print("\n--- Multiple Simultaneous Changes ---")

    multi_changes = [
        ('US_Equity', -0.10, 'absolute'),
        ('Government_Bonds', 0.05, 'absolute'),
        ('Gold', 0.20, 'relative'),
    ]

    print("\nWhat if multiple changes occur:")
    for param, change, change_type in multi_changes:
        print(f"  - {param}: {change:.2%} ({change_type})")

    multi_result = what_if.analyze_multiple_changes(
        portfolio_weights,
        expected_returns,
        multi_changes,
    )

    print(f"\nResults:")
    print(f"  Base Return:        {multi_result['base_return']:.2%}")
    print(f"  New Return:         {multi_result['new_return']:.2%}")
    print(f"  Total Impact:       {multi_result['total_impact']:.2%}")
    print(f"  Sum of Individual:  {sum(multi_result['individual_impacts']):.2%}")
    print(f"  Interaction Effect: {multi_result['interaction_effect']:.4%}")

    # 3. Rebalancing analysis
    print("\n--- Rebalancing Analysis ---")

    # Target weights (more defensive)
    target_weights = portfolio_weights.copy()
    target_weights['US_Equity'] = 0.20  # Reduce equity
    target_weights['Government_Bonds'] = 0.30  # Increase bonds
    target_weights = target_weights / target_weights.sum()  # Normalize

    rebalance_result = what_if.analyze_rebalancing(
        portfolio_weights,
        target_weights,
        expected_returns,
        transaction_cost=0.002,  # 20 bps
    )

    print("\nRebalancing to more defensive allocation:")
    print(f"  Current Return:     {rebalance_result['current_return']:.2%}")
    print(f"  Target Return:      {rebalance_result['target_return']:.2%}")
    print(f"  Turnover:           {rebalance_result['turnover']:.1%}")
    print(f"  Transaction Cost:   {rebalance_result['transaction_cost']:.2%}")
    print(f"  Net Benefit:        {rebalance_result['net_benefit']:.2%}")
    print(f"  Breakeven (years):  {rebalance_result['breakeven_periods']:.1f}")


def example_5_sensitivity_analysis():
    """Example 5: Sensitivity analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Sensitivity Analysis")
    print("=" * 80)

    portfolio_weights, expected_returns = create_sample_portfolio()

    # Initialize sensitivity analyzer
    sensitivity = SensitivityAnalyzer()

    # Define evaluation function
    def portfolio_return_function(equity_return):
        """Portfolio return as function of equity return."""
        modified_returns = expected_returns.copy()
        modified_returns['US_Equity'] = equity_return
        modified_returns['International_Equity'] = equity_return * 0.9
        return (portfolio_weights * modified_returns).sum()

    # 1. Single parameter sensitivity
    print("\n--- Single Parameter Sensitivity (Equity Return) ---")

    base_equity_return = expected_returns['US_Equity']
    sensitivity_df = sensitivity.analyze_parameter_sensitivity(
        base_value=base_equity_return,
        parameter_name="equity_return",
        parameter_range=(-0.20, 0.30),
        n_steps=11,
        evaluation_function=portfolio_return_function,
    )

    print("\nSensitivity Analysis Results:")
    print(sensitivity_df[['value', 'outcome', 'outcome_change']].head(11))

    print(f"\nSensitivity Coefficient: {sensitivity.results['equity_return']['sensitivity']:.4f}")
    print("  (Portfolio return changes by this amount per unit change in equity return)")

    # 2. Multi-parameter sensitivity
    print("\n--- Multi-Parameter Sensitivity ---")

    def multi_param_function(params):
        """Portfolio return with multiple parameters."""
        modified_returns = expected_returns.copy()
        modified_returns['US_Equity'] = params.get('equity_return', 0.08)
        modified_returns['Government_Bonds'] = params.get('bond_return', 0.03)
        modified_returns['Real_Estate'] = params.get('reit_return', 0.06)
        return (portfolio_weights * modified_returns).sum()

    parameters = {
        'equity_return': (-0.20, 0.30),
        'bond_return': (0.00, 0.06),
        'reit_return': (-0.10, 0.15),
    }

    multi_sensitivity_df = sensitivity.analyze_multi_parameter_sensitivity(
        parameters,
        n_steps=5,  # 5x5x5 = 125 combinations
        evaluation_function=multi_param_function,
    )

    print(f"\nTested {len(multi_sensitivity_df)} parameter combinations")

    # Find best and worst outcomes
    best_idx = multi_sensitivity_df['outcome'].idxmax()
    worst_idx = multi_sensitivity_df['outcome'].idxmin()

    print(f"\nBest Outcome: {multi_sensitivity_df.loc[best_idx, 'outcome']:.2%}")
    print(f"  Equity Return: {multi_sensitivity_df.loc[best_idx, 'equity_return']:.2%}")
    print(f"  Bond Return:   {multi_sensitivity_df.loc[best_idx, 'bond_return']:.2%}")
    print(f"  REIT Return:   {multi_sensitivity_df.loc[best_idx, 'reit_return']:.2%}")

    print(f"\nWorst Outcome: {multi_sensitivity_df.loc[worst_idx, 'outcome']:.2%}")
    print(f"  Equity Return: {multi_sensitivity_df.loc[worst_idx, 'equity_return']:.2%}")
    print(f"  Bond Return:   {multi_sensitivity_df.loc[worst_idx, 'bond_return']:.2%}")
    print(f"  REIT Return:   {multi_sensitivity_df.loc[worst_idx, 'reit_return']:.2%}")

    # 3. Tornado chart data
    print("\n--- Tornado Chart Data ---")

    base_outcome = (portfolio_weights * expected_returns).sum()
    sensitivities = {
        'equity_return': 0.45,  # From earlier analysis
        'bond_return': 0.20,
        'reit_return': 0.10,
        'inflation': -0.30,
    }

    param_ranges = {
        'equity_return': (-0.20, 0.30),
        'bond_return': (0.00, 0.06),
        'reit_return': (-0.10, 0.15),
        'inflation': (0.00, 0.06),
    }

    tornado_data = sensitivity.calculate_tornado_chart_data(
        base_outcome, sensitivities, param_ranges
    )

    print("\nTornado Chart Data (sorted by impact):")
    print(tornado_data[['parameter', 'outcome_min', 'outcome_max', 'range']])


def example_6_parametric_testing():
    """Example 6: Parametric scenario testing."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Parametric Scenario Testing")
    print("=" * 80)

    # Initialize parametric tester
    tester = ParametricScenarioTester()

    # Define portfolio evaluation function
    def evaluate_portfolio(equity_weight, bond_weight, volatility_target):
        """Evaluate portfolio with given parameters."""
        # Simple 2-asset portfolio
        expected_return = equity_weight * 0.08 + bond_weight * 0.03
        volatility = np.sqrt(
            (equity_weight**2 * 0.20**2) +
            (bond_weight**2 * 0.05**2) +
            (2 * equity_weight * bond_weight * 0.20 * 0.05 * 0.2)  # correlation = 0.2
        )

        # Penalty for missing volatility target
        volatility_penalty = abs(volatility - volatility_target) * 0.1

        sharpe = (expected_return - 0.02) / volatility if volatility > 0 else 0

        return {
            'return': expected_return,
            'volatility': volatility,
            'sharpe': sharpe - volatility_penalty,
        }

    # Define parameter grid
    print("\n--- Parametric Test Setup ---")
    parameters = {
        'equity_weight': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'bond_weight': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'volatility_target': [0.05, 0.10, 0.15],
    }

    print("Parameter Grid:")
    for param, values in parameters.items():
        print(f"  {param}: {values}")

    # Run parametric test
    test_results = tester.run_parametric_test(
        evaluate_portfolio,
        parameters,
        metrics_to_calculate=['return', 'volatility', 'sharpe'],
    )

    # Filter valid portfolios (weights sum to 1)
    valid_results = test_results[
        np.abs(test_results['equity_weight'] + test_results['bond_weight'] - 1.0) < 0.001
    ]

    print(f"\nTested {len(test_results)} combinations")
    print(f"Valid portfolios (weights sum to 1): {len(valid_results)}")

    # Find optimal parameters
    print("\n--- Finding Optimal Parameters ---")

    # Constraint function
    def weight_constraint(row):
        return abs(row['equity_weight'] + row['bond_weight'] - 1.0) < 0.001

    # Find best Sharpe ratio
    optimal = tester.find_optimal_parameters(
        test_results,
        objective_metric='sharpe',
        constraints=[weight_constraint],
        maximize=True,
    )

    if optimal['status'] == 'optimal':
        print(f"\nOptimal Portfolio (Max Sharpe):")
        print(f"  Equity Weight:     {optimal['parameters']['equity_weight']:.0%}")
        print(f"  Bond Weight:       {optimal['parameters']['bond_weight']:.0%}")
        print(f"  Volatility Target: {optimal['parameters']['volatility_target']:.0%}")
        print(f"\n  Achieved Metrics:")
        print(f"    Return:     {optimal['all_metrics']['return']:.2%}")
        print(f"    Volatility: {optimal['all_metrics']['volatility']:.2%}")
        print(f"    Sharpe:     {optimal['all_metrics']['sharpe']:.3f}")


def example_7_scenario_comparison():
    """Example 7: Scenario comparison framework."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Scenario Comparison Framework")
    print("=" * 80)

    portfolio_weights, _ = create_sample_portfolio()

    # Initialize comparison framework
    comparison = ScenarioComparisonFramework()

    # Create multiple scenarios to compare
    print("\n--- Creating Scenarios for Comparison ---")

    # Historical scenarios
    historical = HistoricalScenarioReplay()

    # Add scenarios with probabilities
    scenarios_to_add = [
        ('Base Case', 0.00, 0.40),  # 40% probability
        ('Mild Recession', -0.15, 0.25),
        ('Severe Recession', -0.35, 0.10),
        ('Recovery', 0.20, 0.15),
        ('Stagflation', -0.10, 0.10),
    ]

    for name, portfolio_return, probability in scenarios_to_add:
        # Create simple impact
        from allocation_station.simulation.scenario_analysis import ScenarioImpact

        impact = ScenarioImpact(
            scenario_name=name,
            initial_value=1000000,
            final_value=1000000 * (1 + portfolio_return),
            absolute_change=1000000 * portfolio_return,
            percentage_change=portfolio_return,
            asset_impacts={asset: portfolio_return for asset in portfolio_weights.index},
        )

        comparison.add_scenario(name, impact, probability)

    # Compare scenarios
    print("\n--- Scenario Comparison ---")
    comparison_df = comparison.compare_scenarios()
    print(comparison_df)

    # Portfolio statistics
    print("\n--- Portfolio Statistics Across Scenarios ---")
    stats = comparison.calculate_portfolio_statistics()

    print(f"  Expected Return:       {stats['expected_return']:7.2%}")
    print(f"  Standard Deviation:    {stats['std_deviation']:7.2%}")
    print(f"  Downside Deviation:    {stats['downside_deviation']:7.2%}")
    print(f"  Value at Risk (5%):    {stats['var_5']:7.2%}")
    print(f"  Conditional VaR (5%):  {stats['cvar_5']:7.2%}")
    print(f"  Best Scenario:         {stats['best_scenario']:7.2%}")
    print(f"  Worst Scenario:        {stats['worst_scenario']:7.2%}")
    print(f"  Range:                 {stats['scenario_range']:7.2%}")

    # Rank scenarios
    print("\n--- Scenario Ranking (by impact) ---")
    ranked = comparison.rank_scenarios('percentage_change', ascending=True)
    print(ranked[['rank', 'scenario', 'percentage_change', 'probability']])

    # Test portfolio robustness
    print("\n--- Portfolio Robustness Test ---")

    portfolio_options = {
        'Aggressive': pd.Series({
            'US_Equity': 0.60,
            'International_Equity': 0.30,
            'Bonds': 0.10,
        }),
        'Balanced': pd.Series({
            'US_Equity': 0.40,
            'Bonds': 0.40,
            'Real_Estate': 0.20,
        }),
        'Conservative': pd.Series({
            'US_Equity': 0.20,
            'Bonds': 0.60,
            'Cash': 0.20,
        }),
    }

    # Add simple asset impacts for comparison
    for scenario_name in comparison.scenarios:
        scenario_data = comparison.scenarios[scenario_name]
        impact = scenario_data['impact']
        # Update with more realistic asset-specific impacts
        if 'Recession' in scenario_name:
            impact.asset_impacts = {
                'US_Equity': impact.percentage_change * 1.5,
                'International_Equity': impact.percentage_change * 1.3,
                'Bonds': -impact.percentage_change * 0.3,
                'Real_Estate': impact.percentage_change * 1.2,
                'Cash': 0.0,
            }

    most_robust = comparison.identify_robust_allocation(portfolio_options)
    print(f"\nMost Robust Portfolio: {most_robust}")


def main():
    """Run all examples."""
    print("\n")
    print("#" * 80)
    print("# SCENARIO ANALYSIS EXAMPLES")
    print("#" * 80)
    print("\nThis script demonstrates scenario analysis tools")
    print("implemented in Allocation Station.\n")

    try:
        example_1_historical_scenarios()
        example_2_custom_scenarios()
        example_3_economic_scenarios()
        example_4_what_if_analysis()
        example_5_sensitivity_analysis()
        example_6_parametric_testing()
        example_7_scenario_comparison()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()