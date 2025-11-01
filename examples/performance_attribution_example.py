"""
Performance Attribution Examples

This script demonstrates the performance attribution capabilities of Allocation Station:
1. Brinson attribution (allocation, selection, interaction effects)
2. Factor-based attribution (Fama-French style analysis)
3. Contribution analysis (asset-level performance and risk)
4. Risk-adjusted attribution (Sharpe, Information Ratio)
5. Benchmark-relative attribution (tracking error, active share)
6. Time-weighted vs money-weighted returns
7. Custom attribution models (style drift, tactical vs strategic)

Run this script to see each attribution method in action.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Note: This example requires the performance_attribution module which may not be fully implemented
# Commenting out imports that don't exist
try:
    from allocation_station.analysis.performance_attribution import (
        BrinsonAttribution,
        FactorAttribution,
        ContributionAnalyzer,
        RiskAdjustedAttribution,
        BenchmarkAttribution,
        ReturnCalculator,
        CustomAttributionModel,
        style_drift_attribution,
        tactical_vs_strategic_attribution,
        AttributionMethod,
        FactorModel,
    )
except ImportError as e:
    print(f"Warning: Some performance attribution features are not available: {e}")
    print("This example requires additional implementation.")
    import sys
    sys.exit(0)


def create_sample_portfolio_data():
    """Create sample portfolio and benchmark data for examples."""
    np.random.seed(42)

    # Sectors
    sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy']

    # Single period weights and returns
    portfolio_weights = pd.Series({
        'Technology': 0.35,
        'Healthcare': 0.25,
        'Financials': 0.20,
        'Consumer': 0.15,
        'Energy': 0.05,
    })

    benchmark_weights = pd.Series({
        'Technology': 0.30,
        'Healthcare': 0.20,
        'Financials': 0.25,
        'Consumer': 0.20,
        'Energy': 0.05,
    })

    # Period returns
    portfolio_returns = pd.Series({
        'Technology': 0.12,
        'Healthcare': 0.08,
        'Financials': 0.04,
        'Consumer': 0.06,
        'Energy': -0.02,
    })

    benchmark_returns = pd.Series({
        'Technology': 0.10,
        'Healthcare': 0.09,
        'Financials': 0.05,
        'Consumer': 0.07,
        'Energy': -0.01,
    })

    return portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns, sectors


def create_time_series_data(n_periods=60):
    """Create time series data for multi-period analysis."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='M')
    sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy']

    # Generate correlated returns
    mean_returns = np.array([0.010, 0.008, 0.006, 0.007, 0.005])
    volatilities = np.array([0.06, 0.05, 0.04, 0.045, 0.07])

    correlation = np.array([
        [1.00, 0.60, 0.50, 0.55, 0.30],
        [0.60, 1.00, 0.45, 0.50, 0.25],
        [0.50, 0.45, 1.00, 0.60, 0.35],
        [0.55, 0.50, 0.60, 1.00, 0.40],
        [0.30, 0.25, 0.35, 0.40, 1.00],
    ])

    cov_matrix = np.outer(volatilities, volatilities) * correlation

    # Generate returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
    returns_df = pd.DataFrame(returns, index=dates, columns=sectors)

    # Create weights (with some variation over time)
    base_port_weights = np.array([0.35, 0.25, 0.20, 0.15, 0.05])
    base_bench_weights = np.array([0.30, 0.20, 0.25, 0.20, 0.05])

    portfolio_weights = []
    benchmark_weights = []

    for i in range(n_periods):
        # Add small random variations
        port_w = base_port_weights + np.random.normal(0, 0.02, 5)
        port_w = np.maximum(port_w, 0)  # No negative weights
        port_w = port_w / port_w.sum()  # Normalize

        bench_w = base_bench_weights + np.random.normal(0, 0.01, 5)
        bench_w = np.maximum(bench_w, 0)
        bench_w = bench_w / bench_w.sum()

        portfolio_weights.append(port_w)
        benchmark_weights.append(bench_w)

    portfolio_weights_df = pd.DataFrame(portfolio_weights, index=dates, columns=sectors)
    benchmark_weights_df = pd.DataFrame(benchmark_weights, index=dates, columns=sectors)

    # Calculate portfolio returns
    portfolio_returns = (portfolio_weights_df * returns_df).sum(axis=1)
    benchmark_returns = (benchmark_weights_df * returns_df).sum(axis=1)

    return {
        'dates': dates,
        'sectors': sectors,
        'sector_returns': returns_df,
        'portfolio_weights': portfolio_weights_df,
        'benchmark_weights': benchmark_weights_df,
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'covariance_matrix': pd.DataFrame(cov_matrix * 12, index=sectors, columns=sectors),  # Annualized
    }


def example_1_brinson_attribution():
    """Example 1: Brinson attribution analysis."""
    print("=" * 80)
    print("EXAMPLE 1: Brinson Attribution Analysis")
    print("=" * 80)

    pw, bw, pr, br, sectors = create_sample_portfolio_data()

    print("\nPortfolio vs Benchmark Positions:")
    comparison = pd.DataFrame({
        'Portfolio Weight': pw,
        'Benchmark Weight': bw,
        'Active Weight': pw - bw,
        'Portfolio Return': pr,
        'Benchmark Return': br,
    })
    print(comparison)

    # Single-period attribution
    print("\n--- Single-Period Brinson Attribution ---")
    brinson = BrinsonAttribution(method=AttributionMethod.BRINSON_FACHLER)
    result = brinson.calculate_single_period(pw, bw, pr, br)

    print(f"\nPortfolio Return: {result.total_return:.2%}")
    print(f"Benchmark Return: {result.benchmark_return:.2%}")
    print(f"Active Return: {result.active_return:.2%}")

    print("\nAttribution Components:")
    print(f"  Allocation Effect:  {result.components['allocation']:.2%}")
    print(f"  Selection Effect:   {result.components['selection']:.2%}")
    print(f"  Interaction Effect: {result.components['interaction']:.2%}")
    print(f"  Total:              {sum(result.components.values()):.2%}")

    print("\nSector-Level Attribution:")
    print(result.by_sector[['allocation_effect', 'selection_effect', 'interaction_effect', 'total_effect']])

    # Multi-period attribution
    print("\n--- Multi-Period Brinson Attribution (12 months) ---")
    ts_data = create_time_series_data(n_periods=12)

    multi_result = brinson.calculate_multi_period(
        ts_data['portfolio_weights'],
        ts_data['benchmark_weights'],
        ts_data['sector_returns'],
        ts_data['sector_returns'],  # Using same returns for simplicity
    )

    print("\nPeriod-by-Period Summary:")
    print(multi_result['summary'][['portfolio_return', 'benchmark_return', 'allocation', 'selection']])

    linked = multi_result['linked_result']
    print(f"\nLinked Attribution Results:")
    print(f"  Cumulative Portfolio Return: {linked.total_return:.2%}")
    print(f"  Cumulative Benchmark Return: {linked.benchmark_return:.2%}")
    print(f"  Cumulative Active Return:    {linked.active_return:.2%}")
    print(f"\n  Linked Allocation Effect:  {linked.components['allocation']:.2%}")
    print(f"  Linked Selection Effect:   {linked.components['selection']:.2%}")


def example_2_factor_attribution():
    """Example 2: Factor-based attribution."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Factor-Based Attribution")
    print("=" * 80)

    # Create portfolio and factor returns
    np.random.seed(42)
    n_periods = 60
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='M')

    # Generate factor returns (Market, Size, Value)
    factor_returns = pd.DataFrame({
        'Market': np.random.normal(0.008, 0.04, n_periods),
        'Size': np.random.normal(0.002, 0.03, n_periods),
        'Value': np.random.normal(0.003, 0.025, n_periods),
    }, index=dates)

    # Generate portfolio returns with factor exposures
    # Portfolio has: beta_market=1.2, beta_size=0.5, beta_value=-0.3, alpha=0.002
    true_alpha = 0.002
    true_betas = np.array([1.2, 0.5, -0.3])

    portfolio_returns = (
        true_alpha +
        (factor_returns.values @ true_betas) +
        np.random.normal(0, 0.01, n_periods)  # Idiosyncratic risk
    )
    portfolio_returns = pd.Series(portfolio_returns, index=dates)

    print("Analyzing portfolio factor exposures...")
    print(f"True parameters: alpha={true_alpha:.2%}, betas={true_betas}")

    # Factor attribution
    factor_attr = FactorAttribution()
    results = factor_attr.analyze(portfolio_returns, factor_returns, risk_free_rate=0.001)

    print(f"\nEstimated Alpha: {results['alpha']:.2%} (monthly)")
    print(f"Estimated Factor Loadings:")
    print(results['factor_loadings'])
    print(f"\nR-squared: {results['r_squared']:.2%}")

    print("\nReturn Decomposition:")
    decomp = results['return_decomposition']
    print(f"  Average Return:       {decomp['average_return']:.2%}")
    print(f"  Alpha Contribution:   {decomp['alpha']:.2%}")
    print(f"  Factor Contribution:  {decomp['factor_contribution']:.2%}")
    print(f"  Residual:             {decomp['residual']:.2%}")

    print("\nAverage Factor Contributions:")
    print(results['factor_contribution_summary'])

    # Active factor attribution vs benchmark
    print("\n--- Active Factor Attribution vs Benchmark ---")

    # Create benchmark with different factor exposures
    benchmark_betas = np.array([1.0, 0.0, 0.0])  # Market-only exposure
    benchmark_returns = (factor_returns.values @ benchmark_betas) + np.random.normal(0, 0.005, n_periods)
    benchmark_returns = pd.Series(benchmark_returns, index=dates)

    active_results = factor_attr.decompose_active_return(
        portfolio_returns, benchmark_returns, factor_returns
    )

    print(f"\nActive Return: {active_results['active_return']:.2%}")
    print(f"Active Alpha:  {active_results['active_alpha']:.2%}")
    print(f"\nActive Factor Exposures:")
    print(active_results['active_betas'])
    print(f"\nFactor Timing Contribution: {active_results['factor_timing_contribution']:.2%}")


def example_3_contribution_analysis():
    """Example 3: Asset-level contribution analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Contribution Analysis")
    print("=" * 80)

    # Create individual asset data
    assets = ['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'JPM', 'XOM', 'PG', 'VZ']

    weights = pd.Series({
        'AAPL': 0.18,
        'MSFT': 0.15,
        'GOOGL': 0.12,
        'JNJ': 0.15,
        'JPM': 0.12,
        'XOM': 0.08,
        'PG': 0.10,
        'VZ': 0.10,
    })

    returns = pd.Series({
        'AAPL': 0.15,
        'MSFT': 0.12,
        'GOOGL': 0.18,
        'JNJ': 0.06,
        'JPM': 0.08,
        'XOM': -0.05,
        'PG': 0.04,
        'VZ': 0.02,
    })

    print("Asset Positions:")
    print(pd.DataFrame({'Weight': weights, 'Return': returns}))

    # Calculate contributions
    contrib_analyzer = ContributionAnalyzer()
    contributions = contrib_analyzer.calculate_contributions(weights, returns)

    print("\nPerformance Contributions:")
    print(contributions[['average_weight', 'return', 'contribution', 'pct_contribution']])

    portfolio_return = contributions['contribution'].sum()
    print(f"\nTotal Portfolio Return: {portfolio_return:.2%}")

    # Top contributors
    print("\n--- Top Contributors Analysis ---")
    top_results = contrib_analyzer.analyze_top_contributors(weights, returns, top_n=3)

    print("\nTop 3 Winners:")
    print(top_results['top_winners'][['average_weight', 'return', 'contribution', 'pct_contribution']])

    print("\nTop 3 Losers:")
    print(top_results['top_losers'][['average_weight', 'return', 'contribution', 'pct_contribution']])

    # Risk contribution
    print("\n--- Risk Contribution Analysis ---")

    # Create sample covariance matrix
    np.random.seed(42)
    corr_matrix = np.random.uniform(0.3, 0.7, (len(assets), len(assets)))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)

    volatilities = np.array([0.25, 0.22, 0.28, 0.15, 0.20, 0.30, 0.12, 0.18])
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    cov_df = pd.DataFrame(cov_matrix, index=assets, columns=assets)

    risk_contrib = contrib_analyzer.calculate_risk_contribution(weights, cov_df)

    print("\nRisk Contributions:")
    print(risk_contrib)


def example_4_risk_adjusted_attribution():
    """Example 4: Risk-adjusted attribution."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Risk-Adjusted Attribution")
    print("=" * 80)

    ts_data = create_time_series_data(n_periods=36)

    # Calculate Sharpe decomposition
    print("--- Sharpe Ratio Decomposition ---")

    risk_attr = RiskAdjustedAttribution()

    sector_weights = ts_data['portfolio_weights'].iloc[-1]  # Latest weights
    sharpe_decomp = risk_attr.sharpe_decomposition(
        ts_data['portfolio_returns'],
        sector_weights,
        ts_data['sector_returns'],
        risk_free_rate=0.002,
    )

    print(f"\nPortfolio Sharpe Ratio: {sharpe_decomp['portfolio_sharpe']:.3f}")

    print("\nSector Sharpe Ratios:")
    print(sharpe_decomp['sector_sharpes'])

    print("\nSector Sharpe Contributions:")
    sharpe_contrib_df = pd.DataFrame({
        'Weight': sharpe_decomp['sector_weights'],
        'Sharpe Ratio': sharpe_decomp['sector_sharpes'],
        'Contribution': sharpe_decomp['sharpe_contribution'],
    })
    print(sharpe_contrib_df)

    # Information ratio attribution
    print("\n--- Information Ratio Attribution ---")

    # Use first 3 assets for simplicity
    asset_returns_df = ts_data['sector_returns'].iloc[:, :3]
    asset_weights = ts_data['portfolio_weights'].iloc[-1, :3]
    asset_weights = asset_weights / asset_weights.sum()  # Normalize
    benchmark_weights = ts_data['benchmark_weights'].iloc[-1, :3]
    benchmark_weights = benchmark_weights / benchmark_weights.sum()

    ir_attr = risk_attr.information_ratio_attribution(
        ts_data['portfolio_returns'],
        ts_data['benchmark_returns'],
        asset_weights,
        asset_returns_df,
        benchmark_weights,
    )

    print(f"\nInformation Ratio: {ir_attr['information_ratio']:.3f}")
    print(f"Active Return: {ir_attr['active_return']:.2%}")
    print(f"Tracking Error: {ir_attr['tracking_error']:.2%}")

    print("\nAsset Active Return Contributions:")
    print(ir_attr['asset_active_contributions'])


def example_5_benchmark_attribution():
    """Example 5: Benchmark-relative attribution."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Benchmark-Relative Attribution")
    print("=" * 80)

    ts_data = create_time_series_data(n_periods=60)

    bench_attr = BenchmarkAttribution()
    results = bench_attr.analyze(
        ts_data['portfolio_returns'],
        ts_data['benchmark_returns'],
        ts_data['portfolio_weights'].iloc[-1],
        ts_data['benchmark_weights'].iloc[-1],
    )

    print("Comprehensive Benchmark Analysis:")
    print(f"\nTotal Returns:")
    print(f"  Portfolio: {results['total_portfolio_return']:.2%}")
    print(f"  Benchmark: {results['total_benchmark_return']:.2%}")
    print(f"  Active:    {results['total_active_return']:.2%}")

    print(f"\nRisk Metrics:")
    print(f"  Average Active Return: {results['avg_active_return']:.2%}")
    print(f"  Tracking Error:        {results['tracking_error']:.2%}")
    print(f"  Information Ratio:     {results['information_ratio']:.3f}")
    print(f"  Beta:                  {results['beta']:.3f}")
    print(f"  Correlation:           {results['correlation']:.3f}")

    print(f"\nCapture Ratios:")
    print(f"  Up Capture:            {results['up_capture']:.2%}")
    print(f"  Down Capture:          {results['down_capture']:.2%}")

    print(f"\nActive Management Metrics:")
    print(f"  Hit Rate:              {results['hit_rate']:.1%}")
    print(f"  Active Share:          {results['active_share']:.2%}")

    # Tracking error decomposition
    print("\n--- Tracking Error Decomposition ---")

    # Create simple factor returns for decomposition
    factor_returns = ts_data['sector_returns'].iloc[:, :2]  # Use first 2 sectors as "factors"

    te_decomp = bench_attr.tracking_error_decomposition(
        ts_data['portfolio_returns'],
        ts_data['benchmark_returns'],
        factor_returns,
    )

    print(f"\nTotal Tracking Error:      {te_decomp['total_tracking_error']:.2%}")
    print(f"Systematic TE (factors):   {te_decomp['systematic_tracking_error']:.2%} ({te_decomp['systematic_pct']:.1f}%)")
    print(f"Specific TE (stock pick):  {te_decomp['specific_tracking_error']:.2%} ({te_decomp['specific_pct']:.1f}%)")


def example_6_return_calculations():
    """Example 6: Time-weighted vs money-weighted returns."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Time-Weighted vs Money-Weighted Returns")
    print("=" * 80)

    # Create portfolio with cash flows
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    n_periods = len(dates)

    # Portfolio values
    initial_value = 100000
    values = [initial_value]

    # Simulate portfolio growth with cash flows
    np.random.seed(42)
    monthly_returns = np.random.normal(0.008, 0.04, n_periods - 1)

    # Cash flows (contributions/withdrawals)
    cash_flows = [0]  # No flow at start

    for i, ret in enumerate(monthly_returns):
        # Add cash flow (contribution every 3 months)
        if (i + 1) % 3 == 0:
            flow = 5000  # $5,000 contribution
        elif (i + 1) == 10:  # Withdrawal in month 10
            flow = -10000
        else:
            flow = 0

        cash_flows.append(flow)

        # Calculate new value
        new_value = values[-1] * (1 + ret) + flow
        values.append(new_value)

    portfolio_values = pd.Series(values, index=dates)
    cash_flows_series = pd.Series(cash_flows, index=dates)

    print("Portfolio Evolution:")
    summary_df = pd.DataFrame({
        'Portfolio Value': portfolio_values,
        'Cash Flow': cash_flows_series,
    })
    print(summary_df.head(13))

    # Calculate different return metrics
    calc = ReturnCalculator()

    print("\n--- Return Calculation Methods ---")

    # Time-weighted return
    twr_result = calc.time_weighted_return(portfolio_values, cash_flows_series)
    print(f"\nTime-Weighted Return (TWR):")
    print(f"  Total Return:      {twr_result['time_weighted_return']:.2%}")
    print(f"  Annualized Return: {twr_result['annualized_twr']:.2%}")
    print(f"  >>> Measures manager skill, independent of cash flow timing")

    # Money-weighted return
    mwr_result = calc.money_weighted_return(portfolio_values, cash_flows_series)
    print(f"\nMoney-Weighted Return (MWR/IRR):")
    print(f"  Internal Rate of Return: {mwr_result['money_weighted_return']:.2%}")
    print(f"  >>> Measures investor experience, affected by cash flow timing")

    # Comparison
    print("\n--- Comparison Table ---")
    comparison = calc.compare_returns(portfolio_values, cash_flows_series)
    print(comparison)

    difference = twr_result['time_weighted_return'] - mwr_result['money_weighted_return']
    print(f"\nDifference (TWR - MWR): {difference:.2%}")
    if difference > 0:
        print("  >>> Investor added money at inopportune times (bought high)")
    elif difference < 0:
        print("  >>> Investor added money at opportune times (bought low)")


def example_7_custom_attribution():
    """Example 7: Custom attribution models."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Custom Attribution Models")
    print("=" * 80)

    sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy']

    # Style drift attribution
    print("--- Style Drift Attribution ---")

    target_weights = pd.Series({
        'Technology': 0.30,
        'Healthcare': 0.25,
        'Financials': 0.20,
        'Consumer': 0.15,
        'Energy': 0.10,
    })

    current_weights = pd.Series({
        'Technology': 0.38,  # Drifted up
        'Healthcare': 0.23,
        'Financials': 0.18,
        'Consumer': 0.14,
        'Energy': 0.07,  # Drifted down
    })

    returns = pd.Series({
        'Technology': 0.12,
        'Healthcare': 0.08,
        'Financials': 0.05,
        'Consumer': 0.06,
        'Energy': 0.02,
    })

    print("\nTarget vs Current Weights:")
    weight_comparison = pd.DataFrame({
        'Target': target_weights,
        'Current': current_weights,
        'Drift': current_weights - target_weights,
        'Return': returns,
    })
    print(weight_comparison)

    drift_result = style_drift_attribution(current_weights, target_weights, returns)

    print(f"\nActual Return:  {drift_result['actual_return']:.2%}")
    print(f"Target Return:  {drift_result['target_return']:.2%}")
    print(f"Drift Impact:   {drift_result['drift_impact']:.2%}")

    print("\nDrift Impact by Sector:")
    print(drift_result['drift_by_asset'].sort_values(ascending=False))

    # Tactical vs Strategic attribution
    print("\n--- Tactical vs Strategic Attribution ---")

    strategic_weights = pd.Series({  # Long-term policy
        'Technology': 0.30,
        'Healthcare': 0.25,
        'Financials': 0.20,
        'Consumer': 0.15,
        'Energy': 0.10,
    })

    tactical_weights = pd.Series({  # Current tactical tilts
        'Technology': 0.35,  # Overweight tech (bullish)
        'Healthcare': 0.25,
        'Financials': 0.15,  # Underweight financials
        'Consumer': 0.18,  # Overweight consumer
        'Energy': 0.07,  # Underweight energy
    })

    tactical_result = tactical_vs_strategic_attribution(strategic_weights, tactical_weights, returns)

    print(f"\nStrategic Return (policy):      {tactical_result['strategic_return']:.2%}")
    print(f"Tactical Return (with tilts):   {tactical_result['tactical_return']:.2%}")
    print(f"Tactical Value Added:           {tactical_result['tactical_value_added']:.2%}")

    print("\nTactical Tilts:")
    tactical_analysis = pd.DataFrame({
        'Strategic': strategic_weights,
        'Tactical': tactical_weights,
        'Tilt': tactical_result['tactical_tilts'],
        'Return': returns,
        'Contribution': tactical_result['tactical_contributions'],
    })
    print(tactical_analysis)

    # Custom attribution model
    print("\n--- Custom Attribution Model Example ---")

    def custom_sector_rotation_attribution(
        weights_t0: pd.Series,
        weights_t1: pd.Series,
        returns: pd.Series,
    ) -> dict:
        """
        Custom attribution: Separate gains from rebalancing vs holding.
        """
        # Return from holding initial positions
        hold_return = (weights_t0 * returns).sum()

        # Return from rebalancing
        rebalancing_gain = ((weights_t1 - weights_t0) * returns).sum()

        # Total return
        total_return = (weights_t1 * returns).sum()

        return {
            'total_return': total_return,
            'hold_return': hold_return,
            'rebalancing_gain': rebalancing_gain,
            'trades': weights_t1 - weights_t0,
        }

    custom_model = CustomAttributionModel(
        attribution_function=custom_sector_rotation_attribution,
        name="Sector Rotation Attribution"
    )

    # Example: rotated from energy to technology
    weights_before = strategic_weights.copy()
    weights_after = tactical_weights.copy()

    custom_result = custom_model.calculate(weights_before, weights_after, returns)

    print(f"\nSector Rotation Analysis:")
    print(f"  Total Return:      {custom_result['total_return']:.2%}")
    print(f"  Hold Return:       {custom_result['hold_return']:.2%}")
    print(f"  Rebalancing Gain:  {custom_result['rebalancing_gain']:.2%}")

    print("\nTrades Made:")
    print(custom_result['trades'].sort_values())


def main():
    """Run all examples."""
    print("\n")
    print("#" * 80)
    print("# PERFORMANCE ATTRIBUTION EXAMPLES")
    print("#" * 80)
    print("\nThis script demonstrates performance attribution techniques")
    print("implemented in Allocation Station.\n")

    try:
        example_1_brinson_attribution()
        example_2_factor_attribution()
        example_3_contribution_analysis()
        example_4_risk_adjusted_attribution()
        example_5_benchmark_attribution()
        example_6_return_calculations()
        example_7_custom_attribution()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
