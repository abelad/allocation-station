"""
Advanced Optimization Examples

This script demonstrates the advanced optimization capabilities of Allocation Station:
1. Black-Litterman model with investor views
2. Robust optimization under parameter uncertainty
3. Hierarchical Risk Parity (HRP)
4. Mean-CVaR optimization
5. Kelly Criterion optimization
6. Custom objective functions
7. Multi-period optimization with transaction costs

Run this script to see each optimization method in action.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from allocation_station.optimization.advanced_optimization import (
    BlackLittermanModel,
    RobustOptimizer,
    HierarchicalRiskParity,
    MeanCVaROptimizer,
    KellyCriterionOptimizer,
    CustomObjectiveOptimizer,
    MultiPeriodOptimizer,
    UncertaintySet,
    InvestorView,
    CustomObjective,
)


def create_sample_data():
    """Create sample return data for optimization examples."""
    np.random.seed(42)

    # 5 assets with different characteristics
    symbols = ['US_EQUITY', 'INTL_EQUITY', 'BONDS', 'REAL_ESTATE', 'COMMODITIES']
    n_assets = len(symbols)
    n_days = 252 * 3  # 3 years of daily data

    # Generate correlated returns
    mean_returns = np.array([0.10, 0.08, 0.04, 0.09, 0.06]) / 252  # Daily returns
    volatilities = np.array([0.20, 0.22, 0.08, 0.18, 0.25]) / np.sqrt(252)  # Daily vol

    # Correlation matrix
    correlation = np.array([
        [1.00, 0.70, -0.15, 0.50, 0.30],
        [0.70, 1.00, -0.10, 0.45, 0.35],
        [-0.15, -0.10, 1.00, -0.05, -0.10],
        [0.50, 0.45, -0.05, 1.00, 0.40],
        [0.30, 0.35, -0.10, 0.40, 1.00],
    ])

    # Create covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation

    # Generate returns using multivariate normal
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)

    # Create DataFrame
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    returns_df = pd.DataFrame(returns, index=dates, columns=symbols)

    # Calculate expected returns and covariance (annualized)
    expected_returns = returns_df.mean() * 252
    covariance_matrix = returns_df.cov() * 252

    return returns_df, expected_returns, covariance_matrix


def example_1_black_litterman():
    """Example 1: Black-Litterman model with investor views."""
    print("=" * 80)
    print("EXAMPLE 1: Black-Litterman Model")
    print("=" * 80)

    returns_df, expected_returns, covariance_matrix = create_sample_data()

    # Market capitalization weights (as proxy for equilibrium portfolio)
    market_caps = pd.Series({
        'US_EQUITY': 30000,
        'INTL_EQUITY': 20000,
        'BONDS': 25000,
        'REAL_ESTATE': 15000,
        'COMMODITIES': 10000,
    })
    market_weights = market_caps / market_caps.sum()

    print("\nMarket Equilibrium Weights:")
    print(market_weights)

    # Create Black-Litterman model
    bl_model = BlackLittermanModel(
        market_weights=market_weights,
        risk_aversion=2.5,
        tau=0.05,
        risk_free_rate=0.02,
    )

    # Calculate market-implied returns
    implied_returns = bl_model.calculate_market_implied_returns(covariance_matrix)
    print("\nMarket-Implied Equilibrium Returns:")
    print(implied_returns)

    # Define investor views
    views = [
        # View 1: US equity will outperform international equity by 3%
        InvestorView(
            assets=['US_EQUITY', 'INTL_EQUITY'],
            weights=[1.0, -1.0],
            expected_return=0.03,
            confidence=0.60,
        ),
        # View 2: Real estate will return 12%
        InvestorView(
            assets=['REAL_ESTATE'],
            weights=[1.0],
            expected_return=0.12,
            confidence=0.80,
        ),
        # View 3: Bonds will underperform commodities by 2%
        InvestorView(
            assets=['BONDS', 'COMMODITIES'],
            weights=[1.0, -1.0],
            expected_return=-0.02,
            confidence=0.50,
        ),
    ]

    print("\nInvestor Views:")
    for i, view in enumerate(views, 1):
        print(f"  View {i}: {view.assets} with weights {view.weights}")
        print(f"          Expected return: {view.expected_return:.2%}, Confidence: {view.confidence:.0%}")

    # Calculate posterior returns incorporating views
    posterior_returns = bl_model.calculate_posterior_returns(
        implied_returns, covariance_matrix, views
    )

    print("\nPosterior Returns (combining market equilibrium + views):")
    print(posterior_returns)

    # Optimize portfolio with posterior returns
    optimal_weights = bl_model.optimize(posterior_returns, covariance_matrix)

    print("\nOptimal Portfolio Weights:")
    for asset, weight in optimal_weights.items():
        print(f"  {asset:20s}: {weight:6.2%}")

    print("\nComparison:")
    comparison = pd.DataFrame({
        'Market Weights': market_weights,
        'BL Optimal Weights': pd.Series(optimal_weights),
        'Difference': pd.Series(optimal_weights) - market_weights,
    })
    print(comparison)


def example_2_robust_optimization():
    """Example 2: Robust optimization under parameter uncertainty."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Robust Optimization")
    print("=" * 80)

    returns_df, expected_returns, covariance_matrix = create_sample_data()

    print("\nEstimated Expected Returns:")
    print(expected_returns)

    # Create robust optimizer
    robust_opt = RobustOptimizer(
        uncertainty_set=UncertaintySet.ELLIPSOIDAL,
        kappa=0.5,  # Moderate uncertainty aversion
        epsilon=0.05,  # 5% uncertainty in returns
        gamma=0.1,  # 10% uncertainty in covariance
    )

    print(f"\nRobust Optimization Settings:")
    print(f"  Uncertainty Set: {robust_opt.uncertainty_set.value}")
    print(f"  Uncertainty Aversion (kappa): {robust_opt.kappa}")
    print(f"  Return Uncertainty (epsilon): {robust_opt.epsilon:.1%}")
    print(f"  Covariance Uncertainty (gamma): {robust_opt.gamma:.1%}")

    # Optimize for worst-case variance
    print("\n--- Worst-Case Variance Optimization ---")
    weights_wcv = robust_opt.optimize_worst_case_var(expected_returns, covariance_matrix)

    print("\nRobust Portfolio Weights (Worst-Case Variance):")
    for asset, weight in weights_wcv.items():
        print(f"  {asset:20s}: {weight:6.2%}")

    # Optimize for worst-case CVaR
    print("\n--- Worst-Case CVaR Optimization ---")

    # Generate scenarios for CVaR calculation
    n_scenarios = 1000
    scenarios = np.random.multivariate_normal(
        expected_returns.values,
        covariance_matrix.values,
        n_scenarios
    )
    scenarios_df = pd.DataFrame(scenarios, columns=expected_returns.index)

    weights_wcc = robust_opt.optimize_worst_case_cvar(
        expected_returns, scenarios_df, confidence_level=0.95
    )

    print("\nRobust Portfolio Weights (Worst-Case CVaR):")
    for asset, weight in weights_wcc.items():
        print(f"  {asset:20s}: {weight:6.2%}")

    # Compare with standard mean-variance optimization
    from allocation_station.optimization.optimizer import MeanVarianceOptimizer
    mv_opt = MeanVarianceOptimizer()
    weights_mv = mv_opt.optimize(expected_returns, covariance_matrix)

    print("\nComparison with Standard Mean-Variance:")
    comparison = pd.DataFrame({
        'Mean-Variance': pd.Series(weights_mv),
        'Robust WCV': pd.Series(weights_wcv),
        'Robust WCCVaR': pd.Series(weights_wcc),
    })
    print(comparison)


def example_3_hierarchical_risk_parity():
    """Example 3: Hierarchical Risk Parity (HRP)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Hierarchical Risk Parity (HRP)")
    print("=" * 80)

    returns_df, expected_returns, covariance_matrix = create_sample_data()

    # Create HRP optimizer
    hrp = HierarchicalRiskParity(linkage_method='single')

    print("HRP Algorithm Steps:")
    print("  1. Calculate correlation-based distance matrix")
    print("  2. Perform hierarchical clustering")
    print("  3. Apply quasi-diagonalization")
    print("  4. Recursive bisection for weight allocation")

    # Optimize
    weights = hrp.optimize(returns_df, covariance_matrix)

    print("\nHRP Portfolio Weights:")
    for asset, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset:20s}: {weight:6.2%}")

    # Calculate portfolio risk contribution
    portfolio_weights = np.array([weights[col] for col in covariance_matrix.columns])
    marginal_risk = covariance_matrix.values @ portfolio_weights
    risk_contribution = portfolio_weights * marginal_risk
    total_risk = np.sqrt(portfolio_weights @ covariance_matrix.values @ portfolio_weights)
    risk_contribution_pct = risk_contribution / total_risk

    print("\nRisk Contribution by Asset:")
    for i, asset in enumerate(covariance_matrix.columns):
        print(f"  {asset:20s}: {risk_contribution_pct[i]:6.2%}")

    print(f"\nPortfolio Volatility: {total_risk:.2%}")

    # Compare with equal-weight and inverse-variance portfolios
    equal_weights = {asset: 1.0 / len(returns_df.columns) for asset in returns_df.columns}

    variances = np.diag(covariance_matrix.values)
    inv_var_weights_raw = 1.0 / variances
    inv_var_weights = {
        asset: w / inv_var_weights_raw.sum()
        for asset, w in zip(returns_df.columns, inv_var_weights_raw)
    }

    print("\nComparison with Other Risk-Based Approaches:")
    comparison = pd.DataFrame({
        'HRP': pd.Series(weights),
        'Equal Weight': pd.Series(equal_weights),
        'Inverse Variance': pd.Series(inv_var_weights),
    })
    print(comparison)


def example_4_mean_cvar():
    """Example 4: Mean-CVaR optimization."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Mean-CVaR Optimization")
    print("=" * 80)

    returns_df, expected_returns, covariance_matrix = create_sample_data()

    # Generate scenarios for CVaR
    n_scenarios = 1000
    scenarios = np.random.multivariate_normal(
        expected_returns.values,
        covariance_matrix.values,
        n_scenarios
    )
    scenarios_df = pd.DataFrame(scenarios, columns=expected_returns.index)

    print(f"Generated {n_scenarios} return scenarios for CVaR calculation")

    # Create CVaR optimizer
    cvar_opt = MeanCVaROptimizer(
        confidence_level=0.95,
        target_return=None,  # Will minimize CVaR subject to constraints
        max_cvar=None,
    )

    print(f"\nCVaR Optimization Settings:")
    print(f"  Confidence Level: {cvar_opt.confidence_level:.0%}")
    print(f"  Focus: Minimize CVaR (conditional tail losses)")

    # Optimize
    weights = cvar_opt.optimize(expected_returns, scenarios_df)

    print("\nCVaR-Optimal Portfolio Weights:")
    for asset, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset:20s}: {weight:6.2%}")

    # Calculate portfolio statistics
    portfolio_weights = np.array([weights[col] for col in scenarios_df.columns])
    portfolio_scenarios = scenarios_df.values @ portfolio_weights

    var_95 = np.percentile(portfolio_scenarios, 5)
    cvar_95 = portfolio_scenarios[portfolio_scenarios <= var_95].mean()
    expected_return = portfolio_scenarios.mean()
    volatility = portfolio_scenarios.std()

    print(f"\nPortfolio Statistics:")
    print(f"  Expected Return: {expected_return:.2%}")
    print(f"  Volatility: {volatility:.2%}")
    print(f"  VaR (95%): {var_95:.2%}")
    print(f"  CVaR (95%): {cvar_95:.2%}")

    # Compare with mean-variance optimization
    from allocation_station.optimization.optimizer import MeanVarianceOptimizer
    mv_opt = MeanVarianceOptimizer(target_return=expected_return)
    weights_mv = mv_opt.optimize(expected_returns, covariance_matrix)

    portfolio_weights_mv = np.array([weights_mv[col] for col in scenarios_df.columns])
    portfolio_scenarios_mv = scenarios_df.values @ portfolio_weights_mv
    cvar_95_mv = portfolio_scenarios_mv[portfolio_scenarios_mv <= np.percentile(portfolio_scenarios_mv, 5)].mean()

    print(f"\nComparison with Mean-Variance (same expected return):")
    print(f"  CVaR-optimized CVaR: {cvar_95:.2%}")
    print(f"  Mean-Variance CVaR: {cvar_95_mv:.2%}")
    print(f"  CVaR Improvement: {(cvar_95_mv - cvar_95):.2%}")


def example_5_kelly_criterion():
    """Example 5: Kelly Criterion optimization."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Kelly Criterion Optimization")
    print("=" * 80)

    returns_df, expected_returns, covariance_matrix = create_sample_data()

    print("Kelly Criterion: Maximizes expected logarithmic wealth growth")
    print("Full Kelly formula: w* = Σ^-1 * (μ - rf)")

    # Full Kelly
    print("\n--- Full Kelly ---")
    kelly_full = KellyCriterionOptimizer(
        risk_free_rate=0.02,
        fractional=1.0,  # Full Kelly
        max_leverage=None,
    )

    weights_full = kelly_full.optimize(expected_returns, covariance_matrix)

    print("\nFull Kelly Weights:")
    total_weight = sum(weights_full.values())
    for asset, weight in sorted(weights_full.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset:20s}: {weight:6.2%}")
    print(f"  Total Allocation: {total_weight:6.2%}")

    if total_weight > 1.0:
        print(f"  >>> Uses leverage: {total_weight - 1.0:.2%}")

    # Half Kelly (more conservative)
    print("\n--- Half Kelly (Fractional Kelly with 50%) ---")
    kelly_half = KellyCriterionOptimizer(
        risk_free_rate=0.02,
        fractional=0.5,  # Half Kelly
        max_leverage=None,
    )

    weights_half = kelly_half.optimize(expected_returns, covariance_matrix)

    print("\nHalf Kelly Weights:")
    total_weight_half = sum(weights_half.values())
    for asset, weight in sorted(weights_half.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset:20s}: {weight:6.2%}")
    print(f"  Total Allocation: {total_weight_half:6.2%}")

    # Constrained Kelly (no leverage)
    print("\n--- Constrained Kelly (Max Leverage = 1.0) ---")
    kelly_constrained = KellyCriterionOptimizer(
        risk_free_rate=0.02,
        fractional=1.0,
        max_leverage=1.0,  # No leverage allowed
    )

    weights_constrained = kelly_constrained.optimize(expected_returns, covariance_matrix)

    print("\nConstrained Kelly Weights:")
    total_weight_const = sum(weights_constrained.values())
    for asset, weight in sorted(weights_constrained.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset:20s}: {weight:6.2%}")
    print(f"  Total Allocation: {total_weight_const:6.2%}")

    # Simulate growth rates
    n_simulations = 10000
    simulated_returns = np.random.multivariate_normal(
        expected_returns.values,
        covariance_matrix.values,
        n_simulations
    )

    def calculate_log_wealth_growth(weights_dict, returns_matrix):
        weights_array = np.array([weights_dict[col] for col in expected_returns.index])
        portfolio_returns = returns_matrix @ weights_array
        log_growth = np.log(1 + portfolio_returns)
        return log_growth.mean(), log_growth.std()

    log_growth_full, log_vol_full = calculate_log_wealth_growth(weights_full, simulated_returns)
    log_growth_half, log_vol_half = calculate_log_wealth_growth(weights_half, simulated_returns)
    log_growth_const, log_vol_const = calculate_log_wealth_growth(weights_constrained, simulated_returns)

    print("\nExpected Log Wealth Growth Rates:")
    print(f"  Full Kelly:        {log_growth_full:.2%} (volatility: {log_vol_full:.2%})")
    print(f"  Half Kelly:        {log_growth_half:.2%} (volatility: {log_vol_half:.2%})")
    print(f"  Constrained Kelly: {log_growth_const:.2%} (volatility: {log_vol_const:.2%})")


def example_6_custom_objectives():
    """Example 6: Custom objective functions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Custom Objective Functions")
    print("=" * 80)

    returns_df, expected_returns, covariance_matrix = create_sample_data()

    # Omega Ratio Optimization
    print("\n--- Omega Ratio Optimization ---")
    print("Omega Ratio = Probability-weighted gains / Probability-weighted losses")

    omega_opt = CustomObjectiveOptimizer(
        objective=CustomObjective.OMEGA_RATIO,
        threshold_return=0.0,  # Threshold for Omega ratio
    )

    weights_omega = omega_opt.optimize(returns_df, expected_returns, covariance_matrix)

    print("\nOmega Ratio Portfolio Weights:")
    for asset, weight in sorted(weights_omega.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset:20s}: {weight:6.2%}")

    # Calculate Omega ratio for the portfolio
    portfolio_weights = np.array([weights_omega[col] for col in returns_df.columns])
    portfolio_returns = returns_df.values @ portfolio_weights
    threshold = 0.0
    gains = portfolio_returns[portfolio_returns > threshold] - threshold
    losses = threshold - portfolio_returns[portfolio_returns <= threshold]
    omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf

    print(f"\nPortfolio Omega Ratio: {omega_ratio:.3f}")

    # Maximum Drawdown Optimization
    print("\n--- Minimum Maximum Drawdown Optimization ---")
    print("Seeks to minimize the worst peak-to-trough decline")

    maxdd_opt = CustomObjectiveOptimizer(
        objective=CustomObjective.MAX_DRAWDOWN,
    )

    weights_maxdd = maxdd_opt.optimize(returns_df, expected_returns, covariance_matrix)

    print("\nMin-MaxDD Portfolio Weights:")
    for asset, weight in sorted(weights_maxdd.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset:20s}: {weight:6.2%}")

    # Calculate maximum drawdown
    portfolio_weights_dd = np.array([weights_maxdd[col] for col in returns_df.columns])
    portfolio_returns_dd = returns_df.values @ portfolio_weights_dd
    cumulative = (1 + portfolio_returns_dd).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    print(f"\nPortfolio Maximum Drawdown: {max_drawdown:.2%}")

    # Custom Objective Function
    print("\n--- User-Defined Custom Objective ---")

    def custom_objective_function(weights, returns_df, expected_returns, covariance_matrix):
        """
        Custom objective: Maximize return/volatility while penalizing concentration.

        Objective = Sharpe Ratio - Concentration Penalty
        """
        portfolio_return = expected_returns.values @ weights
        portfolio_vol = np.sqrt(weights @ covariance_matrix.values @ weights)
        sharpe = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0

        # Concentration penalty using Herfindahl Index
        herfindahl = np.sum(weights ** 2)
        concentration_penalty = herfindahl * 2.0  # Scale factor

        # Minimize negative objective
        return -(sharpe - concentration_penalty)

    custom_opt = CustomObjectiveOptimizer(
        objective=custom_objective_function,
    )

    weights_custom = custom_opt.optimize(returns_df, expected_returns, covariance_matrix)

    print("\nCustom Objective Portfolio Weights:")
    for asset, weight in sorted(weights_custom.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset:20s}: {weight:6.2%}")

    # Calculate metrics
    portfolio_weights_c = np.array([weights_custom[col] for col in returns_df.columns])
    portfolio_return = expected_returns.values @ portfolio_weights_c
    portfolio_vol = np.sqrt(portfolio_weights_c @ covariance_matrix.values @ portfolio_weights_c)
    sharpe = (portfolio_return - 0.02) / portfolio_vol
    herfindahl = np.sum(portfolio_weights_c ** 2)

    print(f"\nPortfolio Metrics:")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Herfindahl Index: {herfindahl:.3f}")
    print(f"  Effective N: {1/herfindahl:.1f} assets")


def example_7_multi_period():
    """Example 7: Multi-period optimization with transaction costs."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Multi-Period Optimization")
    print("=" * 80)

    returns_df, expected_returns, covariance_matrix = create_sample_data()

    # Simulate changing market conditions over multiple periods
    n_periods = 4

    # Create different expected returns for each period (simulating changing views)
    expected_returns_list = []
    covariance_matrices = []

    for i in range(n_periods):
        # Add some noise to simulate changing expectations
        noise = np.random.normal(0, 0.01, len(expected_returns))
        period_returns = expected_returns + noise
        expected_returns_list.append(period_returns)

        # Use same covariance for simplicity (could vary)
        covariance_matrices.append(covariance_matrix)

    print(f"Optimizing over {n_periods} periods with transaction costs")
    print(f"Transaction cost: 10 bps (0.10%) per trade")

    # Initial portfolio (equal weight)
    initial_weights = pd.Series(
        {asset: 1.0 / len(expected_returns) for asset in expected_returns.index}
    )

    print("\nInitial Portfolio Weights:")
    for asset, weight in initial_weights.items():
        print(f"  {asset:20s}: {weight:6.2%}")

    # Multi-period optimizer
    mp_opt = MultiPeriodOptimizer(
        n_periods=n_periods,
        transaction_cost=0.0010,  # 10 bps
        risk_aversion=2.0,
    )

    # Optimize
    weights_by_period = mp_opt.optimize(
        expected_returns_list,
        covariance_matrices,
        initial_weights,
    )

    print("\nOptimal Weights by Period:")
    for period, weights in enumerate(weights_by_period):
        print(f"\n  Period {period + 1}:")
        for asset, weight in weights.items():
            print(f"    {asset:20s}: {weight:6.2%}")

    # Calculate turnover for each period
    print("\nPortfolio Turnover by Period:")
    prev_weights = initial_weights
    total_turnover = 0

    for period, weights in enumerate(weights_by_period):
        weights_series = pd.Series(weights)
        turnover = (weights_series - prev_weights).abs().sum() / 2
        total_turnover += turnover
        print(f"  Period {period + 1}: {turnover:.2%}")
        prev_weights = weights_series

    print(f"\nTotal Turnover: {total_turnover:.2%}")
    print(f"Total Transaction Costs: {total_turnover * mp_opt.transaction_cost:.2%}")

    # Compare with single-period optimization (ignoring transaction costs)
    print("\n--- Comparison with Single-Period Rebalancing ---")

    from allocation_station.optimization.optimizer import MeanVarianceOptimizer
    mv_opt = MeanVarianceOptimizer()

    single_period_weights = []
    for period_returns in expected_returns_list:
        weights = mv_opt.optimize(period_returns, covariance_matrix)
        single_period_weights.append(weights)

    prev_weights = initial_weights
    total_turnover_sp = 0

    for period, weights in enumerate(single_period_weights):
        weights_series = pd.Series(weights)
        turnover = (weights_series - prev_weights).abs().sum() / 2
        total_turnover_sp += turnover
        prev_weights = weights_series

    print(f"Single-Period Total Turnover: {total_turnover_sp:.2%}")
    print(f"Single-Period Transaction Costs: {total_turnover_sp * mp_opt.transaction_cost:.2%}")
    print(f"\nTurnover Reduction: {(total_turnover_sp - total_turnover):.2%}")
    print(f"Cost Savings: {(total_turnover_sp - total_turnover) * mp_opt.transaction_cost:.2%}")


def main():
    """Run all examples."""
    print("\n")
    print("#" * 80)
    print("# ADVANCED OPTIMIZATION EXAMPLES")
    print("#" * 80)
    print("\nThis script demonstrates advanced portfolio optimization techniques")
    print("implemented in Allocation Station.\n")

    try:
        example_1_black_litterman()
        example_2_robust_optimization()
        example_3_hierarchical_risk_parity()
        example_4_mean_cvar()
        example_5_kelly_criterion()
        example_6_custom_objectives()
        example_7_multi_period()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
