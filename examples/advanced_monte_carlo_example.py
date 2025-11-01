"""
Advanced Monte Carlo Simulation Examples

This script demonstrates the advanced Monte Carlo simulation capabilities:
1. Regime-switching models (Hidden Markov Models)
2. GARCH volatility modeling (time-varying volatility)
3. Copula-based dependency structures (flexible correlations)
4. Jump diffusion processes (Merton, Kou models)
5. Stochastic volatility models (Heston model)
6. Custom distributions (mixtures, fat tails)
7. Importance sampling (variance reduction for tail events)

Run this script to see each simulation method in action.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from allocation_station.simulation.advanced_monte_carlo import (
    RegimeSwitchingModel,
    RegimeParameters,
    MarketRegime,
    GARCHModel,
    GARCHVariant,
    CopulaSimulator,
    CopulaType,
    JumpDiffusionModel,
    JumpType,
    StochasticVolatilityModel,
    CustomDistribution,
    ImportanceSampler,
)


def example_1_regime_switching():
    """Example 1: Regime-switching model simulation."""
    print("=" * 80)
    print("EXAMPLE 1: Regime-Switching Model")
    print("=" * 80)

    # Define three market regimes: Bull, Bear, Sideways
    regimes = [
        RegimeParameters(
            regime=MarketRegime.BULL,
            mean_return=0.012,  # 1.2% monthly
            volatility=0.04,
            probability=0.45,
        ),
        RegimeParameters(
            regime=MarketRegime.BEAR,
            mean_return=-0.008,  # -0.8% monthly
            volatility=0.06,
            probability=0.25,
        ),
        RegimeParameters(
            regime=MarketRegime.SIDEWAYS,
            mean_return=0.002,  # 0.2% monthly
            volatility=0.03,
            probability=0.30,
        ),
    ]

    print("Market Regimes:")
    for reg in regimes:
        print(f"  {reg.regime.upper():10s}: Return={reg.mean_return:6.2%}, "
              f"Vol={reg.volatility:6.2%}, Prob={reg.probability:5.1%}")

    # Create transition matrix (custom)
    transition_matrix = np.array([
        # From Bull to: [Bull, Bear, Sideways]
        [0.85, 0.05, 0.10],
        # From Bear to: [Bull, Bear, Sideways]
        [0.15, 0.70, 0.15],
        # From Sideways to: [Bull, Bear, Sideways]
        [0.30, 0.10, 0.60],
    ])

    print("\nTransition Matrix:")
    print("         To: Bull    Bear    Sideways")
    regime_names = ['Bull', 'Bear', 'Sideways']
    for i, from_regime in enumerate(regime_names):
        print(f"  From {from_regime:8s}:", end='')
        for prob in transition_matrix[i]:
            print(f" {prob:6.1%}", end='')
        print()

    # Create model
    rsm = RegimeSwitchingModel(regimes, transition_matrix)

    # Simulate returns
    print("\n--- Simulating 60 months with 1000 paths ---")
    results = rsm.simulate_returns(
        n_periods=60,
        n_simulations=1000,
        initial_regime=0,  # Start in Bull
    )

    returns = results['returns']
    regime_paths = results['regimes']

    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod(axis=1) - 1

    print(f"\nSimulation Statistics:")
    print(f"  Average final return: {cumulative_returns[:, -1].mean():.2%}")
    print(f"  Median final return:  {np.median(cumulative_returns[:, -1]):.2%}")
    print(f"  5th percentile:       {np.percentile(cumulative_returns[:, -1], 5):.2%}")
    print(f"  95th percentile:      {np.percentile(cumulative_returns[:, -1], 95):.2%}")

    # Regime statistics
    print(f"\nRegime Statistics (across all paths):")
    for i, reg in enumerate(regimes):
        regime_pct = (regime_paths == i).sum() / regime_paths.size * 100
        print(f"  Time in {reg.regime.upper():10s}: {regime_pct:5.1%}")

    # Example single path
    print(f"\nExample Path 1 - First 12 months:")
    for month in range(12):
        regime_idx = regime_paths[0, month]
        regime_name = regimes[regime_idx].regime
        monthly_return = returns[0, month]
        print(f"  Month {month+1:2d}: {regime_name.upper():10s} return = {monthly_return:7.2%}")


def example_2_garch_volatility():
    """Example 2: GARCH volatility modeling."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: GARCH Volatility Modeling")
    print("=" * 80)

    # Generate historical returns with volatility clustering
    np.random.seed(42)
    n_obs = 500

    # Simulate "true" GARCH process for demonstration
    true_omega = 0.0001
    true_alpha = 0.15
    true_beta = 0.80

    historical_returns = np.zeros(n_obs)
    true_variance = np.zeros(n_obs)
    true_variance[0] = true_omega / (1 - true_alpha - true_beta)

    for t in range(n_obs):
        historical_returns[t] = np.random.normal(0, np.sqrt(true_variance[t]))
        if t < n_obs - 1:
            true_variance[t+1] = (
                true_omega +
                true_alpha * historical_returns[t]**2 +
                true_beta * true_variance[t]
            )

    returns_series = pd.Series(historical_returns)

    print(f"Historical Returns Summary:")
    print(f"  Mean:     {returns_series.mean():.4%}")
    print(f"  Std Dev:  {returns_series.std():.4%}")
    print(f"  Skewness: {returns_series.skew():.3f}")
    print(f"  Kurtosis: {returns_series.kurtosis():.3f}")

    # Fit GARCH(1,1) model
    print("\n--- Fitting GARCH(1,1) Model ---")
    garch = GARCHModel(variant=GARCHVariant.GARCH, p=1, q=1)
    params = garch.fit(returns_series)

    print(f"\nEstimated Parameters:")
    print(f"  omega: {params['omega']:.6f}  (true: {true_omega:.6f})")
    print(f"  alpha: {params['alpha']:.4f}  (true: {true_alpha:.4f})")
    print(f"  beta:  {params['beta']:.4f}  (true: {true_beta:.4f})")
    print(f"  alpha + beta: {params['alpha'] + params['beta']:.4f}  (persistence)")

    # Unconditional variance
    uncond_var = params['omega'] / (1 - params['alpha'] - params['beta'])
    print(f"\nUnconditional volatility: {np.sqrt(uncond_var):.4%}")

    # Simulate future returns
    print("\n--- Simulating 120 periods ahead ---")
    sim_results = garch.simulate(
        n_periods=120,
        n_simulations=1000,
    )

    sim_returns = sim_results['returns']
    sim_volatilities = sim_results['volatilities']

    print(f"\nSimulated Returns Statistics:")
    print(f"  Mean volatility (period 1):   {sim_volatilities[:, 0].mean():.4%}")
    print(f"  Mean volatility (period 60):  {sim_volatilities[:, 59].mean():.4%}")
    print(f"  Mean volatility (period 120): {sim_volatilities[:, 119].mean():.4%}")

    # Volatility clustering visible in simulations
    avg_vol_path = sim_volatilities.mean(axis=0)
    print(f"\nAverage volatility path converges to: {avg_vol_path[-1]:.4%}")

    # Fit GJR-GARCH (asymmetric)
    print("\n--- Fitting GJR-GARCH (Asymmetric) ---")
    gjr_garch = GARCHModel(variant=GARCHVariant.GJR_GARCH, p=1, q=1)
    gjr_params = gjr_garch.fit(returns_series)

    print(f"\nGJR-GARCH Parameters:")
    print(f"  omega:  {gjr_params['omega']:.6f}")
    print(f"  alpha:  {gjr_params['alpha']:.4f}")
    print(f"  gamma:  {gjr_params['gamma']:.4f}  (leverage effect)")
    print(f"  beta:   {gjr_params['beta']:.4f}")

    if gjr_params['gamma'] > 0:
        print(f"\n  Negative returns increase volatility by additional {gjr_params['gamma']:.4f}")


def example_3_copula_simulation():
    """Example 3: Copula-based multivariate simulation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Copula-Based Dependency Structures")
    print("=" * 80)

    n_assets = 3
    n_simulations = 10000

    # Correlation matrix
    correlation = np.array([
        [1.00, 0.60, 0.40],
        [0.60, 1.00, 0.50],
        [0.40, 0.50, 1.00],
    ])

    print(f"Correlation Matrix:")
    print(correlation)

    # 1. Gaussian Copula
    print("\n--- Gaussian Copula (Normal Tail Dependence) ---")
    gaussian_cop = CopulaSimulator(
        copula_type=CopulaType.GAUSSIAN,
        correlation_matrix=correlation,
    )

    # Define marginal distributions (different for each asset)
    marginals = [
        stats.norm(loc=0.08/12, scale=0.20/np.sqrt(12)),  # Stock: normal
        stats.norm(loc=0.04/12, scale=0.08/np.sqrt(12)),  # Bond: normal
        stats.t(df=5, loc=0.10/12, scale=0.25/np.sqrt(12)),  # Alternative: t-dist
    ]

    gaussian_samples = gaussian_cop.simulate(n_simulations, n_assets, marginals)

    print(f"\nGaussian Copula Results:")
    print(f"  Empirical correlation:")
    print(np.corrcoef(gaussian_samples.T))

    # 2. Student-t Copula (Tail Dependence)
    print("\n--- Student-t Copula (Symmetric Tail Dependence) ---")
    t_cop = CopulaSimulator(
        copula_type=CopulaType.T,
        correlation_matrix=correlation,
        df=5,  # Degrees of freedom (lower = stronger tail dependence)
    )

    t_samples = t_cop.simulate(n_simulations, n_assets, marginals)

    print(f"\nStudent-t Copula Results:")
    print(f"  Empirical correlation:")
    print(np.corrcoef(t_samples.T))

    # Compare tail dependence
    print("\n--- Tail Dependence Comparison ---")

    # Left tail (5th percentile)
    threshold = 0.05

    # Gaussian copula tail dependence
    asset1_low = gaussian_samples[:, 0] < np.percentile(gaussian_samples[:, 0], 5)
    asset2_low_given_1_gaussian = gaussian_samples[asset1_low, 1] < np.percentile(gaussian_samples[:, 1], 5)
    tail_dep_gaussian = asset2_low_given_1_gaussian.mean()

    # t-copula tail dependence
    asset1_low_t = t_samples[:, 0] < np.percentile(t_samples[:, 0], 5)
    asset2_low_given_1_t = t_samples[asset1_low_t, 1] < np.percentile(t_samples[:, 1], 5)
    tail_dep_t = asset2_low_given_1_t.mean()

    print(f"\nLeft Tail Dependence (P(Asset2 low | Asset1 low)):")
    print(f"  Gaussian Copula: {tail_dep_gaussian:.3f}")
    print(f"  t-Copula:        {tail_dep_t:.3f}")
    print(f"  >>> t-Copula shows stronger tail dependence (co-crashes)")

    # 3. Clayton Copula (Lower Tail Dependence)
    print("\n--- Clayton Copula (Lower Tail Dependence) ---")
    clayton_cop = CopulaSimulator(
        copula_type=CopulaType.CLAYTON,
        theta=2.0,  # Dependence parameter
    )

    clayton_samples = clayton_cop.simulate(n_simulations, 2)  # Only 2 assets

    print(f"\nClayton Copula (2 assets):")
    print(f"  Empirical correlation: {np.corrcoef(clayton_samples.T)[0, 1]:.3f}")
    print(f"  >>> Models downside co-movement better than upside")


def example_4_jump_diffusion():
    """Example 4: Jump diffusion processes."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Jump Diffusion Processes")
    print("=" * 80)

    # Merton jump diffusion parameters
    S0 = 100  # Initial price
    T = 1.0   # 1 year
    n_steps = 252  # Daily steps

    print("Merton (1976) Jump Diffusion Model")
    print(f"  Initial Price: ${S0}")
    print(f"  Time Horizon: {T} year")

    jd_model = JumpDiffusionModel(
        jump_type=JumpType.MERTON,
        drift=0.08,  # 8% annual drift
        volatility=0.20,  # 20% annual volatility
        jump_intensity=2.0,  # 2 jumps per year on average
        jump_mean=-0.03,  # Jumps are negative on average
        jump_std=0.05,  # Jump size volatility
    )

    print(f"\nModel Parameters:")
    print(f"  Continuous drift (mu):       {jd_model.drift:.2%}")
    print(f"  Continuous volatility (sigma): {jd_model.volatility:.2%}")
    print(f"  Jump intensity (lambda):     {jd_model.jump_intensity:.1f} jumps/year")
    print(f"  Jump mean:                   {jd_model.jump_mean:.2%}")
    print(f"  Jump std:                    {jd_model.jump_std:.2%}")

    # Simulate
    print(f"\n--- Simulating {n_steps} daily steps, 1000 paths ---")
    results = jd_model.simulate_paths(
        S0=S0,
        T=T,
        n_steps=n_steps,
        n_simulations=1000,
    )

    prices = results['prices']
    jump_times = results['jump_times']
    jump_sizes = results['jump_sizes']

    # Statistics
    final_prices = prices[:, -1]
    returns = final_prices / S0 - 1

    print(f"\nFinal Price Distribution:")
    print(f"  Mean:          ${final_prices.mean():.2f}")
    print(f"  Median:        ${np.median(final_prices):.2f}")
    print(f"  Std Dev:       ${final_prices.std():.2f}")
    print(f"  5th percentile: ${np.percentile(final_prices, 5):.2f}")
    print(f"  95th percentile: ${np.percentile(final_prices, 95):.2f}")

    print(f"\nReturn Distribution:")
    print(f"  Mean:     {returns.mean():.2%}")
    print(f"  Std Dev:  {returns.std():.2%}")
    print(f"  Skewness: {pd.Series(returns).skew():.3f}")
    print(f"  Kurtosis: {pd.Series(returns).kurtosis():.3f}")

    # Jump statistics
    total_jumps = sum(len(jt) for jt in jump_times)
    avg_jumps_per_path = total_jumps / len(jump_times)

    print(f"\nJump Statistics:")
    print(f"  Total jumps across all paths: {total_jumps}")
    print(f"  Average jumps per path:       {avg_jumps_per_path:.2f}")
    print(f"  Expected jumps per path:      {jd_model.jump_intensity * T:.2f}")

    # Example path with jumps
    path_idx = 0
    if len(jump_times[path_idx]) > 0:
        print(f"\nExample Path {path_idx} - Jumps:")
        for time, size in zip(jump_times[path_idx][:5], jump_sizes[path_idx][:5]):
            print(f"  Time {time:.3f}: Jump size = {size:.2%}")


def example_5_stochastic_volatility():
    """Example 5: Stochastic volatility (Heston) model."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Stochastic Volatility (Heston Model)")
    print("=" * 80)

    print("Heston Model:")
    print("  dS_t/S_t = r dt + sqrt(v_t) dW^S_t")
    print("  dv_t = kappa(theta - v_t) dt + sigma_v sqrt(v_t) dW^v_t")
    print("  where Corr(dW^S, dW^v) = rho")

    sv_model = StochasticVolatilityModel(
        model_type='heston',
        kappa=2.0,      # Mean reversion speed
        theta=0.04,     # Long-term variance (20% vol)
        sigma_v=0.3,    # Volatility of volatility
        rho=-0.7,       # Negative correlation (leverage effect)
        v0=0.06,        # Initial variance (24.5% vol)
    )

    print(f"\nModel Parameters:")
    print(f"  Mean reversion (kappa):   {sv_model.kappa}")
    print(f"  Long-term variance (theta): {sv_model.theta:.4f} ({np.sqrt(sv_model.theta):.2%} vol)")
    print(f"  Vol of vol (sigma_v):     {sv_model.sigma_v}")
    print(f"  Correlation (rho):        {sv_model.rho}")
    print(f"  Initial variance (v_0):   {sv_model.v0:.4f} ({np.sqrt(sv_model.v0):.2%} vol)")

    # Simulate
    S0 = 100
    r = 0.05  # Risk-free rate
    T = 1.0
    n_steps = 252

    print(f"\n--- Simulating {n_steps} steps over {T} year ---")
    results = sv_model.simulate_heston(
        S0=S0,
        r=r,
        T=T,
        n_steps=n_steps,
        n_simulations=1000,
    )

    prices = results['prices']
    volatilities = results['volatilities']

    # Final statistics
    final_prices = prices[:, -1]
    final_vols = volatilities[:, -1]

    print(f"\nFinal Price Distribution:")
    print(f"  Mean:   ${final_prices.mean():.2f}")
    print(f"  Median: ${np.median(final_prices):.2f}")
    print(f"  Std:    ${final_prices.std():.2f}")

    print(f"\nFinal Volatility Distribution:")
    print(f"  Mean:   {final_vols.mean():.2%}")
    print(f"  Median: {np.median(final_vols):.2%}")
    print(f"  Std:    {final_vols.std():.2%}")
    print(f"  Long-term vol: {np.sqrt(sv_model.theta):.2%}")

    # Volatility path analysis
    avg_vol_path = volatilities.mean(axis=0)

    print(f"\nVolatility Path (mean across simulations):")
    print(f"  Initial:     {avg_vol_path[0]:.2%}")
    print(f"  After 3m:    {avg_vol_path[63]:.2%}")
    print(f"  After 6m:    {avg_vol_path[126]:.2%}")
    print(f"  After 1y:    {avg_vol_path[-1]:.2%}")
    print(f"  >>> Volatility mean-reverts toward {np.sqrt(sv_model.theta):.2%}")

    # Leverage effect
    print(f"\nLeverage Effect (rho = {sv_model.rho}):")
    # Calculate correlation between returns and volatility changes
    returns_data = prices[:, 1:] / prices[:, :-1] - 1
    vol_changes = volatilities[:, 1:] - volatilities[:, :-1]

    empirical_corr = np.corrcoef(returns_data.flatten(), vol_changes.flatten())[0, 1]
    print(f"  Empirical correlation(returns, delta_vol): {empirical_corr:.3f}")
    print(f"  >>> Negative correlation = falling prices increase volatility")


def example_6_custom_distributions():
    """Example 6: Custom and mixture distributions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Custom Distributions")
    print("=" * 80)

    # 1. Student-t distribution (fat tails)
    print("--- Student-t Distribution (Fat Tails) ---")

    t_dist = CustomDistribution(
        distribution=stats.t,
        parameters={'df': 5, 'loc': 0.005, 'scale': 0.04},
    )

    t_samples = t_dist.sample(10000)

    print(f"\nStudent-t Distribution (df=5):")
    print(f"  Mean:     {t_samples.mean():.4%}")
    print(f"  Std Dev:  {t_samples.std():.4%}")
    print(f"  Skewness: {pd.Series(t_samples).skew():.3f}")
    print(f"  Kurtosis: {pd.Series(t_samples).kurtosis():.3f}")
    print(f"  >>> Kurtosis > 0 indicates fat tails")

    # Compare to normal
    normal_samples = np.random.normal(0.005, 0.04, 10000)
    print(f"\nNormal Distribution (for comparison):")
    print(f"  Kurtosis: {pd.Series(normal_samples).kurtosis():.3f}")

    # 2. Mixture distribution
    print("\n--- Mixture Distribution (Bimodal) ---")
    print("70% Normal Market + 30% Crisis")

    mixture = CustomDistribution.create_mixture_distribution([
        (stats.norm(loc=0.008, scale=0.03), 0.70),  # Normal market
        (stats.norm(loc=-0.05, scale=0.08), 0.30),  # Crisis
    ])

    mixture_samples = mixture.sample(10000)

    print(f"\nMixture Distribution:")
    print(f"  Mean:     {mixture_samples.mean():.4%}")
    print(f"  Std Dev:  {mixture_samples.std():.4%}")
    print(f"  Skewness: {pd.Series(mixture_samples).skew():.3f}")
    print(f"  Kurtosis: {pd.Series(mixture_samples).kurtosis():.3f}")
    print(f"  >>> Negative skew from crisis component")

    # Tail comparison
    var_95_mixture = np.percentile(mixture_samples, 5)
    var_95_normal = np.percentile(normal_samples, 5)

    print(f"\n5% VaR Comparison:")
    print(f"  Mixture:  {var_95_mixture:.2%}")
    print(f"  Normal:   {var_95_normal:.2%}")
    print(f"  Difference: {(var_95_mixture - var_95_normal):.2%}")


def example_7_importance_sampling():
    """Example 7: Importance sampling for tail events."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Importance Sampling (Variance Reduction)")
    print("=" * 80)

    print("Goal: Estimate probability of extreme loss (returns < -20%)")
    print("This is a rare event, so importance sampling is more efficient.\n")

    target_mean = 0.08
    target_std = 0.20
    threshold = -0.20  # Extreme loss threshold

    # Standard Monte Carlo
    print("--- Standard Monte Carlo ---")
    n_standard = 100000

    standard_samples = np.random.normal(target_mean, target_std, n_standard)
    standard_estimate = (standard_samples < threshold).mean()
    standard_se = np.sqrt(standard_estimate * (1 - standard_estimate) / n_standard)

    print(f"Samples: {n_standard}")
    print(f"Estimated P(return < -20%): {standard_estimate:.4%}")
    print(f"Standard Error:              {standard_se:.4%}")
    print(f"95% CI: [{(standard_estimate - 1.96*standard_se):.4%}, "
          f"{(standard_estimate + 1.96*standard_se):.4%}]")

    # Importance Sampling
    print("\n--- Importance Sampling ---")

    # Shift proposal distribution toward tail
    shift = -3.0  # Shift 3 std devs left

    importance_sampler = ImportanceSampler.create_tail_sampler(
        target_mean=target_mean,
        target_std=target_std,
        shift=shift,
    )

    # Estimator function
    def tail_probability(samples):
        return (samples < threshold).astype(float)

    n_importance = 10000  # Need fewer samples with importance sampling

    is_result = importance_sampler.sample(n_importance, tail_probability)

    print(f"Samples: {n_importance}")
    print(f"Estimated P(return < -20%): {is_result['estimate']:.4%}")
    print(f"Standard Error:              {is_result['standard_error']:.4%}")
    print(f"Effective Sample Size:       {is_result['effective_sample_size']:.0f}")

    # Efficiency comparison
    variance_ratio = standard_se**2 / is_result['standard_error']**2
    print(f"\nEfficiency Gain:")
    print(f"  Variance Ratio: {variance_ratio:.1f}x")
    print(f"  >>> Importance sampling achieves {variance_ratio:.1f}x lower variance")
    print(f"  >>> Equivalent to {variance_ratio * n_importance:.0f} standard MC samples")

    # True probability (for reference, using much larger sample)
    true_prob = stats.norm.cdf(threshold, target_mean, target_std)
    print(f"\nTrue Probability (analytical): {true_prob:.4%}")
    print(f"Standard MC error: {abs(standard_estimate - true_prob):.4%}")
    print(f"Importance Sampling error: {abs(is_result['estimate'] - true_prob):.4%}")


def main():
    """Run all examples."""
    print("\n")
    print("#" * 80)
    print("# ADVANCED MONTE CARLO SIMULATION EXAMPLES")
    print("#" * 80)
    print("\nThis script demonstrates advanced simulation techniques")
    print("implemented in Allocation Station.\n")

    try:
        example_1_regime_switching()
        example_2_garch_volatility()
        example_3_copula_simulation()
        example_4_jump_diffusion()
        example_5_stochastic_volatility()
        example_6_custom_distributions()
        example_7_importance_sampling()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
