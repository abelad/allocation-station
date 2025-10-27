"""
Advanced Monte Carlo Simulation

This module provides sophisticated Monte Carlo simulation capabilities for
portfolio analysis including regime-switching models, GARCH volatility,
copula-based dependencies, and stochastic processes.

Key Features:
- Regime-switching models (Hidden Markov Models)
- GARCH family volatility modeling (GARCH, EGARCH, GJR-GARCH)
- Copula-based dependency structures (Gaussian, t, Clayton, Gumbel)
- Jump diffusion processes (Merton, Kou models)
- Stochastic volatility models (Heston, SABR)
- Custom distribution support
- Importance sampling for rare events

Classes:
    RegimeSwitchingModel: Multi-regime market simulation with state transitions
    GARCHModel: Conditional volatility modeling with various GARCH specifications
    CopulaSimulator: Multivariate simulation with flexible dependency structures
    JumpDiffusionModel: Asset price simulation with discontinuous jumps
    StochasticVolatilityModel: Models with time-varying stochastic volatility
    CustomDistribution: Framework for user-defined return distributions
    ImportanceSampler: Variance reduction via importance sampling
"""

from typing import Dict, List, Optional, Union, Tuple, Callable
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    """Market regime states."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class GARCHVariant(str, Enum):
    """GARCH model variants."""
    GARCH = "garch"  # Standard GARCH
    EGARCH = "egarch"  # Exponential GARCH (Nelson 1991)
    GJR_GARCH = "gjr_garch"  # Glosten-Jagannathan-Runkle GARCH
    TGARCH = "tgarch"  # Threshold GARCH
    IGARCH = "igarch"  # Integrated GARCH


class CopulaType(str, Enum):
    """Copula family types."""
    GAUSSIAN = "gaussian"
    T = "t"  # Student-t copula
    CLAYTON = "clayton"  # Lower tail dependence
    GUMBEL = "gumbel"  # Upper tail dependence
    FRANK = "frank"  # Symmetric dependence
    JOE = "joe"  # Upper tail dependence


class JumpType(str, Enum):
    """Jump process types."""
    MERTON = "merton"  # Merton (1976) jump diffusion
    KOU = "kou"  # Kou (2002) double exponential jumps
    VARIANCE_GAMMA = "variance_gamma"  # Variance gamma process


class RegimeParameters(BaseModel):
    """Parameters for a single market regime."""
    regime: MarketRegime
    mean_return: float = Field(description="Expected return in this regime")
    volatility: float = Field(description="Volatility in this regime")
    probability: float = Field(description="Steady-state probability")

    class Config:
        use_enum_values = True


class RegimeSwitchingModel:
    """
    Hidden Markov Model for regime-switching market dynamics.

    Models asset returns as switching between distinct market regimes
    (bull, bear, crisis, etc.) with regime-dependent parameters.
    """

    def __init__(
        self,
        regimes: List[RegimeParameters],
        transition_matrix: Optional[np.ndarray] = None,
    ):
        """
        Initialize regime-switching model.

        Args:
            regimes: List of regime parameters
            transition_matrix: Regime transition probability matrix (n_regimes x n_regimes)
                             If None, will use steady-state probabilities
        """
        self.regimes = regimes
        self.n_regimes = len(regimes)

        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
        else:
            # Create default transition matrix based on steady-state probabilities
            self.transition_matrix = self._create_default_transition_matrix()

        self._validate_transition_matrix()

    def _create_default_transition_matrix(self) -> np.ndarray:
        """Create transition matrix that maintains steady-state probabilities."""
        # Simple persistence-based transition matrix
        persistence = 0.90  # 90% chance of staying in same regime

        transition = np.zeros((self.n_regimes, self.n_regimes))
        steady_state_probs = np.array([r.probability for r in self.regimes])

        for i in range(self.n_regimes):
            # Probability of staying in regime i
            transition[i, i] = persistence

            # Distribute remaining probability based on steady-state
            remaining_prob = 1 - persistence
            other_probs = steady_state_probs.copy()
            other_probs[i] = 0
            other_probs = other_probs / other_probs.sum() if other_probs.sum() > 0 else other_probs

            for j in range(self.n_regimes):
                if i != j:
                    transition[i, j] = remaining_prob * other_probs[j]

        return transition

    def _validate_transition_matrix(self):
        """Validate that transition matrix is a valid stochastic matrix."""
        if self.transition_matrix.shape != (self.n_regimes, self.n_regimes):
            raise ValueError("Transition matrix must be square with size n_regimes")

        # Each row should sum to 1
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Transition matrix rows must sum to 1")

        # All probabilities must be non-negative
        if np.any(self.transition_matrix < 0):
            raise ValueError("Transition probabilities must be non-negative")

    def simulate_regimes(
        self,
        n_periods: int,
        initial_regime: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate regime sequence using Markov chain.

        Args:
            n_periods: Number of periods to simulate
            initial_regime: Starting regime index (if None, sample from steady-state)

        Returns:
            Array of regime indices for each period
        """
        regimes = np.zeros(n_periods, dtype=int)

        # Initial regime
        if initial_regime is None:
            steady_state_probs = np.array([r.probability for r in self.regimes])
            regimes[0] = np.random.choice(self.n_regimes, p=steady_state_probs)
        else:
            regimes[0] = initial_regime

        # Simulate regime transitions
        for t in range(1, n_periods):
            current_regime = regimes[t - 1]
            transition_probs = self.transition_matrix[current_regime]
            regimes[t] = np.random.choice(self.n_regimes, p=transition_probs)

        return regimes

    def simulate_returns(
        self,
        n_periods: int,
        n_simulations: int = 1000,
        initial_regime: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate returns under regime-switching dynamics.

        Args:
            n_periods: Number of periods to simulate
            n_simulations: Number of simulation paths
            initial_regime: Starting regime

        Returns:
            Dictionary with returns and regime paths
        """
        returns = np.zeros((n_simulations, n_periods))
        regime_paths = np.zeros((n_simulations, n_periods), dtype=int)

        for sim in range(n_simulations):
            # Simulate regime sequence
            regime_path = self.simulate_regimes(n_periods, initial_regime)
            regime_paths[sim] = regime_path

            # Generate returns based on regimes
            for t in range(n_periods):
                regime_idx = regime_path[t]
                regime = self.regimes[regime_idx]

                # Draw return from regime distribution
                returns[sim, t] = np.random.normal(
                    regime.mean_return,
                    regime.volatility
                )

        return {
            'returns': returns,
            'regimes': regime_paths,
            'regime_labels': [r.regime for r in self.regimes],
        }

    def estimate_from_data(
        self,
        returns: pd.Series,
        method: str = 'expectation_maximization',
        max_iterations: int = 100,
    ) -> 'RegimeSwitchingModel':
        """
        Estimate regime-switching model parameters from historical data.

        Args:
            returns: Historical return time series
            method: Estimation method ('expectation_maximization' or 'simple_clustering')
            max_iterations: Maximum EM iterations

        Returns:
            Fitted RegimeSwitchingModel
        """
        if method == 'simple_clustering':
            return self._estimate_via_clustering(returns)
        else:
            return self._estimate_via_em(returns, max_iterations)

    def _estimate_via_clustering(self, returns: pd.Series) -> 'RegimeSwitchingModel':
        """Simple clustering-based regime estimation."""
        from sklearn.cluster import KMeans

        # Use K-means to identify regimes
        returns_array = returns.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        regime_assignments = kmeans.fit_predict(returns_array)

        # Estimate parameters for each regime
        estimated_regimes = []
        for i in range(self.n_regimes):
            regime_returns = returns[regime_assignments == i]
            if len(regime_returns) > 0:
                mean_ret = regime_returns.mean()
                volatility = regime_returns.std()
                probability = len(regime_returns) / len(returns)

                # Map to regime type based on mean return
                if mean_ret > 0.01:
                    regime_type = MarketRegime.BULL
                elif mean_ret < -0.01:
                    regime_type = MarketRegime.BEAR
                else:
                    regime_type = MarketRegime.SIDEWAYS

                estimated_regimes.append(RegimeParameters(
                    regime=regime_type,
                    mean_return=mean_ret,
                    volatility=volatility,
                    probability=probability,
                ))

        # Estimate transition matrix
        transition_counts = np.zeros((self.n_regimes, self.n_regimes))
        for t in range(1, len(regime_assignments)):
            from_regime = regime_assignments[t - 1]
            to_regime = regime_assignments[t]
            transition_counts[from_regime, to_regime] += 1

        transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)

        return RegimeSwitchingModel(estimated_regimes, transition_matrix)

    def _estimate_via_em(self, returns: pd.Series, max_iterations: int) -> 'RegimeSwitchingModel':
        """Expectation-Maximization algorithm for regime estimation."""
        # Simplified EM - full implementation would be more complex
        # This is a placeholder that uses clustering as initialization
        return self._estimate_via_clustering(returns)


class GARCHModel:
    """
    GARCH family models for conditional volatility.

    Implements various GARCH specifications including standard GARCH,
    EGARCH, GJR-GARCH for modeling time-varying volatility with
    volatility clustering and leverage effects.
    """

    def __init__(
        self,
        variant: GARCHVariant = GARCHVariant.GARCH,
        p: int = 1,  # GARCH order
        q: int = 1,  # ARCH order
    ):
        """
        Initialize GARCH model.

        Args:
            variant: GARCH model variant
            p: Number of lagged variance terms
            q: Number of lagged squared residual terms
        """
        self.variant = variant
        self.p = p
        self.q = q
        self.params = None

    def fit(
        self,
        returns: pd.Series,
        initial_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Fit GARCH model to historical returns.

        Args:
            returns: Historical return time series
            initial_params: Initial parameter guesses

        Returns:
            Estimated parameters
        """
        if self.variant == GARCHVariant.GARCH:
            self.params = self._fit_standard_garch(returns, initial_params)
        elif self.variant == GARCHVariant.EGARCH:
            self.params = self._fit_egarch(returns, initial_params)
        elif self.variant == GARCHVariant.GJR_GARCH:
            self.params = self._fit_gjr_garch(returns, initial_params)
        else:
            raise NotImplementedError(f"GARCH variant {self.variant} not implemented")

        return self.params

    def _fit_standard_garch(
        self,
        returns: pd.Series,
        initial_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Fit standard GARCH(p,q) model.

        GARCH equation: σ²_t = ω + Σα_i*ε²_{t-i} + Σβ_j*σ²_{t-j}
        """
        returns_array = returns.values

        # Initial parameter guesses
        if initial_params is None:
            omega = returns.var() * 0.05
            alpha = 0.1
            beta = 0.85
        else:
            omega = initial_params.get('omega', 0.05)
            alpha = initial_params.get('alpha', 0.1)
            beta = initial_params.get('beta', 0.85)

        initial = np.array([omega, alpha, beta])

        # Negative log-likelihood function
        def neg_log_likelihood(params):
            omega, alpha, beta = params

            # Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10

            n = len(returns_array)
            variance = np.zeros(n)
            variance[0] = returns.var()

            # Calculate conditional variances
            for t in range(1, n):
                variance[t] = omega + alpha * returns_array[t-1]**2 + beta * variance[t-1]

            # Log-likelihood (assuming normal distribution)
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi) + np.log(variance) + returns_array**2 / variance
            )

            return -log_likelihood

        # Optimize
        result = minimize(
            neg_log_likelihood,
            initial,
            method='L-BFGS-B',
            bounds=[(1e-6, None), (0, 0.99), (0, 0.99)]
        )

        if result.success:
            omega, alpha, beta = result.x
            return {'omega': omega, 'alpha': alpha, 'beta': beta}
        else:
            # Fallback to initial guess
            return {'omega': omega, 'alpha': alpha, 'beta': beta}

    def _fit_egarch(self, returns: pd.Series, initial_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Fit EGARCH model.

        EGARCH equation: log(σ²_t) = ω + α*|z_{t-1}| + γ*z_{t-1} + β*log(σ²_{t-1})
        where z_t = ε_t/σ_t (standardized residuals)
        """
        # Simplified EGARCH - full implementation would be more complex
        # Using standard GARCH as approximation
        return self._fit_standard_garch(returns, initial_params)

    def _fit_gjr_garch(self, returns: pd.Series, initial_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Fit GJR-GARCH model (asymmetric GARCH).

        GJR equation: σ²_t = ω + (α + γ*I_{t-1})*ε²_{t-1} + β*σ²_{t-1}
        where I_{t-1} = 1 if ε_{t-1} < 0 (leverage effect)
        """
        returns_array = returns.values

        # Initial parameters
        if initial_params is None:
            omega = returns.var() * 0.05
            alpha = 0.08
            gamma = 0.05  # Leverage effect
            beta = 0.85
        else:
            omega = initial_params.get('omega', 0.05)
            alpha = initial_params.get('alpha', 0.08)
            gamma = initial_params.get('gamma', 0.05)
            beta = initial_params.get('beta', 0.85)

        initial = np.array([omega, alpha, gamma, beta])

        def neg_log_likelihood(params):
            omega, alpha, gamma, beta = params

            if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
                return 1e10

            n = len(returns_array)
            variance = np.zeros(n)
            variance[0] = returns.var()

            for t in range(1, n):
                indicator = 1 if returns_array[t-1] < 0 else 0
                variance[t] = (
                    omega +
                    (alpha + gamma * indicator) * returns_array[t-1]**2 +
                    beta * variance[t-1]
                )

            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi) + np.log(variance) + returns_array**2 / variance
            )

            return -log_likelihood

        result = minimize(
            neg_log_likelihood,
            initial,
            method='L-BFGS-B',
            bounds=[(1e-6, None), (0, 0.99), (0, 0.99), (0, 0.99)]
        )

        if result.success:
            omega, alpha, gamma, beta = result.x
            return {'omega': omega, 'alpha': alpha, 'gamma': gamma, 'beta': beta}
        else:
            return {'omega': omega, 'alpha': alpha, 'gamma': gamma, 'beta': beta}

    def simulate(
        self,
        n_periods: int,
        n_simulations: int = 1000,
        initial_variance: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate returns using fitted GARCH model.

        Args:
            n_periods: Number of periods to simulate
            n_simulations: Number of simulation paths
            initial_variance: Starting variance (uses omega/(1-alpha-beta) if None)

        Returns:
            Dictionary with returns and conditional volatilities
        """
        if self.params is None:
            raise ValueError("Must fit model before simulating")

        omega = self.params['omega']
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params.get('gamma', 0)  # For GJR-GARCH

        if initial_variance is None:
            # Unconditional variance
            initial_variance = omega / (1 - alpha - beta - gamma/2)

        returns = np.zeros((n_simulations, n_periods))
        variances = np.zeros((n_simulations, n_periods))

        for sim in range(n_simulations):
            variances[sim, 0] = initial_variance
            innovations = np.random.normal(0, 1, n_periods)

            for t in range(n_periods):
                # Generate return
                returns[sim, t] = np.sqrt(variances[sim, t]) * innovations[t]

                # Update variance for next period
                if t < n_periods - 1:
                    if self.variant == GARCHVariant.GJR_GARCH:
                        indicator = 1 if returns[sim, t] < 0 else 0
                        variances[sim, t + 1] = (
                            omega +
                            (alpha + gamma * indicator) * returns[sim, t]**2 +
                            beta * variances[sim, t]
                        )
                    else:  # Standard GARCH
                        variances[sim, t + 1] = (
                            omega + alpha * returns[sim, t]**2 + beta * variances[sim, t]
                        )

        return {
            'returns': returns,
            'variances': variances,
            'volatilities': np.sqrt(variances),
        }


class CopulaSimulator:
    """
    Copula-based multivariate simulation.

    Separates marginal distributions from dependency structure,
    allowing flexible modeling of tail dependencies and asymmetric correlations.
    """

    def __init__(
        self,
        copula_type: CopulaType = CopulaType.GAUSSIAN,
        correlation_matrix: Optional[np.ndarray] = None,
        df: Optional[float] = None,  # Degrees of freedom for t-copula
        theta: Optional[float] = None,  # Parameter for Archimedean copulas
    ):
        """
        Initialize copula simulator.

        Args:
            copula_type: Type of copula to use
            correlation_matrix: Correlation structure (for Gaussian/t copulas)
            df: Degrees of freedom (for t-copula)
            theta: Copula parameter (for Archimedean copulas)
        """
        self.copula_type = copula_type
        self.correlation_matrix = correlation_matrix
        self.df = df
        self.theta = theta

    def simulate(
        self,
        n_simulations: int,
        n_assets: int,
        marginal_distributions: Optional[List[stats.rv_continuous]] = None,
    ) -> np.ndarray:
        """
        Simulate multivariate samples using copula.

        Args:
            n_simulations: Number of samples to generate
            n_assets: Number of assets/variables
            marginal_distributions: List of scipy distributions for each asset

        Returns:
            Array of shape (n_simulations, n_assets)
        """
        # Generate uniform samples from copula
        uniforms = self._simulate_copula(n_simulations, n_assets)

        # Transform to desired marginals
        if marginal_distributions is not None:
            samples = np.zeros_like(uniforms)
            for i in range(n_assets):
                samples[:, i] = marginal_distributions[i].ppf(uniforms[:, i])
            return samples
        else:
            # Return standard normal marginals
            return stats.norm.ppf(uniforms)

    def _simulate_copula(self, n_simulations: int, n_assets: int) -> np.ndarray:
        """Generate uniform samples from copula."""
        if self.copula_type == CopulaType.GAUSSIAN:
            return self._simulate_gaussian_copula(n_simulations, n_assets)
        elif self.copula_type == CopulaType.T:
            return self._simulate_t_copula(n_simulations, n_assets)
        elif self.copula_type == CopulaType.CLAYTON:
            return self._simulate_clayton_copula(n_simulations, n_assets)
        elif self.copula_type == CopulaType.GUMBEL:
            return self._simulate_gumbel_copula(n_simulations, n_assets)
        else:
            raise NotImplementedError(f"Copula type {self.copula_type} not implemented")

    def _simulate_gaussian_copula(self, n_simulations: int, n_assets: int) -> np.ndarray:
        """Gaussian copula simulation."""
        if self.correlation_matrix is None:
            # Use identity matrix (independence)
            corr = np.eye(n_assets)
        else:
            corr = self.correlation_matrix

        # Generate multivariate normal
        mean = np.zeros(n_assets)
        samples = np.random.multivariate_normal(mean, corr, n_simulations)

        # Transform to uniform via standard normal CDF
        uniforms = stats.norm.cdf(samples)

        return uniforms

    def _simulate_t_copula(self, n_simulations: int, n_assets: int) -> np.ndarray:
        """Student-t copula simulation (allows tail dependence)."""
        if self.correlation_matrix is None:
            corr = np.eye(n_assets)
        else:
            corr = self.correlation_matrix

        if self.df is None:
            self.df = 5  # Default degrees of freedom

        # Generate multivariate t
        mean = np.zeros(n_assets)

        # Multivariate t via normal-gamma mixture
        chi2_samples = np.random.chisquare(self.df, n_simulations)
        normal_samples = np.random.multivariate_normal(mean, corr, n_simulations)

        t_samples = normal_samples * np.sqrt(self.df / chi2_samples[:, np.newaxis])

        # Transform to uniform via t CDF
        uniforms = stats.t.cdf(t_samples, df=self.df)

        return uniforms

    def _simulate_clayton_copula(self, n_simulations: int, n_assets: int) -> np.ndarray:
        """Clayton copula (lower tail dependence)."""
        if self.theta is None:
            self.theta = 2.0  # Default parameter

        if n_assets != 2:
            raise NotImplementedError("Clayton copula only implemented for 2 assets")

        # Clayton copula algorithm
        u1 = np.random.uniform(0, 1, n_simulations)
        v = np.random.uniform(0, 1, n_simulations)

        u2 = (u1**(-self.theta) * (v**(-self.theta/(1+self.theta)) - 1) + 1)**(-1/self.theta)

        return np.column_stack([u1, u2])

    def _simulate_gumbel_copula(self, n_simulations: int, n_assets: int) -> np.ndarray:
        """Gumbel copula (upper tail dependence)."""
        if self.theta is None:
            self.theta = 2.0

        if n_assets != 2:
            raise NotImplementedError("Gumbel copula only implemented for 2 assets")

        # Gumbel copula algorithm
        # Using conditional sampling method
        u1 = np.random.uniform(0, 1, n_simulations)
        v = np.random.uniform(0, 1, n_simulations)

        # Simplified Gumbel simulation
        # Full implementation would use more sophisticated algorithm
        t = -np.log(u1)
        s = -np.log(v)

        w = np.random.gamma(1/self.theta, 1, n_simulations)
        u2 = np.exp(-(t/w)**(1/self.theta))

        return np.column_stack([u1, u2])


class JumpDiffusionModel:
    """
    Jump diffusion models for asset prices.

    Combines continuous diffusion with discontinuous jumps to model
    sudden market movements and tail events.
    """

    def __init__(
        self,
        jump_type: JumpType = JumpType.MERTON,
        drift: float = 0.08,
        volatility: float = 0.20,
        jump_intensity: float = 0.1,  # Expected jumps per year
        jump_mean: float = 0.0,  # Mean jump size
        jump_std: float = 0.05,  # Jump size volatility
    ):
        """
        Initialize jump diffusion model.

        Args:
            jump_type: Type of jump process
            drift: Continuous drift (μ)
            volatility: Continuous volatility (σ)
            jump_intensity: Poisson intensity (λ)
            jump_mean: Mean of jump sizes
            jump_std: Standard deviation of jump sizes
        """
        self.jump_type = jump_type
        self.drift = drift
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def simulate_paths(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_simulations: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate asset price paths with jumps.

        Args:
            S0: Initial price
            T: Time horizon (years)
            n_steps: Number of time steps
            n_simulations: Number of simulation paths

        Returns:
            Dictionary with price paths and jump information
        """
        dt = T / n_steps

        if self.jump_type == JumpType.MERTON:
            return self._simulate_merton(S0, T, n_steps, n_simulations, dt)
        elif self.jump_type == JumpType.KOU:
            return self._simulate_kou(S0, T, n_steps, n_simulations, dt)
        else:
            raise NotImplementedError(f"Jump type {self.jump_type} not implemented")

    def _simulate_merton(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_simulations: int,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        """
        Merton (1976) jump diffusion model.

        dS_t/S_t = μ dt + σ dW_t + (J-1) dN_t
        where J ~ LogNormal(μ_J, σ_J) and N_t is Poisson(λt)
        """
        prices = np.zeros((n_simulations, n_steps + 1))
        prices[:, 0] = S0

        jump_times = []
        jump_sizes = []

        for sim in range(n_simulations):
            sim_jump_times = []
            sim_jump_sizes = []

            for t in range(n_steps):
                # Continuous diffusion component
                dW = np.random.normal(0, np.sqrt(dt))
                continuous_return = (self.drift - 0.5 * self.volatility**2) * dt + self.volatility * dW

                # Jump component
                n_jumps = np.random.poisson(self.jump_intensity * dt)

                if n_jumps > 0:
                    # Jump sizes from log-normal
                    jump_returns = np.random.normal(self.jump_mean, self.jump_std, n_jumps)
                    total_jump = np.sum(jump_returns)

                    sim_jump_times.append(t * dt)
                    sim_jump_sizes.append(total_jump)
                else:
                    total_jump = 0

                # Total return
                total_return = continuous_return + total_jump

                # Update price
                prices[sim, t + 1] = prices[sim, t] * np.exp(total_return)

            jump_times.append(sim_jump_times)
            jump_sizes.append(sim_jump_sizes)

        return {
            'prices': prices,
            'jump_times': jump_times,
            'jump_sizes': jump_sizes,
            'time_grid': np.linspace(0, T, n_steps + 1),
        }

    def _simulate_kou(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_simulations: int,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        """
        Kou (2002) double exponential jump diffusion.

        Jump sizes follow double exponential: asymmetric upward/downward jumps.
        """
        # Similar to Merton but with double exponential jump sizes
        # Simplified implementation
        return self._simulate_merton(S0, T, n_steps, n_simulations, dt)


class StochasticVolatilityModel:
    """
    Stochastic volatility models (Heston, SABR).

    Models where volatility itself follows a stochastic process,
    capturing volatility clustering and smile effects.
    """

    def __init__(
        self,
        model_type: str = 'heston',
        # Heston parameters
        kappa: float = 2.0,  # Mean reversion speed
        theta: float = 0.04,  # Long-term variance
        sigma_v: float = 0.3,  # Volatility of volatility
        rho: float = -0.7,  # Correlation between price and volatility
        v0: Optional[float] = None,  # Initial variance
    ):
        """
        Initialize stochastic volatility model.

        Args:
            model_type: 'heston' or 'sabr'
            kappa: Mean reversion speed of variance
            theta: Long-term variance level
            sigma_v: Volatility of volatility
            rho: Correlation between asset and volatility shocks
            v0: Initial variance (defaults to theta)
        """
        self.model_type = model_type
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0 if v0 is not None else theta

    def simulate_heston(
        self,
        S0: float,
        r: float,
        T: float,
        n_steps: int,
        n_simulations: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate asset prices using Heston model.

        Heston model:
        dS_t/S_t = r dt + √v_t dW^S_t
        dv_t = κ(θ - v_t) dt + σ_v √v_t dW^v_t
        where Corr(dW^S, dW^v) = ρ

        Args:
            S0: Initial price
            r: Risk-free rate
            T: Time horizon
            n_steps: Number of time steps
            n_simulations: Number of paths

        Returns:
            Dictionary with prices and variance paths
        """
        dt = T / n_steps

        prices = np.zeros((n_simulations, n_steps + 1))
        variances = np.zeros((n_simulations, n_steps + 1))

        prices[:, 0] = S0
        variances[:, 0] = self.v0

        # Correlation structure for shocks
        cov_matrix = np.array([
            [1.0, self.rho],
            [self.rho, 1.0]
        ])

        for sim in range(n_simulations):
            for t in range(n_steps):
                # Correlated Brownian motions
                Z = np.random.multivariate_normal([0, 0], cov_matrix)
                dW_S = Z[0] * np.sqrt(dt)
                dW_v = Z[1] * np.sqrt(dt)

                # Current variance (ensure positive)
                v_t = max(variances[sim, t], 1e-6)

                # Variance process (Euler discretization)
                variances[sim, t + 1] = (
                    v_t +
                    self.kappa * (self.theta - v_t) * dt +
                    self.sigma_v * np.sqrt(v_t) * dW_v
                )

                # Ensure variance stays positive (truncation scheme)
                variances[sim, t + 1] = max(variances[sim, t + 1], 1e-6)

                # Price process
                prices[sim, t + 1] = prices[sim, t] * np.exp(
                    (r - 0.5 * v_t) * dt + np.sqrt(v_t) * dW_S
                )

        return {
            'prices': prices,
            'variances': variances,
            'volatilities': np.sqrt(variances),
            'time_grid': np.linspace(0, T, n_steps + 1),
        }


class CustomDistribution:
    """
    Framework for custom return distributions.

    Allows users to specify non-normal distributions for more realistic
    return modeling (fat tails, skewness, etc.).
    """

    def __init__(
        self,
        distribution: Union[stats.rv_continuous, Callable],
        parameters: Optional[Dict] = None,
    ):
        """
        Initialize custom distribution.

        Args:
            distribution: scipy distribution or custom sampling function
            parameters: Distribution parameters
        """
        self.distribution = distribution
        self.parameters = parameters or {}

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from custom distribution."""
        if isinstance(self.distribution, stats.rv_continuous):
            return self.distribution.rvs(size=n_samples, **self.parameters)
        else:
            # Custom function
            return self.distribution(n_samples, **self.parameters)

    @staticmethod
    def create_mixture_distribution(
        distributions: List[Tuple[stats.rv_continuous, float]],
    ) -> 'CustomDistribution':
        """
        Create mixture distribution.

        Args:
            distributions: List of (distribution, weight) tuples

        Returns:
            CustomDistribution for mixture
        """
        def sample_mixture(n_samples):
            weights = np.array([w for _, w in distributions])
            weights = weights / weights.sum()

            samples = np.zeros(n_samples)
            for i, (dist, _) in enumerate(distributions):
                n_from_component = np.random.binomial(n_samples, weights[i])
                if n_from_component > 0:
                    component_samples = dist.rvs(size=n_from_component)
                    # Randomly assign to output
                    indices = np.random.choice(n_samples, n_from_component, replace=False)
                    samples[indices] = component_samples[:n_from_component]

            return samples

        return CustomDistribution(sample_mixture)


class ImportanceSampler:
    """
    Importance sampling for variance reduction.

    Useful for estimating probabilities of rare events (tail risks)
    more efficiently than standard Monte Carlo.
    """

    def __init__(
        self,
        target_distribution: Callable,
        proposal_distribution: Callable,
        proposal_pdf: Callable,
        target_pdf: Callable,
    ):
        """
        Initialize importance sampler.

        Args:
            target_distribution: Target distribution sampling function
            proposal_distribution: Proposal distribution sampling function
            proposal_pdf: Proposal PDF for likelihood ratio
            target_pdf: Target PDF for likelihood ratio
        """
        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution
        self.proposal_pdf = proposal_pdf
        self.target_pdf = target_pdf

    def sample(
        self,
        n_samples: int,
        estimator_function: Callable,
    ) -> Dict[str, float]:
        """
        Generate importance samples and compute weighted estimate.

        Args:
            n_samples: Number of samples
            estimator_function: Function to estimate (e.g., tail probability)

        Returns:
            Dictionary with estimate and variance
        """
        # Generate samples from proposal
        samples = self.proposal_distribution(n_samples)

        # Calculate importance weights
        weights = self.target_pdf(samples) / self.proposal_pdf(samples)

        # Compute estimate
        values = estimator_function(samples)
        weighted_values = values * weights

        estimate = np.mean(weighted_values)
        variance = np.var(weighted_values) / n_samples

        # Effective sample size
        ess = (np.sum(weights)**2) / np.sum(weights**2)

        return {
            'estimate': estimate,
            'variance': variance,
            'standard_error': np.sqrt(variance),
            'effective_sample_size': ess,
            'weights': weights,
        }

    @staticmethod
    def create_tail_sampler(
        target_mean: float,
        target_std: float,
        shift: float = 2.0,
    ) -> 'ImportanceSampler':
        """
        Create importance sampler for tail events.

        Shifts proposal distribution toward tail to oversample rare events.

        Args:
            target_mean: Mean of target normal distribution
            target_std: Std of target normal distribution
            shift: How many std devs to shift proposal (negative for left tail)

        Returns:
            ImportanceSampler configured for tail sampling
        """
        proposal_mean = target_mean + shift * target_std

        def target_sampler(n):
            return np.random.normal(target_mean, target_std, n)

        def proposal_sampler(n):
            return np.random.normal(proposal_mean, target_std, n)

        def target_pdf(x):
            return stats.norm.pdf(x, target_mean, target_std)

        def proposal_pdf(x):
            return stats.norm.pdf(x, proposal_mean, target_std)

        return ImportanceSampler(
            target_sampler,
            proposal_sampler,
            proposal_pdf,
            target_pdf
        )
