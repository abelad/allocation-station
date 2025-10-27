"""Asset class correlation models and analysis."""

from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from pydantic import BaseModel, Field
import warnings


class CorrelationMethod(str, Enum):
    """Methods for calculating correlation."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class CorrelationRegime(str, Enum):
    """Market correlation regimes."""
    LOW = "low"  # < 0.3
    MODERATE = "moderate"  # 0.3 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    EXTREME = "extreme"  # > 0.8


class AssetClassCorrelationModel(BaseModel):
    """
    Model for asset class correlations.

    Provides default correlation matrices and methods for estimating
    correlations between different asset classes.
    """

    # Default long-term correlations (based on historical data)
    default_correlations: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            'equity': {
                'equity': 1.0,
                'bond': -0.15,
                'cash': 0.05,
                'commodity': 0.25,
                'real_estate': 0.60,
                'cryptocurrency': 0.30,
            },
            'bond': {
                'equity': -0.15,
                'bond': 1.0,
                'cash': 0.20,
                'commodity': -0.10,
                'real_estate': 0.10,
                'cryptocurrency': -0.05,
            },
            'cash': {
                'equity': 0.05,
                'bond': 0.20,
                'cash': 1.0,
                'commodity': 0.0,
                'real_estate': 0.05,
                'cryptocurrency': 0.0,
            },
            'commodity': {
                'equity': 0.25,
                'bond': -0.10,
                'cash': 0.0,
                'commodity': 1.0,
                'real_estate': 0.20,
                'cryptocurrency': 0.15,
            },
            'real_estate': {
                'equity': 0.60,
                'bond': 0.10,
                'cash': 0.05,
                'commodity': 0.20,
                'real_estate': 1.0,
                'cryptocurrency': 0.25,
            },
            'cryptocurrency': {
                'equity': 0.30,
                'bond': -0.05,
                'cash': 0.0,
                'commodity': 0.15,
                'real_estate': 0.25,
                'cryptocurrency': 1.0,
            },
        },
        description="Default correlation matrix"
    )

    # Stress scenario correlations
    stress_correlations: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            'equity': {
                'equity': 1.0,
                'bond': -0.30,
                'cash': 0.10,
                'commodity': 0.50,
                'real_estate': 0.75,
                'cryptocurrency': 0.55,
            },
            'bond': {
                'equity': -0.30,
                'bond': 1.0,
                'cash': 0.30,
                'commodity': -0.20,
                'real_estate': -0.10,
                'cryptocurrency': -0.15,
            },
            'cash': {
                'equity': 0.10,
                'bond': 0.30,
                'cash': 1.0,
                'commodity': 0.05,
                'real_estate': 0.10,
                'cryptocurrency': 0.05,
            },
            'commodity': {
                'equity': 0.50,
                'bond': -0.20,
                'cash': 0.05,
                'commodity': 1.0,
                'real_estate': 0.40,
                'cryptocurrency': 0.35,
            },
            'real_estate': {
                'equity': 0.75,
                'bond': -0.10,
                'cash': 0.10,
                'commodity': 0.40,
                'real_estate': 1.0,
                'cryptocurrency': 0.50,
            },
            'cryptocurrency': {
                'equity': 0.55,
                'bond': -0.15,
                'cash': 0.05,
                'commodity': 0.35,
                'real_estate': 0.50,
                'cryptocurrency': 1.0,
            },
        },
        description="Correlation matrix during stress periods"
    )

    class Config:
        arbitrary_types_allowed = True

    def get_correlation(
        self,
        asset_class_1: str,
        asset_class_2: str,
        stress_mode: bool = False
    ) -> float:
        """
        Get correlation between two asset classes.

        Args:
            asset_class_1: First asset class
            asset_class_2: Second asset class
            stress_mode: Use stress correlations

        Returns:
            Correlation coefficient
        """
        corr_matrix = self.stress_correlations if stress_mode else self.default_correlations

        asset_class_1 = asset_class_1.lower()
        asset_class_2 = asset_class_2.lower()

        if asset_class_1 in corr_matrix and asset_class_2 in corr_matrix[asset_class_1]:
            return corr_matrix[asset_class_1][asset_class_2]

        # Default to 0 if not found
        return 0.0

    def get_correlation_matrix(
        self,
        asset_classes: List[str],
        stress_mode: bool = False
    ) -> pd.DataFrame:
        """
        Get correlation matrix for specified asset classes.

        Args:
            asset_classes: List of asset classes
            stress_mode: Use stress correlations

        Returns:
            Correlation matrix as DataFrame
        """
        n = len(asset_classes)
        matrix = np.zeros((n, n))

        for i, ac1 in enumerate(asset_classes):
            for j, ac2 in enumerate(asset_classes):
                matrix[i, j] = self.get_correlation(ac1, ac2, stress_mode)

        return pd.DataFrame(matrix, index=asset_classes, columns=asset_classes)

    def estimate_correlation_breakdown(
        self,
        returns: pd.DataFrame,
        window: int = 252,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Detect correlation breakdown periods.

        Args:
            returns: DataFrame of asset returns
            window: Rolling window size
            threshold: Correlation threshold for breakdown

        Returns:
            DataFrame with correlation breakdown indicators
        """
        # Calculate rolling correlations
        rolling_corr = returns.rolling(window).corr()

        # Identify breakdown periods (correlation below threshold)
        breakdown_periods = []

        for date in returns.index[window:]:
            correlations = rolling_corr.loc[date]
            # Average pairwise correlation
            avg_corr = correlations.values[np.triu_indices_from(correlations.values, k=1)].mean()

            breakdown_periods.append({
                'date': date,
                'avg_correlation': avg_corr,
                'breakdown': avg_corr < threshold
            })

        return pd.DataFrame(breakdown_periods)


class DynamicCorrelationModel:
    """
    Dynamic Conditional Correlation (DCC) model.

    Estimates time-varying correlations between assets.
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize DCC model.

        Args:
            returns: DataFrame of asset returns
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)

    def calculate_rolling_correlation(
        self,
        window: int = 60,
        method: CorrelationMethod = CorrelationMethod.PEARSON
    ) -> Dict[Tuple[str, str], pd.Series]:
        """
        Calculate rolling correlations between all asset pairs.

        Args:
            window: Rolling window size
            method: Correlation method

        Returns:
            Dictionary mapping asset pairs to correlation time series
        """
        correlations = {}

        for i, asset1 in enumerate(self.assets):
            for j, asset2 in enumerate(self.assets):
                if i < j:  # Only calculate upper triangle
                    rolling_corr = self.returns[asset1].rolling(window).corr(
                        self.returns[asset2]
                    )
                    correlations[(asset1, asset2)] = rolling_corr

        return correlations

    def calculate_ewma_correlation(
        self,
        halflife: int = 30
    ) -> pd.DataFrame:
        """
        Calculate exponentially weighted moving average correlation.

        Args:
            halflife: Half-life for exponential weighting

        Returns:
            Current correlation matrix
        """
        ewm_cov = self.returns.ewm(halflife=halflife).cov()

        # Get the most recent correlation matrix
        latest_cov = ewm_cov.iloc[-self.n_assets:]

        # Convert covariance to correlation
        std = np.sqrt(np.diag(latest_cov))
        corr = latest_cov / np.outer(std, std)

        return corr

    def detect_correlation_regimes(
        self,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Detect different correlation regimes over time.

        Args:
            window: Window for regime detection

        Returns:
            DataFrame with regime classifications
        """
        # Calculate average rolling correlation
        rolling_corrs = self.calculate_rolling_correlation(window)

        regime_data = []

        for idx in self.returns.index[window:]:
            # Get correlations at this point
            corr_values = [corr.loc[idx] for corr in rolling_corrs.values() if idx in corr.index]

            if corr_values:
                avg_corr = np.mean(corr_values)

                # Classify regime
                if avg_corr < 0.3:
                    regime = CorrelationRegime.LOW
                elif avg_corr < 0.6:
                    regime = CorrelationRegime.MODERATE
                elif avg_corr < 0.8:
                    regime = CorrelationRegime.HIGH
                else:
                    regime = CorrelationRegime.EXTREME

                regime_data.append({
                    'date': idx,
                    'avg_correlation': avg_corr,
                    'regime': regime.value
                })

        return pd.DataFrame(regime_data)


class HierarchicalCorrelationModel:
    """
    Hierarchical clustering of assets based on correlations.

    Useful for understanding asset relationships and diversification.
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize hierarchical model.

        Args:
            returns: DataFrame of asset returns
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.correlation_matrix = returns.corr()

    def calculate_distance_matrix(self) -> np.ndarray:
        """
        Calculate distance matrix from correlations.

        Returns:
            Distance matrix
        """
        # Convert correlation to distance: distance = sqrt(0.5 * (1 - correlation))
        distance = np.sqrt(0.5 * (1 - self.correlation_matrix))
        return distance

    def get_linkage_matrix(self, method: str = 'ward') -> np.ndarray:
        """
        Calculate hierarchical clustering linkage.

        Args:
            method: Linkage method ('ward', 'single', 'complete', 'average')

        Returns:
            Linkage matrix
        """
        distance = self.calculate_distance_matrix()
        distance_condensed = squareform(distance, checks=False)
        linkage = hierarchy.linkage(distance_condensed, method=method)
        return linkage

    def get_clusters(self, n_clusters: int) -> Dict[str, int]:
        """
        Get asset clusters.

        Args:
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping assets to cluster IDs
        """
        linkage = self.get_linkage_matrix()
        clusters = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')

        return {asset: int(cluster) for asset, cluster in zip(self.assets, clusters)}

    def get_diversification_score(self) -> float:
        """
        Calculate diversification score based on correlation structure.

        Returns:
            Diversification score (0-1, higher is better)
        """
        # Average pairwise correlation
        n = len(self.assets)
        if n < 2:
            return 1.0

        # Get upper triangle correlations
        corr_values = self.correlation_matrix.values[np.triu_indices_from(
            self.correlation_matrix.values, k=1
        )]

        avg_corr = np.mean(np.abs(corr_values))

        # Diversification score: 1 - average correlation
        diversification = 1 - avg_corr

        return max(0, min(1, diversification))


class CopulaCorrelationModel:
    """
    Copula-based correlation model for capturing tail dependencies.

    More sophisticated than linear correlation for extreme events.
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize copula model.

        Args:
            returns: DataFrame of asset returns
        """
        self.returns = returns
        self.assets = returns.columns.tolist()

    def calculate_tail_dependence(
        self,
        asset1: str,
        asset2: str,
        quantile: float = 0.05
    ) -> Tuple[float, float]:
        """
        Calculate upper and lower tail dependence.

        Args:
            asset1: First asset
            asset2: Second asset
            quantile: Quantile for tail analysis

        Returns:
            Tuple of (lower_tail_dependence, upper_tail_dependence)
        """
        r1 = self.returns[asset1]
        r2 = self.returns[asset2]

        # Lower tail (both assets in bottom quantile)
        lower_threshold_1 = r1.quantile(quantile)
        lower_threshold_2 = r2.quantile(quantile)

        both_lower = ((r1 <= lower_threshold_1) & (r2 <= lower_threshold_2)).sum()
        either_lower = ((r1 <= lower_threshold_1) | (r2 <= lower_threshold_2)).sum()

        lower_tail_dep = both_lower / either_lower if either_lower > 0 else 0

        # Upper tail (both assets in top quantile)
        upper_threshold_1 = r1.quantile(1 - quantile)
        upper_threshold_2 = r2.quantile(1 - quantile)

        both_upper = ((r1 >= upper_threshold_1) & (r2 >= upper_threshold_2)).sum()
        either_upper = ((r1 >= upper_threshold_1) | (r2 >= upper_threshold_2)).sum()

        upper_tail_dep = both_upper / either_upper if either_upper > 0 else 0

        return lower_tail_dep, upper_tail_dep

    def calculate_all_tail_dependencies(
        self,
        quantile: float = 0.05
    ) -> pd.DataFrame:
        """
        Calculate tail dependencies for all asset pairs.

        Args:
            quantile: Quantile for tail analysis

        Returns:
            DataFrame with tail dependency coefficients
        """
        results = []

        for i, asset1 in enumerate(self.assets):
            for j, asset2 in enumerate(self.assets):
                if i < j:
                    lower, upper = self.calculate_tail_dependence(asset1, asset2, quantile)
                    results.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'lower_tail_dependence': lower,
                        'upper_tail_dependence': upper
                    })

        return pd.DataFrame(results)

    def identify_contagion_risk(self, threshold: float = 0.5) -> List[Tuple[str, str]]:
        """
        Identify asset pairs with high contagion risk (high tail dependence).

        Args:
            threshold: Threshold for high tail dependence

        Returns:
            List of asset pairs with high contagion risk
        """
        tail_deps = self.calculate_all_tail_dependencies()

        # Find pairs with high lower tail dependence (crisis contagion)
        high_risk = tail_deps[
            (tail_deps['lower_tail_dependence'] > threshold) |
            (tail_deps['upper_tail_dependence'] > threshold)
        ]

        return [(row['asset1'], row['asset2']) for _, row in high_risk.iterrows()]


def estimate_correlation_from_data(
    returns: pd.DataFrame,
    method: CorrelationMethod = CorrelationMethod.PEARSON,
    min_periods: int = 30
) -> pd.DataFrame:
    """
    Estimate correlation matrix from return data.

    Args:
        returns: DataFrame of asset returns
        method: Correlation method
        min_periods: Minimum number of observations required

    Returns:
        Correlation matrix
    """
    if method == CorrelationMethod.PEARSON:
        return returns.corr(method='pearson', min_periods=min_periods)
    elif method == CorrelationMethod.SPEARMAN:
        return returns.corr(method='spearman', min_periods=min_periods)
    elif method == CorrelationMethod.KENDALL:
        return returns.corr(method='kendall', min_periods=min_periods)
    else:
        raise ValueError(f"Unknown correlation method: {method}")


def shrink_correlation_matrix(
    sample_corr: pd.DataFrame,
    shrinkage_factor: float = 0.2
) -> pd.DataFrame:
    """
    Apply shrinkage to correlation matrix (Ledoit-Wolf shrinkage).

    Shrinks sample correlation towards identity matrix for better estimation.

    Args:
        sample_corr: Sample correlation matrix
        shrinkage_factor: Shrinkage intensity (0-1)

    Returns:
        Shrunk correlation matrix
    """
    n = len(sample_corr)
    target = np.eye(n)  # Identity matrix as target

    # Linear combination of sample and target
    shrunk = (1 - shrinkage_factor) * sample_corr + shrinkage_factor * target

    return pd.DataFrame(shrunk, index=sample_corr.index, columns=sample_corr.columns)


def ensure_positive_definite(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correlation matrix is positive definite.

    Args:
        corr_matrix: Correlation matrix

    Returns:
        Positive definite correlation matrix
    """
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Set negative eigenvalues to small positive value
    eigenvalues[eigenvalues < 0] = 1e-8

    # Reconstruct matrix
    corr_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Rescale to correlation matrix (diagonal = 1)
    d = np.sqrt(np.diag(corr_fixed))
    corr_fixed = corr_fixed / np.outer(d, d)

    return pd.DataFrame(corr_fixed, index=corr_matrix.index, columns=corr_matrix.columns)


def calculate_rolling_correlation_stability(
    returns: pd.DataFrame,
    window: int = 60,
    step: int = 20
) -> Dict[str, float]:
    """
    Calculate stability of correlations over time.

    Args:
        returns: DataFrame of asset returns
        window: Rolling window size
        step: Step size for windows

    Returns:
        Dictionary with stability metrics for each asset pair
    """
    stability_scores = {}
    assets = returns.columns.tolist()

    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            if i < j:
                correlations = []

                # Calculate correlations for non-overlapping windows
                for start in range(0, len(returns) - window, step):
                    window_data = returns.iloc[start:start + window]
                    corr = window_data[asset1].corr(window_data[asset2])
                    correlations.append(corr)

                # Stability = 1 - std of correlations
                if correlations:
                    stability = 1 - np.std(correlations)
                    stability_scores[f"{asset1}_{asset2}"] = max(0, stability)

    return stability_scores
