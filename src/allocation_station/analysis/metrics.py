"""
Portfolio metrics calculation module.

Provides functions and classes for calculating common portfolio performance
and risk metrics including Sharpe ratio, Sortino ratio, maximum drawdown,
and other standard portfolio analytics.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import pandas as pd


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""

    volatility: float
    """Annualized volatility (standard deviation of returns)"""

    downside_deviation: float
    """Downside deviation for Sortino ratio calculation"""

    max_drawdown: float
    """Maximum peak-to-trough decline"""

    var_95: float
    """Value at Risk at 95% confidence level"""

    cvar_95: float
    """Conditional Value at Risk (Expected Shortfall) at 95%"""

    beta: Optional[float] = None
    """Beta relative to benchmark (if provided)"""

    tracking_error: Optional[float] = None
    """Tracking error relative to benchmark (if provided)"""


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""

    total_return: float
    """Cumulative total return"""

    annualized_return: float
    """Annualized return (CAGR)"""

    sharpe_ratio: float
    """Sharpe ratio (risk-adjusted return)"""

    sortino_ratio: float
    """Sortino ratio (downside risk-adjusted return)"""

    calmar_ratio: float
    """Calmar ratio (return / max drawdown)"""

    information_ratio: Optional[float] = None
    """Information ratio relative to benchmark (if provided)"""

    alpha: Optional[float] = None
    """Alpha relative to benchmark (if provided)"""

    treynor_ratio: Optional[float] = None
    """Treynor ratio (if beta available)"""


def calculate_returns(
    prices: Union[pd.Series, pd.DataFrame],
    method: str = 'simple'
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate returns from price series.

    Args:
        prices: Price series or DataFrame
        method: 'simple' or 'log' returns

    Returns:
        Returns series or DataFrame
    """
    if method == 'simple':
        return prices.pct_change().dropna()
    elif method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_volatility(
    returns: pd.Series,
    annualization_factor: int = 252
) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Return series
        annualization_factor: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized volatility
    """
    return float(returns.std() * np.sqrt(annualization_factor))


def calculate_downside_deviation(
    returns: pd.Series,
    target_return: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate downside deviation (semi-deviation).

    Args:
        returns: Return series
        target_return: Minimum acceptable return (MAR)
        annualization_factor: Number of periods per year

    Returns:
        Annualized downside deviation
    """
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return 0.0
    return float(downside_returns.std() * np.sqrt(annualization_factor))


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        prices: Price series

    Returns:
        Maximum drawdown as a positive number (e.g., 0.20 for 20% drawdown)
    """
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return float(abs(drawdown.min()))


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Return series
        confidence_level: Confidence level (default 0.95 for 95%)

    Returns:
        VaR as a positive number
    """
    return float(abs(returns.quantile(1 - confidence_level)))


def calculate_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Return series
        confidence_level: Confidence level (default 0.95 for 95%)

    Returns:
        CVaR as a positive number
    """
    var = calculate_var(returns, confidence_level)
    return float(abs(returns[returns <= -var].mean()))


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        annualization_factor: Number of periods per year

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / annualization_factor)
    if excess_returns.std() == 0:
        return 0.0
    return float(
        (excess_returns.mean() * annualization_factor) /
        (excess_returns.std() * np.sqrt(annualization_factor))
    )


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate Sortino ratio.

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        target_return: Minimum acceptable return
        annualization_factor: Number of periods per year

    Returns:
        Sortino ratio
    """
    excess_returns = returns - (risk_free_rate / annualization_factor)
    downside_dev = calculate_downside_deviation(returns, target_return, annualization_factor)

    if downside_dev == 0:
        return 0.0
    return float((excess_returns.mean() * annualization_factor) / downside_dev)


def calculate_calmar_ratio(
    returns: pd.Series,
    prices: pd.Series,
    annualization_factor: int = 252
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).

    Args:
        returns: Return series
        prices: Price series (for drawdown calculation)
        annualization_factor: Number of periods per year

    Returns:
        Calmar ratio
    """
    annual_return = returns.mean() * annualization_factor
    max_dd = calculate_max_drawdown(prices)

    if max_dd == 0:
        return 0.0
    return float(annual_return / max_dd)


def calculate_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate beta relative to benchmark.

    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series

    Returns:
        Beta coefficient
    """
    covariance = returns.cov(benchmark_returns)
    benchmark_variance = benchmark_returns.var()

    if benchmark_variance == 0:
        return 0.0
    return float(covariance / benchmark_variance)


def calculate_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate Jensen's alpha.

    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
        risk_free_rate: Annual risk-free rate
        annualization_factor: Number of periods per year

    Returns:
        Annualized alpha
    """
    beta = calculate_beta(returns, benchmark_returns)

    rf_per_period = risk_free_rate / annualization_factor
    portfolio_return = returns.mean()
    benchmark_return = benchmark_returns.mean()

    alpha_per_period = portfolio_return - (rf_per_period + beta * (benchmark_return - rf_per_period))
    return float(alpha_per_period * annualization_factor)


def calculate_tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization_factor: int = 252
) -> float:
    """
    Calculate tracking error.

    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
        annualization_factor: Number of periods per year

    Returns:
        Annualized tracking error
    """
    active_returns = returns - benchmark_returns
    return float(active_returns.std() * np.sqrt(annualization_factor))


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization_factor: int = 252
) -> float:
    """
    Calculate information ratio.

    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
        annualization_factor: Number of periods per year

    Returns:
        Information ratio
    """
    active_returns = returns - benchmark_returns
    tracking_error = calculate_tracking_error(returns, benchmark_returns, annualization_factor)

    if tracking_error == 0:
        return 0.0
    return float((active_returns.mean() * annualization_factor) / tracking_error)


def calculate_treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate Treynor ratio.

    Args:
        returns: Portfolio return series
        benchmark_returns: Benchmark return series
        risk_free_rate: Annual risk-free rate
        annualization_factor: Number of periods per year

    Returns:
        Treynor ratio
    """
    beta = calculate_beta(returns, benchmark_returns)

    if beta == 0:
        return 0.0

    excess_return = (returns.mean() * annualization_factor) - risk_free_rate
    return float(excess_return / beta)


def calculate_portfolio_metrics(
    returns: pd.Series,
    prices: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
    annualization_factor: int = 252
) -> tuple[RiskMetrics, PerformanceMetrics]:
    """
    Calculate comprehensive portfolio metrics.

    Args:
        returns: Portfolio return series
        prices: Price series (if None, reconstructed from returns)
        benchmark_returns: Optional benchmark return series
        risk_free_rate: Annual risk-free rate (default 2%)
        annualization_factor: Periods per year (252 for daily, 12 for monthly)

    Returns:
        Tuple of (RiskMetrics, PerformanceMetrics)
    """
    # Reconstruct prices if not provided
    if prices is None:
        prices = (1 + returns).cumprod()

    # Calculate risk metrics
    volatility = calculate_volatility(returns, annualization_factor)
    downside_dev = calculate_downside_deviation(returns, 0.0, annualization_factor)
    max_dd = calculate_max_drawdown(prices)
    var_95 = calculate_var(returns, 0.95)
    cvar_95 = calculate_cvar(returns, 0.95)

    # Calculate benchmark-relative metrics if benchmark provided
    beta = None
    tracking_error = None
    if benchmark_returns is not None:
        beta = calculate_beta(returns, benchmark_returns)
        tracking_error = calculate_tracking_error(returns, benchmark_returns, annualization_factor)

    risk_metrics = RiskMetrics(
        volatility=volatility,
        downside_deviation=downside_dev,
        max_drawdown=max_dd,
        var_95=var_95,
        cvar_95=cvar_95,
        beta=beta,
        tracking_error=tracking_error
    )

    # Calculate performance metrics
    total_return = float((prices.iloc[-1] / prices.iloc[0]) - 1)
    n_periods = len(returns)
    years = n_periods / annualization_factor
    annualized_return = float(((1 + total_return) ** (1 / years)) - 1) if years > 0 else 0.0

    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, annualization_factor)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, 0.0, annualization_factor)
    calmar = calculate_calmar_ratio(returns, prices, annualization_factor)

    # Calculate benchmark-relative performance metrics
    alpha = None
    info_ratio = None
    treynor = None
    if benchmark_returns is not None:
        alpha = calculate_alpha(returns, benchmark_returns, risk_free_rate, annualization_factor)
        info_ratio = calculate_information_ratio(returns, benchmark_returns, annualization_factor)
        treynor = calculate_treynor_ratio(returns, benchmark_returns, risk_free_rate, annualization_factor)

    performance_metrics = PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        information_ratio=info_ratio,
        alpha=alpha,
        treynor_ratio=treynor
    )

    return risk_metrics, performance_metrics


__all__ = [
    'RiskMetrics',
    'PerformanceMetrics',
    'calculate_portfolio_metrics',
    'calculate_returns',
    'calculate_volatility',
    'calculate_downside_deviation',
    'calculate_max_drawdown',
    'calculate_var',
    'calculate_cvar',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_beta',
    'calculate_alpha',
    'calculate_tracking_error',
    'calculate_information_ratio',
    'calculate_treynor_ratio',
]
