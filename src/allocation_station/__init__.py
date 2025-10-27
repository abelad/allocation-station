"""
Allocation Station: Asset Allocation Strategy Testing Framework

A comprehensive framework for testing and analyzing asset allocation strategies
across various market conditions using Monte Carlo simulations, backtesting,
and modern portfolio theory.
"""

__version__ = "0.1.0"

from .core.asset import Asset, AssetClass
from .core.portfolio import Portfolio
from .portfolio.strategy import AllocationStrategy
from .simulation.monte_carlo import MonteCarloSimulator
from .backtesting.engine import BacktestEngine

__all__ = [
    "Asset",
    "AssetClass",
    "Portfolio",
    "AllocationStrategy",
    "MonteCarloSimulator",
    "BacktestEngine",
]