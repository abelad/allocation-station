"""Visualization utilities for portfolio analysis."""

from .charts import (
    plot_portfolio_performance,
    plot_efficient_frontier,
    plot_allocation_pie,
    plot_monte_carlo_paths,
    plot_drawdown,
    plot_returns_distribution,
)

# Try to import optional modules
try:
    from .dashboard import Dashboard, DashboardConfig
    _has_dashboard = True
except ImportError:
    _has_dashboard = False

# Build __all__ dynamically
__all__ = [
    "plot_portfolio_performance",
    "plot_efficient_frontier",
    "plot_allocation_pie",
    "plot_monte_carlo_paths",
    "plot_drawdown",
    "plot_returns_distribution",
]

if _has_dashboard:
    __all__.extend(["Dashboard", "DashboardConfig"])