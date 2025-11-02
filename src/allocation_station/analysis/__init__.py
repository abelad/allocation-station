"""Portfolio analysis and metrics modules."""

# Import only modules that exist
from .efficient_frontier import EfficientFrontier, OptimizationObjective

# Import metrics module
from .metrics import calculate_portfolio_metrics, RiskMetrics, PerformanceMetrics

try:
    from .comparison import StrategyComparison, ComparisonReport  # type: ignore[import-not-found]
    _has_comparison = True
except ImportError:
    _has_comparison = False

# Build __all__ dynamically
__all__ = [
    "EfficientFrontier",
    "OptimizationObjective",
    "calculate_portfolio_metrics",
    "RiskMetrics",
    "PerformanceMetrics",
]

if _has_comparison:
    __all__.extend(["StrategyComparison", "ComparisonReport"])