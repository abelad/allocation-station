"""Portfolio analysis and metrics modules."""

# Import only modules that exist
from .efficient_frontier import EfficientFrontier, OptimizationObjective

# Try to import optional modules
try:
    from .metrics import calculate_portfolio_metrics, RiskMetrics, PerformanceMetrics  # type: ignore[import-not-found]
    _has_metrics = True
except ImportError:
    _has_metrics = False

try:
    from .comparison import StrategyComparison, ComparisonReport  # type: ignore[import-not-found]
    _has_comparison = True
except ImportError:
    _has_comparison = False

# Build __all__ dynamically
__all__ = [
    "EfficientFrontier",
    "OptimizationObjective",
]

if _has_metrics:
    __all__.extend(["calculate_portfolio_metrics", "RiskMetrics", "PerformanceMetrics"])

if _has_comparison:
    __all__.extend(["StrategyComparison", "ComparisonReport"])