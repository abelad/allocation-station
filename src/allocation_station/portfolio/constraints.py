"""Portfolio constraints and optimization boundaries."""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from ..core import AssetClass


class PortfolioConstraints(BaseModel):
    """
    Defines constraints for portfolio optimization and allocation.
    """

    # Weight constraints
    min_weights: Dict[str, float] = Field(default_factory=dict, description="Minimum weights by asset")
    max_weights: Dict[str, float] = Field(default_factory=dict, description="Maximum weights by asset")

    # Asset class constraints
    asset_class_min: Dict[AssetClass, float] = Field(default_factory=dict, description="Min by asset class")
    asset_class_max: Dict[AssetClass, float] = Field(default_factory=dict, description="Max by asset class")

    # Sector constraints
    sector_limits: Dict[str, Tuple[float, float]] = Field(default_factory=dict, description="Sector min/max")
    max_sector_concentration: float = Field(0.4, description="Maximum sector concentration")

    # Risk constraints
    max_volatility: Optional[float] = Field(None, description="Maximum portfolio volatility")
    max_var: Optional[float] = Field(None, description="Maximum Value at Risk")
    max_drawdown: Optional[float] = Field(None, description="Maximum acceptable drawdown")
    min_sharpe: Optional[float] = Field(None, description="Minimum Sharpe ratio")

    # Diversification constraints
    min_assets: int = Field(3, description="Minimum number of assets")
    max_assets: Optional[int] = Field(None, description="Maximum number of assets")
    max_correlation: float = Field(0.9, description="Maximum correlation between assets")

    # Liquidity constraints
    min_liquidity: float = Field(0.1, description="Minimum liquid allocation")
    min_daily_volume: Optional[float] = Field(None, description="Minimum daily trading volume")

    # ESG constraints
    esg_minimum_score: Optional[float] = Field(None, description="Minimum ESG score")
    excluded_sectors: List[str] = Field(default_factory=list, description="Excluded sectors")

    @validator("max_weights")
    def validate_max_weights(cls, v, values):
        """Ensure max weights are greater than min weights."""
        min_weights = values.get('min_weights', {})
        for asset, max_weight in v.items():
            if asset in min_weights and max_weight < min_weights[asset]:
                raise ValueError(f"Max weight for {asset} must be >= min weight")
        return v

    def check_allocation(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if allocation satisfies all constraints.

        Args:
            allocation: Proposed allocation weights

        Returns:
            Dictionary with constraint violations
        """
        violations = {}

        # Check individual weight constraints
        for symbol, weight in allocation.items():
            if symbol in self.min_weights and weight < self.min_weights[symbol]:
                violations[f"{symbol}_min"] = f"Weight {weight} < min {self.min_weights[symbol]}"
            if symbol in self.max_weights and weight > self.max_weights[symbol]:
                violations[f"{symbol}_max"] = f"Weight {weight} > max {self.max_weights[symbol]}"

        # Check number of assets
        n_assets = len([w for w in allocation.values() if w > 0])
        if n_assets < self.min_assets:
            violations['min_assets'] = f"Only {n_assets} assets, minimum is {self.min_assets}"
        if self.max_assets and n_assets > self.max_assets:
            violations['max_assets'] = f"{n_assets} assets exceeds maximum {self.max_assets}"

        return violations

    def apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply constraints to portfolio weights.

        Args:
            weights: Proposed weights

        Returns:
            Constrained weights
        """
        constrained = weights.copy()

        # Apply min/max constraints
        for symbol, weight in weights.items():
            if symbol in self.min_weights:
                constrained[symbol] = max(weight, self.min_weights[symbol])
            if symbol in self.max_weights:
                constrained[symbol] = min(weight, self.max_weights[symbol])

        # Renormalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v/total for k, v in constrained.items()}

        return constrained