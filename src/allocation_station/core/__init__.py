"""Core data models and structures for the allocation framework."""

from .asset import Asset, AssetClass
from .portfolio import Portfolio
from .allocation import Allocation

# Import enhanced asset classes
try:
    from .enhanced_assets import (
        OptionAsset, OptionType, OptionStyle,
        REITAsset, REITType,
        CryptoAsset, CryptoType,
        CommodityAsset, CommodityType,
        FutureContract, FutureType,
        AlternativeAsset, AlternativeType,
        StructuredProduct, StructuredProductType
    )
    _ENHANCED_ASSETS_AVAILABLE = True
except ImportError:
    _ENHANCED_ASSETS_AVAILABLE = False

# Import correlation models
try:
    from .correlation_models import (
        AssetClassCorrelationModel,
        DynamicCorrelationModel,
        HierarchicalCorrelationModel,
        CopulaCorrelationModel,
        CorrelationMethod,
        CorrelationRegime
    )
    _CORRELATION_MODELS_AVAILABLE = True
except ImportError:
    _CORRELATION_MODELS_AVAILABLE = False

# Build exports
__all__ = ["Asset", "AssetClass", "Portfolio", "Allocation"]

if _ENHANCED_ASSETS_AVAILABLE:
    __all__.extend([
        "OptionAsset", "OptionType", "OptionStyle",
        "REITAsset", "REITType",
        "CryptoAsset", "CryptoType",
        "CommodityAsset", "CommodityType",
        "FutureContract", "FutureType",
        "AlternativeAsset", "AlternativeType",
        "StructuredProduct", "StructuredProductType"
    ])

if _CORRELATION_MODELS_AVAILABLE:
    __all__.extend([
        "AssetClassCorrelationModel",
        "DynamicCorrelationModel",
        "HierarchicalCorrelationModel",
        "CopulaCorrelationModel",
        "CorrelationMethod",
        "CorrelationRegime"
    ])