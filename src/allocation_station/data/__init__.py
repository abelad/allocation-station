"""Data fetching and management modules."""

from .market_data import MarketDataProvider, DataSource

# Try to import new modules
try:
    from .sources import (
        AlphaVantageSource,
        NasdaqDataLinkSource,
        FREDSource,
        YahooFinanceSource,
        DataSourceInterface
    )
    _SOURCES_AVAILABLE = True
except ImportError:
    _SOURCES_AVAILABLE = False

try:
    from .validation import DataValidator, DataCleaner, ValidationReport
    _VALIDATION_AVAILABLE = True
except ImportError:
    _VALIDATION_AVAILABLE = False

try:
    from .export import DataExporter, DataImporter, ExportFormat
    _EXPORT_AVAILABLE = True
except ImportError:
    _EXPORT_AVAILABLE = False

try:
    from .scheduler import DataScheduler, ScheduleFrequency
    _SCHEDULER_AVAILABLE = True
except ImportError:
    _SCHEDULER_AVAILABLE = False

try:
    from .plugins import PluginManager, DataSourcePlugin, PluginMetadata
    _PLUGINS_AVAILABLE = True
except ImportError:
    _PLUGINS_AVAILABLE = False

# Build __all__ list dynamically
__all__ = ["MarketDataProvider", "DataSource"]

if _SOURCES_AVAILABLE:
    __all__.extend([
        "AlphaVantageSource",
        "NasdaqDataLinkSource",
        "FREDSource",
        "YahooFinanceSource",
        "DataSourceInterface"
    ])

if _VALIDATION_AVAILABLE:
    __all__.extend(["DataValidator", "DataCleaner", "ValidationReport"])

if _EXPORT_AVAILABLE:
    __all__.extend(["DataExporter", "DataImporter", "ExportFormat"])

if _SCHEDULER_AVAILABLE:
    __all__.extend(["DataScheduler", "ScheduleFrequency"])

if _PLUGINS_AVAILABLE:
    __all__.extend(["PluginManager", "DataSourcePlugin", "PluginMetadata"])