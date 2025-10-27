"""Plugin system for custom data sources."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Type
from datetime import datetime
import pandas as pd
from pathlib import Path
import importlib.util
import inspect
from pydantic import BaseModel, Field


class PluginMetadata(BaseModel):
    """Metadata for a data source plugin."""
    name: str
    version: str
    author: str
    description: str
    data_source_type: str
    requires_api_key: bool = False
    supported_frequencies: List[str] = Field(default_factory=list)
    supported_asset_types: List[str] = Field(default_factory=list)


class DataSourcePlugin(ABC):
    """
    Abstract base class for data source plugins.

    Custom data sources should inherit from this class and implement
    all required methods.
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """
        Initialize the plugin with configuration.

        Args:
            config: Plugin configuration dictionary
        """
        pass

    @abstractmethod
    def fetch_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "daily",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical market data.

        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency
            **kwargs: Additional parameters

        Returns:
            DataFrame with historical data
        """
        pass

    @abstractmethod
    def fetch_real_time_data(
        self,
        symbols: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Fetch real-time market data.

        Args:
            symbols: Single symbol or list of symbols
            **kwargs: Additional parameters

        Returns:
            Dictionary with real-time data for each symbol
        """
        pass

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported by this data source.

        Args:
            symbol: Symbol to validate

        Returns:
            True if symbol is valid
        """
        pass

    def test_connection(self) -> bool:
        """
        Test connection to data source.

        Returns:
            True if connection is successful
        """
        try:
            # Default implementation - try to fetch a test symbol
            test_data = self.fetch_real_time_data(["SPY"])
            return bool(test_data)
        except:
            return False

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from data source.

        Returns:
            List of available symbols
        """
        # Default implementation - return empty list
        # Plugins can override this if they support symbol lookup
        return []

    def cleanup(self):
        """
        Cleanup resources when plugin is unloaded.

        Override this method if your plugin needs cleanup.
        """
        pass


class PluginManager:
    """
    Manages data source plugins.

    Handles plugin discovery, loading, and lifecycle management.
    """

    def __init__(self, plugin_dir: str = "data/plugins"):
        """
        Initialize plugin manager.

        Args:
            plugin_dir: Directory containing plugin files
        """
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(parents=True, exist_ok=True)

        self.plugins: Dict[str, DataSourcePlugin] = {}
        self.plugin_classes: Dict[str, Type[DataSourcePlugin]] = {}

    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in plugin directory.

        Returns:
            List of discovered plugin names
        """
        discovered = []

        for file in self.plugin_dir.glob("*.py"):
            if file.stem.startswith("_"):
                continue

            try:
                # Load module
                spec = importlib.util.spec_from_file_location(file.stem, file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find DataSourcePlugin subclasses
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, DataSourcePlugin) and
                            obj is not DataSourcePlugin):
                            self.plugin_classes[name] = obj
                            discovered.append(name)

            except Exception as e:
                print(f"Error loading plugin from {file}: {e}")
                continue

        return discovered

    def load_plugin(
        self,
        plugin_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> DataSourcePlugin:
        """
        Load and initialize a plugin.

        Args:
            plugin_name: Name of plugin to load
            config: Plugin configuration

        Returns:
            Initialized plugin instance
        """
        if plugin_name in self.plugins:
            return self.plugins[plugin_name]

        if plugin_name not in self.plugin_classes:
            raise ValueError(f"Plugin not found: {plugin_name}")

        # Create instance
        plugin_class = self.plugin_classes[plugin_name]
        plugin = plugin_class()

        # Initialize
        if config:
            plugin.initialize(config)

        # Store
        self.plugins[plugin_name] = plugin

        return plugin

    def unload_plugin(self, plugin_name: str):
        """
        Unload a plugin.

        Args:
            plugin_name: Name of plugin to unload
        """
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            plugin.cleanup()
            del self.plugins[plugin_name]

    def get_plugin(self, plugin_name: str) -> Optional[DataSourcePlugin]:
        """
        Get a loaded plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin instance or None
        """
        return self.plugins.get(plugin_name)

    def get_all_plugins(self) -> Dict[str, DataSourcePlugin]:
        """
        Get all loaded plugins.

        Returns:
            Dictionary of plugin name to plugin instance
        """
        return self.plugins.copy()

    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """
        Get metadata for a plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin metadata or None
        """
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.metadata
        return None

    def list_available_plugins(self) -> List[Dict[str, Any]]:
        """
        List all available plugins with their metadata.

        Returns:
            List of plugin information dictionaries
        """
        plugins_info = []

        for name, plugin_class in self.plugin_classes.items():
            try:
                # Create temporary instance to get metadata
                temp_plugin = plugin_class()
                metadata = temp_plugin.metadata

                plugins_info.append({
                    'name': name,
                    'loaded': name in self.plugins,
                    'metadata': metadata.dict()
                })
            except:
                plugins_info.append({
                    'name': name,
                    'loaded': name in self.plugins,
                    'metadata': None
                })

        return plugins_info

    def reload_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Reload a plugin.

        Args:
            plugin_name: Plugin name
            config: New configuration
        """
        if plugin_name in self.plugins:
            self.unload_plugin(plugin_name)

        self.load_plugin(plugin_name, config)

    def test_all_plugins(self) -> Dict[str, bool]:
        """
        Test connections for all loaded plugins.

        Returns:
            Dictionary mapping plugin name to test result
        """
        results = {}

        for name, plugin in self.plugins.items():
            try:
                results[name] = plugin.test_connection()
            except Exception as e:
                print(f"Error testing plugin {name}: {e}")
                results[name] = False

        return results


class CSVDataSourcePlugin(DataSourcePlugin):
    """
    Example plugin for CSV-based data sources.

    Demonstrates how to create a custom data source plugin.
    """

    def __init__(self):
        """Initialize CSV data source plugin."""
        self.data_dir: Optional[Path] = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="CSV Data Source",
            version="1.0.0",
            author="Allocation Station",
            description="Load market data from CSV files",
            data_source_type="file",
            requires_api_key=False,
            supported_frequencies=["daily", "weekly", "monthly"],
            supported_asset_types=["equity", "etf", "bond"]
        )

    def initialize(self, config: Dict[str, Any]):
        """
        Initialize plugin with configuration.

        Args:
            config: Configuration dictionary with 'data_dir' key
        """
        data_dir = config.get('data_dir', 'data/csv')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "daily",
        **kwargs
    ) -> pd.DataFrame:
        """Fetch historical data from CSV files."""
        if self.data_dir is None:
            raise RuntimeError("Plugin not initialized")

        if isinstance(symbols, str):
            symbols = [symbols]

        all_data = []

        for symbol in symbols:
            csv_file = self.data_dir / f"{symbol}.csv"

            if not csv_file.exists():
                print(f"CSV file not found for {symbol}")
                continue

            try:
                df = pd.read_csv(csv_file, parse_dates=['date'])

                # Filter by date range
                if start_date:
                    df = df[df['date'] >= start_date]
                if end_date:
                    df = df[df['date'] <= end_date]

                # Add symbol column if not present
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol

                all_data.append(df)

            except Exception as e:
                print(f"Error reading CSV for {symbol}: {e}")
                continue

        if all_data:
            return pd.concat(all_data, ignore_index=True)

        return pd.DataFrame()

    def fetch_real_time_data(
        self,
        symbols: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Fetch latest data from CSV files."""
        if self.data_dir is None:
            raise RuntimeError("Plugin not initialized")

        if isinstance(symbols, str):
            symbols = [symbols]

        real_time_data = {}

        for symbol in symbols:
            csv_file = self.data_dir / f"{symbol}.csv"

            if not csv_file.exists():
                real_time_data[symbol] = {}
                continue

            try:
                df = pd.read_csv(csv_file, parse_dates=['date'])
                latest = df.iloc[-1]

                real_time_data[symbol] = {
                    'price': latest.get('close', 0),
                    'date': str(latest.get('date', '')),
                    'volume': latest.get('volume', 0)
                }

            except Exception as e:
                print(f"Error reading CSV for {symbol}: {e}")
                real_time_data[symbol] = {}

        return real_time_data

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if CSV file exists for symbol."""
        if self.data_dir is None:
            return False

        csv_file = self.data_dir / f"{symbol}.csv"
        return csv_file.exists()

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with CSV files."""
        if self.data_dir is None:
            return []

        symbols = []
        for file in self.data_dir.glob("*.csv"):
            symbols.append(file.stem)

        return sorted(symbols)


class DatabaseDataSourcePlugin(DataSourcePlugin):
    """
    Example plugin for database-based data sources.

    Demonstrates database integration for custom data sources.
    """

    def __init__(self):
        """Initialize database data source plugin."""
        self.connection = None
        self.table_name = "market_data"

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="Database Data Source",
            version="1.0.0",
            author="Allocation Station",
            description="Load market data from database",
            data_source_type="database",
            requires_api_key=False,
            supported_frequencies=["daily", "intraday"],
            supported_asset_types=["equity", "etf", "bond", "crypto"]
        )

    def initialize(self, config: Dict[str, Any]):
        """
        Initialize plugin with database configuration.

        Args:
            config: Configuration with 'connection_string' and optional 'table_name'
        """
        # Note: This is a template - actual implementation would use
        # sqlalchemy or another database library
        self.connection_string = config.get('connection_string')
        self.table_name = config.get('table_name', 'market_data')

        # In real implementation, create database connection here
        # self.connection = create_engine(self.connection_string)

    def fetch_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "daily",
        **kwargs
    ) -> pd.DataFrame:
        """Fetch historical data from database."""
        if self.connection is None:
            raise RuntimeError("Plugin not initialized or database not connected")

        # Template query - actual implementation would execute this
        # query = f"""
        #     SELECT * FROM {self.table_name}
        #     WHERE symbol IN ({','.join(['?']*len(symbols))})
        #     AND date BETWEEN ? AND ?
        # """
        # return pd.read_sql(query, self.connection, params=...)

        # Placeholder
        return pd.DataFrame()

    def fetch_real_time_data(
        self,
        symbols: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Fetch latest data from database."""
        if self.connection is None:
            raise RuntimeError("Plugin not initialized or database not connected")

        # Placeholder
        return {}

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists in database."""
        if self.connection is None:
            return False

        # Placeholder - would query database
        return True

    def cleanup(self):
        """Close database connection."""
        if self.connection:
            # self.connection.close()
            pass
