# Data Infrastructure Documentation

This document describes the enhanced data infrastructure features implemented in Allocation Station.

## Overview

The data infrastructure has been significantly enhanced with:

1. **Multiple Data Sources**: Support for Alpha Vantage, NASDAQ Data Link (Quandl), FRED, and Yahoo Finance
2. **Data Validation & Cleaning**: Comprehensive pipelines for data quality assurance
3. **Data Export**: Multiple format support (CSV, Excel, Parquet, JSON, HDF5, Feather)
4. **Automated Scheduling**: Background jobs for automatic data updates
5. **Plugin System**: Extensible architecture for custom data sources

## Data Sources

### Yahoo Finance (Default)
```python
from allocation_station.data import YahooFinanceSource

source = YahooFinanceSource()
data = source.fetch_historical_data(
    symbols=['AAPL', 'MSFT'],
    start_date=datetime(2024, 1, 1),
    end_date=datetime.now()
)
```

**Features:**
- No API key required
- Historical and real-time data
- Options chains and fundamental data
- Intraday data support

### Alpha Vantage
```python
from allocation_station.data import AlphaVantageSource

source = AlphaVantageSource(api_key='YOUR_API_KEY')
data = source.fetch_historical_data(
    symbols=['AAPL'],
    frequency='daily'
)
```

**Features:**
- Comprehensive fundamental data
- Technical indicators
- Adjusted price data
- Global market coverage
- API Key: https://www.alphavantage.co/support/#api-key

### NASDAQ Data Link (Quandl)
```python
from allocation_station.data import NasdaqDataLinkSource

source = NasdaqDataLinkSource(api_key='YOUR_API_KEY')
data = source.fetch_historical_data(
    symbols=['WIKI/AAPL'],
    start_date=datetime(2023, 1, 1)
)
```

**Features:**
- Premium datasets
- Alternative data sources
- Economic and financial data
- API Key: https://data.nasdaq.com/

### FRED (Federal Reserve Economic Data)
```python
from allocation_station.data import FREDSource

source = FREDSource(api_key='YOUR_FRED_API_KEY')
data = source.fetch_historical_data(
    symbols=['GDP', 'UNRATE', 'DGS10'],
    start_date=datetime(2020, 1, 1)
)
```

**Features:**
- 816,000+ economic time series
- Federal Reserve data
- Macroeconomic indicators
- Interest rates, GDP, inflation, etc.
- API Key: https://fred.stlouisfed.org/docs/api/api_key.html

## Data Validation

### Validation Checks

The `DataValidator` performs comprehensive checks:

- **Missing Values**: Detects and reports missing data
- **Duplicates**: Identifies duplicate date-symbol combinations
- **Data Types**: Verifies correct column types
- **Price Anomalies**: Checks OHLC relationships
- **Volume Anomalies**: Detects unusual volume patterns
- **Date Gaps**: Identifies missing business days
- **Negative Prices**: Flags invalid negative values
- **Outliers**: Statistical outlier detection using IQR
- **Constant Values**: Identifies columns with no variance
- **Correlation Breaks**: Detects unusual correlation patterns

### Usage

```python
from allocation_station.data import DataValidator

validator = DataValidator(strict_mode=False)
report = validator.validate(data)

print(f"Validation passed: {report.passed}")
print(f"Issues found: {len(report.issues)}")

# Show issues by severity
for issue in report.issues:
    print(f"[{issue.severity}] {issue.issue_type}: {issue.description}")
```

## Data Cleaning

### Cleaning Methods

The `DataCleaner` provides automated cleaning:

- **Remove Duplicates**: Eliminate duplicate rows
- **Fill Missing Values**: Forward/backward fill for prices
- **Fix Data Types**: Convert columns to appropriate types
- **Remove Negative Prices**: Filter invalid price data
- **Handle Outliers**: Winsorize or remove statistical outliers
- **Sort by Date**: Ensure chronological ordering
- **Interpolate**: Fill gaps using interpolation methods
- **Resample**: Change data frequency (daily → weekly, etc.)

### Usage

```python
from allocation_station.data import DataCleaner

cleaner = DataCleaner()
cleaned_data, log = cleaner.clean(data)

# Review cleaning actions
for action in log['actions']:
    print(f"{action['method']}: {action}")
```

## Data Export

### Supported Formats

- **CSV**: Comma-separated values (with optional compression)
- **Excel**: Multi-sheet workbooks with formatting
- **Parquet**: Columnar storage (efficient for large datasets)
- **JSON**: Human-readable format
- **HDF5**: Hierarchical data format
- **Feather**: Fast binary format

### Usage

```python
from allocation_station.data import DataExporter, ExportFormat

exporter = DataExporter(output_dir="data/exports")

# Export to different formats
exporter.export_to_csv(data, "market_data")
exporter.export_to_excel(data, "market_data")
exporter.export_to_parquet(data, "market_data", compression='snappy')

# Export with metadata
metadata = {
    "symbols": ["AAPL", "MSFT"],
    "date_range": "2024-01-01 to 2024-10-27",
    "source": "Yahoo Finance"
}
exporter.export_with_metadata(data, "market_data", metadata)

# Export portfolio with summary
exporter.export_portfolio_data(
    data,
    "portfolio_report",
    format=ExportFormat.EXCEL,
    include_summary=True
)
```

### Data Import

```python
from allocation_station.data import DataImporter

importer = DataImporter(input_dir="data/exports")

# Auto-detect format
data = importer.import_data("market_data.parquet")

# Import with metadata
data, metadata = importer.import_with_metadata("market_data")
```

## Automated Scheduling

### Scheduler Features

- **Flexible Scheduling**: Minutely, hourly, daily, weekly, monthly
- **Market-Based**: Schedule at market open/close
- **Job Management**: Enable, disable, pause, resume jobs
- **Error Tracking**: Monitor success/failure rates
- **Logging**: Comprehensive activity logs
- **Export/Import**: Save and restore schedule configurations

### Usage

```python
from allocation_station.data import DataScheduler, ScheduleFrequency
from allocation_station.data import MarketDataProvider, MarketDataConfig
from datetime import time

# Setup
config = MarketDataConfig(cache_dir="data/cache")
data_provider = MarketDataProvider(config)
scheduler = DataScheduler(data_provider, cache_dir="data/cache")

# Add daily update job
job = scheduler.add_update_job(
    job_id="daily_update",
    name="Daily Market Data Update",
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    frequency=ScheduleFrequency.DAILY,
    time_of_day=time(9, 30)  # 9:30 AM
)

# Add market close refresh
job = scheduler.add_market_data_refresh_job(
    symbols=['SPY', 'QQQ'],
    frequency=ScheduleFrequency.MARKET_CLOSE
)

# Add cache cleanup
scheduler.add_cache_cleanup_job(
    max_age_days=30,
    frequency=ScheduleFrequency.WEEKLY
)

# Start scheduler
scheduler.start()

# Monitor jobs
stats = scheduler.get_statistics()
print(f"Total jobs: {stats['total_jobs']}")
print(f"Successful runs: {stats['total_successful_runs']}")

# Stop when done
scheduler.stop()
```

## Plugin System

### Creating Custom Data Sources

Create a custom data source by inheriting from `DataSourcePlugin`:

```python
from allocation_station.data import DataSourcePlugin, PluginMetadata
from typing import Dict, List, Union, Optional
from datetime import datetime
import pandas as pd

class MyCustomDataSource(DataSourcePlugin):

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="My Custom Data Source",
            version="1.0.0",
            author="Your Name",
            description="Custom data source implementation",
            data_source_type="custom",
            requires_api_key=True,
            supported_frequencies=["daily", "weekly"],
            supported_asset_types=["equity", "etf"]
        )

    def initialize(self, config: Dict[str, Any]):
        self.api_key = config.get('api_key')
        # Initialize your data source

    def fetch_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "daily",
        **kwargs
    ) -> pd.DataFrame:
        # Implement data fetching logic
        pass

    def fetch_real_time_data(
        self,
        symbols: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        # Implement real-time data logic
        pass

    def validate_symbol(self, symbol: str) -> bool:
        # Validate symbol
        pass
```

### Using Plugins

```python
from allocation_station.data import PluginManager

# Create manager
plugin_manager = PluginManager(plugin_dir="data/plugins")

# Discover plugins
plugins = plugin_manager.discover_plugins()
print(f"Found plugins: {plugins}")

# Load plugin
plugin = plugin_manager.load_plugin(
    "MyCustomDataSource",
    config={'api_key': 'YOUR_KEY'}
)

# Use plugin
data = plugin.fetch_historical_data(
    symbols=['AAPL'],
    start_date=datetime(2024, 1, 1)
)

# Test connection
if plugin.test_connection():
    print("Plugin connected successfully")

# Unload when done
plugin_manager.unload_plugin("MyCustomDataSource")
```

## Intraday Data Support

All data sources support intraday frequencies:

```python
# Yahoo Finance intraday
data = yahoo_source.fetch_historical_data(
    symbols=['AAPL'],
    frequency='intraday_5m',  # 5-minute bars
    start_date=datetime.now() - timedelta(days=7)
)

# Alpha Vantage intraday
data = av_source.fetch_historical_data(
    symbols=['AAPL'],
    frequency='intraday'
)
```

**Supported Frequencies:**
- `intraday_1m`: 1-minute bars
- `intraday_5m`: 5-minute bars
- `intraday_15m`: 15-minute bars
- `intraday_30m`: 30-minute bars
- `hourly`: Hourly bars
- `daily`: Daily bars (default)
- `weekly`: Weekly bars
- `monthly`: Monthly bars

## Best Practices

### 1. Data Validation

Always validate data after fetching:

```python
# Fetch → Validate → Clean → Use
data = source.fetch_historical_data(symbols)
report = validator.validate(data)

if not report.passed:
    print("Data quality issues detected")
    cleaned_data, log = cleaner.clean(data)
    data = cleaned_data
```

### 2. Caching

Use caching to minimize API calls:

```python
from allocation_station.data import MarketDataConfig, MarketDataProvider

config = MarketDataConfig(
    cache_dir="data/cache",
    cache_expiry=3600  # 1 hour
)
provider = MarketDataProvider(config)
```

### 3. Error Handling

Wrap data fetching in try-except blocks:

```python
try:
    data = source.fetch_historical_data(symbols)
except Exception as e:
    print(f"Error fetching data: {e}")
    # Fallback to cached data or alternative source
```

### 4. Scheduled Updates

Use scheduler for production systems:

```python
# Setup automated daily updates
scheduler.add_portfolio_update_job(
    portfolio_name="production_portfolio",
    symbols=portfolio_symbols,
    frequency=ScheduleFrequency.DAILY,
    time_of_day=time(16, 30)  # After market close
)
scheduler.start()
```

### 5. Data Export

Export important data for backup and analysis:

```python
# Regular backups
exporter.export_to_parquet(data, f"backup_{datetime.now():%Y%m%d}")

# Analysis-ready formats
exporter.export_to_excel(
    data,
    "portfolio_analysis",
    include_summary=True
)
```

## Performance Considerations

### Large Datasets

For large datasets, use Parquet format:

```python
# Most efficient for large data
exporter.export_to_parquet(
    data,
    "large_dataset",
    compression='snappy'  # or 'gzip' for more compression
)
```

### Memory Efficiency

Process data in chunks for very large datasets:

```python
# Process symbol by symbol
for symbol in symbols:
    data = source.fetch_historical_data([symbol])
    process_data(data)
    exporter.export_to_parquet(data, f"data_{symbol}")
```

### API Rate Limits

Respect API rate limits:

```python
import time

for symbol in symbols:
    data = source.fetch_historical_data([symbol])
    time.sleep(0.1)  # 10 requests per second
```

## Troubleshooting

### Missing API Key Errors

```python
# Check if data source is available
if hasattr(source, 'is_available'):
    if not source.is_available():
        print("Data source not available - check API key")
```

### Validation Failures

```python
# Review specific issues
for issue in report.issues:
    if issue.severity == ValidationSeverity.CRITICAL:
        print(f"Critical issue: {issue.description}")
        # Take corrective action
```

### Scheduler Issues

```python
# Check scheduler status
stats = scheduler.get_statistics()
if stats['total_errors'] > 0:
    print("Scheduler has errors")

    # Review job status
    for job in scheduler.get_all_jobs():
        if job.error_count > 0:
            print(f"Job {job.name} errors: {job.last_error}")
```

## Examples

See [examples/data_sources_example.py](../examples/data_sources_example.py) for comprehensive examples demonstrating all features.

## API Reference

For detailed API documentation, see:
- [sources.py](../src/allocation_station/data/sources.py) - Data source implementations
- [validation.py](../src/allocation_station/data/validation.py) - Validation and cleaning
- [export.py](../src/allocation_station/data/export.py) - Export and import functionality
- [scheduler.py](../src/allocation_station/data/scheduler.py) - Automated scheduling
- [plugins.py](../src/allocation_station/data/plugins.py) - Plugin system
