"""
Example demonstrating the enhanced data infrastructure features.

This example shows how to:
1. Use different data sources (Alpha Vantage, NASDAQ Data Link, FRED, Yahoo Finance)
2. Validate and clean market data
3. Export data to various formats
4. Schedule automated data updates
5. Create custom data source plugins
"""

from datetime import datetime, timedelta, time
from allocation_station.data import (
    MarketDataProvider,
    AlphaVantageSource,
    NasdaqDataLinkSource,
    FREDSource,
    YahooFinanceSource,
    DataValidator,
    DataCleaner,
    DataExporter,
    DataImporter,
    DataScheduler,
    ScheduleFrequency,
    PluginManager,
    ExportFormat
)


def example_1_multiple_data_sources():
    """Example 1: Using multiple data sources."""
    print("\n" + "="*60)
    print("Example 1: Multiple Data Sources")
    print("="*60)

    # Yahoo Finance (default, no API key needed)
    print("\n1. Yahoo Finance Data:")
    yahoo_source = YahooFinanceSource()
    yahoo_data = yahoo_source.fetch_historical_data(
        symbols=['AAPL', 'MSFT'],
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        frequency='daily'
    )
    print(f"Fetched {len(yahoo_data)} rows from Yahoo Finance")
    print(yahoo_data.head())

    # Alpha Vantage (requires API key)
    print("\n2. Alpha Vantage Data:")
    print("Note: Requires API key from https://www.alphavantage.co/support/#api-key")
    # av_source = AlphaVantageSource(api_key='YOUR_API_KEY')
    # av_data = av_source.fetch_historical_data(
    #     symbols=['AAPL'],
    #     frequency='daily'
    # )

    # FRED Economic Data (requires API key)
    print("\n3. FRED Economic Data:")
    print("Note: Requires API key from https://fred.stlouisfed.org/docs/api/api_key.html")
    # fred_source = FREDSource(api_key='YOUR_FRED_API_KEY')
    # gdp_data = fred_source.fetch_historical_data(
    #     symbols=['GDP', 'UNRATE', 'DGS10'],  # GDP, Unemployment Rate, 10-Year Treasury
    #     start_date=datetime(2020, 1, 1),
    #     end_date=datetime.now()
    # )
    # print(f"Fetched {len(gdp_data)} economic data points")

    # NASDAQ Data Link (requires API key)
    print("\n4. NASDAQ Data Link (Quandl):")
    print("Note: Requires API key from https://data.nasdaq.com/")
    # nasdaq_source = NasdaqDataLinkSource(api_key='YOUR_NASDAQ_KEY')
    # nasdaq_data = nasdaq_source.fetch_historical_data(
    #     symbols=['WIKI/AAPL'],
    #     start_date=datetime(2023, 1, 1)
    # )


def example_2_data_validation():
    """Example 2: Data validation and cleaning."""
    print("\n" + "="*60)
    print("Example 2: Data Validation & Cleaning")
    print("="*60)

    # Fetch some data
    yahoo_source = YahooFinanceSource()
    data = yahoo_source.fetch_historical_data(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now()
    )

    print(f"\nOriginal data: {len(data)} rows")

    # Validate data
    print("\n1. Validating data...")
    validator = DataValidator(strict_mode=False)
    report = validator.validate(data)

    print(f"\nValidation Summary:")
    print(f"  Total rows: {report.total_rows}")
    print(f"  Total columns: {report.total_columns}")
    print(f"  Passed: {report.passed}")
    print(f"\nIssues by severity:")
    for severity, count in report.get_summary().items():
        print(f"  {severity.upper()}: {count}")

    # Show issues
    if report.issues:
        print(f"\nFound {len(report.issues)} issues:")
        for issue in report.issues[:5]:  # Show first 5
            print(f"  - [{issue.severity.value}] {issue.issue_type}: {issue.description}")

    # Clean data
    print("\n2. Cleaning data...")
    cleaner = DataCleaner()
    cleaned_data, cleaning_log = cleaner.clean(data)

    print(f"\nCleaned data: {len(cleaned_data)} rows")
    print(f"Cleaning actions performed:")
    for action in cleaning_log['actions']:
        print(f"  - {action.get('method', 'Unknown')}: {action}")


def example_3_data_export():
    """Example 3: Exporting data to various formats."""
    print("\n" + "="*60)
    print("Example 3: Data Export")
    print("="*60)

    # Fetch data
    yahoo_source = YahooFinanceSource()
    data = yahoo_source.fetch_historical_data(
        symbols=['AAPL', 'MSFT'],
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )

    # Create exporter
    exporter = DataExporter(output_dir="data/exports")

    # Export to CSV
    print("\n1. Exporting to CSV...")
    csv_path = exporter.export_to_csv(data, "market_data_sample")
    print(f"   Exported to: {csv_path}")

    # Export to Excel
    print("\n2. Exporting to Excel...")
    excel_path = exporter.export_to_excel(data, "market_data_sample")
    print(f"   Exported to: {excel_path}")

    # Export to Parquet
    print("\n3. Exporting to Parquet...")
    parquet_path = exporter.export_to_parquet(data, "market_data_sample")
    print(f"   Exported to: {parquet_path}")

    # Export to JSON
    print("\n4. Exporting to JSON...")
    json_path = exporter.export_to_json(data, "market_data_sample")
    print(f"   Exported to: {json_path}")

    # Export with metadata
    print("\n5. Exporting with metadata...")

    # Get date column name - check for 'Date' or use index if it's a DatetimeIndex
    date_col = None
    if 'Date' in data.columns:
        date_col = 'Date'
    elif 'date' in data.columns:
        date_col = 'date'
    elif isinstance(data.index, pd.DatetimeIndex):
        date_range = f"{data.index.min()} to {data.index.max()}"
    else:
        date_range = "Unknown"

    if date_col:
        date_range = f"{data[date_col].min()} to {data[date_col].max()}"

    metadata = {
        "symbols": ["AAPL", "MSFT"],
        "date_range": date_range,
        "export_date": str(datetime.now()),
        "source": "Yahoo Finance"
    }
    meta_path = exporter.export_with_metadata(
        data,
        "market_data_with_metadata",
        metadata,
        format=ExportFormat.PARQUET
    )
    print(f"   Exported to: {meta_path}")

    # Get export info
    print("\n6. Export directory info:")
    info = exporter.get_export_info()
    print(f"   Total files: {info['total_files']}")
    print(f"   Total size: {info['total_size_mb']:.2f} MB")

    # Import data back
    print("\n7. Importing data back...")
    importer = DataImporter(input_dir="data/exports")
    imported_data = importer.import_data("market_data_sample.parquet")
    print(f"   Imported {len(imported_data)} rows")


def example_4_scheduled_updates():
    """Example 4: Automated data updates with scheduling."""
    print("\n" + "="*60)
    print("Example 4: Scheduled Data Updates")
    print("="*60)

    # Create data provider
    from allocation_station.data.market_data import MarketDataProvider, MarketDataConfig, DataSource
    config = MarketDataConfig(data_source=DataSource.YAHOO, cache_dir="data/cache")
    data_provider = MarketDataProvider(config)

    # Create scheduler
    print("\n1. Creating scheduler...")
    scheduler = DataScheduler(
        data_provider=data_provider,
        cache_dir="data/cache",
        log_dir="data/logs"
    )

    # Add daily update job
    print("\n2. Adding daily market data update job...")
    daily_job = scheduler.add_update_job(
        job_id="daily_portfolio_update",
        name="Daily Portfolio Update",
        symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        frequency=ScheduleFrequency.DAILY,
        time_of_day=time(9, 30)  # 9:30 AM
    )
    print(f"   Job added: {daily_job.name}")
    print(f"   Next run: {daily_job.next_run}")

    # Add market close update
    print("\n3. Adding market close update job...")
    close_job = scheduler.add_market_data_refresh_job(
        symbols=['SPY', 'QQQ', 'IWM'],
        frequency=ScheduleFrequency.MARKET_CLOSE
    )
    print(f"   Job added: {close_job.name}")
    print(f"   Next run: {close_job.next_run}")

    # Add weekly cleanup
    print("\n4. Adding weekly cache cleanup job...")
    scheduler.add_cache_cleanup_job(
        max_age_days=30,
        frequency=ScheduleFrequency.WEEKLY
    )

    # Start scheduler (commented out to prevent actual scheduling in example)
    # print("\n5. Starting scheduler...")
    # scheduler.start()

    # Get statistics
    print("\n5. Scheduler statistics:")
    stats = scheduler.get_statistics()
    print(f"   Total jobs: {stats['total_jobs']}")
    print(f"   Enabled jobs: {stats['enabled_jobs']}")
    print(f"   Running: {stats['scheduler_running']}")

    print("\n6. All scheduled jobs:")
    for job in scheduler.get_all_jobs():
        print(f"   - {job.name}")
        print(f"     Frequency: {job.frequency}")
        print(f"     Next run: {job.next_run}")
        print(f"     Symbols: {len(job.symbols)}")

    # Export schedule
    print("\n7. Exporting schedule configuration...")
    scheduler.export_schedule("data/schedule_config.json")
    print("   Schedule exported to data/schedule_config.json")

    # Clean up
    # scheduler.stop()


def example_5_custom_plugin():
    """Example 5: Using custom data source plugins."""
    print("\n" + "="*60)
    print("Example 5: Custom Data Source Plugins")
    print("="*60)

    # Create plugin manager
    print("\n1. Creating plugin manager...")
    plugin_manager = PluginManager(plugin_dir="data/plugins")

    # Discover available plugins
    print("\n2. Discovering plugins...")
    plugins = plugin_manager.discover_plugins()
    print(f"   Found {len(plugins)} plugins: {plugins}")

    # List available plugins
    print("\n3. Available plugins:")
    for plugin_info in plugin_manager.list_available_plugins():
        print(f"   - {plugin_info['name']}")
        print(f"     Loaded: {plugin_info['loaded']}")
        if plugin_info['metadata']:
            print(f"     Version: {plugin_info['metadata']['version']}")
            print(f"     Description: {plugin_info['metadata']['description']}")

    # Load and use CSV plugin
    print("\n4. Loading CSV data source plugin...")
    try:
        csv_plugin = plugin_manager.load_plugin(
            "CSVDataSourcePlugin",
            config={'data_dir': 'data/csv'}
        )

        # Test connection
        print(f"   Plugin loaded: {csv_plugin.metadata.name}")
        print(f"   Testing connection: {csv_plugin.test_connection()}")

        # Get available symbols
        symbols = csv_plugin.get_available_symbols()
        print(f"   Available symbols: {symbols}")

    except Exception as e:
        print(f"   Could not load plugin: {e}")
        print("   (This is expected if CSV files haven't been created yet)")


def example_6_comprehensive_workflow():
    """Example 6: Comprehensive workflow combining all features."""
    print("\n" + "="*60)
    print("Example 6: Comprehensive Workflow")
    print("="*60)

    symbols = ['AAPL', 'MSFT', 'GOOGL']

    # Step 1: Fetch data
    print("\n1. Fetching data from Yahoo Finance...")
    yahoo_source = YahooFinanceSource()
    data = yahoo_source.fetch_historical_data(
        symbols=symbols,
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now()
    )
    print(f"   Fetched {len(data)} rows")

    # Step 2: Validate
    print("\n2. Validating data quality...")
    validator = DataValidator()
    report = validator.validate(data)
    print(f"   Validation passed: {report.passed}")
    print(f"   Issues found: {len(report.issues)}")

    # Step 3: Clean
    print("\n3. Cleaning data...")
    cleaner = DataCleaner()
    cleaned_data, log = cleaner.clean(data)
    print(f"   Cleaned data: {len(cleaned_data)} rows")

    # Step 4: Export to multiple formats
    print("\n4. Exporting to multiple formats...")
    exporter = DataExporter(output_dir="data/exports")

    exports = {
        'csv': exporter.export_to_csv(cleaned_data, "portfolio_data"),
        'excel': exporter.export_to_excel(cleaned_data, "portfolio_data"),
        'parquet': exporter.export_to_parquet(cleaned_data, "portfolio_data")
    }

    for format_name, path in exports.items():
        print(f"   Exported {format_name.upper()}: {path}")

    # Step 5: Setup automated updates
    print("\n5. Setting up automated updates...")
    from allocation_station.data.market_data import MarketDataProvider, MarketDataConfig
    config = MarketDataConfig(cache_dir="data/cache")
    data_provider = MarketDataProvider(config)

    scheduler = DataScheduler(data_provider, cache_dir="data/cache")
    job = scheduler.add_portfolio_update_job(
        portfolio_name="tech_portfolio",
        symbols=symbols,
        frequency=ScheduleFrequency.DAILY,
        time_of_day=time(16, 30)  # 4:30 PM
    )
    print(f"   Created scheduled job: {job.name}")
    print(f"   Next run: {job.next_run}")

    print("\nWorkflow complete!")


if __name__ == "__main__":
    """Run all examples."""

    print("\n" + "="*70)
    print(" Allocation Station - Enhanced Data Infrastructure Examples")
    print("="*70)

    try:
        # Run examples
        example_1_multiple_data_sources()
        example_2_data_validation()
        example_3_data_export()
        example_4_scheduled_updates()
        example_5_custom_plugin()
        example_6_comprehensive_workflow()

        print("\n" + "="*70)
        print(" All examples completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n\nNOTE: Some features require API keys:")
    print("  - Alpha Vantage: https://www.alphavantage.co/support/#api-key")
    print("  - FRED: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("  - NASDAQ Data Link: https://data.nasdaq.com/")
    print("\nUncomment the relevant sections in the examples to use these sources.")
