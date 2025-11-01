"""Data export functionality for various formats."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from enum import Enum


class ExportFormat(str, Enum):
    """Supported export formats."""
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    JSON = "json"
    HDF5 = "hdf5"
    FEATHER = "feather"


class DataExporter:
    """
    Export market data to various file formats.

    Supports CSV, Excel, Parquet, JSON, HDF5, and Feather formats with
    configurable options for each format.
    """

    def __init__(self, output_dir: str = "data/exports"):
        """
        Initialize data exporter.

        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        data: pd.DataFrame,
        filename: str,
        format: ExportFormat = ExportFormat.CSV,
        **kwargs
    ) -> str:
        """
        Export data to specified format.

        Args:
            data: DataFrame to export
            filename: Output filename (without extension)
            format: Export format
            **kwargs: Format-specific options

        Returns:
            Path to exported file
        """
        if format == ExportFormat.CSV:
            return self.export_to_csv(data, filename, **kwargs)
        elif format == ExportFormat.EXCEL:
            return self.export_to_excel(data, filename, **kwargs)
        elif format == ExportFormat.PARQUET:
            return self.export_to_parquet(data, filename, **kwargs)
        elif format == ExportFormat.JSON:
            return self.export_to_json(data, filename, **kwargs)
        elif format == ExportFormat.HDF5:
            return self.export_to_hdf5(data, filename, **kwargs)
        elif format == ExportFormat.FEATHER:
            return self.export_to_feather(data, filename, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def export_to_csv(
        self,
        data: pd.DataFrame,
        filename: str,
        index: bool = False,
        compression: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export data to CSV format.

        Args:
            data: DataFrame to export
            filename: Output filename
            index: Include index in output
            compression: Compression type ('gzip', 'bz2', 'zip', 'xz')
            **kwargs: Additional pandas to_csv parameters

        Returns:
            Path to exported file
        """
        # Add appropriate extension
        if compression:
            ext = f".csv.{compression}"
        else:
            ext = ".csv"

        filepath = self.output_dir / f"{filename}{ext}"

        # Export
        data.to_csv(filepath, index=index, compression=compression, **kwargs)

        return str(filepath)

    def export_to_excel(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        filename: str,
        sheet_name: str = "Sheet1",
        index: bool = False,
        **kwargs
    ) -> str:
        """
        Export data to Excel format.

        Args:
            data: DataFrame or dict of DataFrames (for multiple sheets)
            filename: Output filename
            sheet_name: Sheet name (if single DataFrame)
            index: Include index in output
            **kwargs: Additional pandas to_excel parameters

        Returns:
            Path to exported file
        """
        filepath = self.output_dir / f"{filename}.xlsx"

        # Export
        if isinstance(data, dict):
            # Multiple sheets
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                for name, df in data.items():
                    # Remove timezone from datetime columns
                    df_copy = df.copy()
                    for col in df_copy.select_dtypes(include=['datetime64[ns, US/Eastern]', 'datetimetz']).columns:
                        df_copy[col] = df_copy[col].dt.tz_localize(None)
                    df_copy.to_excel(writer, sheet_name=name, index=index, **kwargs)
        else:
            # Single sheet - remove timezone from datetime columns
            data_copy = data.copy()
            for col in data_copy.select_dtypes(include=['datetime64[ns, US/Eastern]', 'datetimetz']).columns:
                data_copy[col] = data_copy[col].dt.tz_localize(None)
            data_copy.to_excel(filepath, sheet_name=sheet_name, index=index, **kwargs)

        return str(filepath)

    def export_to_parquet(
        self,
        data: pd.DataFrame,
        filename: str,
        compression: str = 'snappy',
        index: bool = False,
        **kwargs
    ) -> str:
        """
        Export data to Parquet format.

        Args:
            data: DataFrame to export
            filename: Output filename
            compression: Compression algorithm ('snappy', 'gzip', 'brotli', 'none')
            index: Include index in output
            **kwargs: Additional pandas to_parquet parameters

        Returns:
            Path to exported file
        """
        filepath = self.output_dir / f"{filename}.parquet"

        # Export
        data.to_parquet(filepath, compression=compression, index=index, **kwargs)

        return str(filepath)

    def export_to_json(
        self,
        data: pd.DataFrame,
        filename: str,
        orient: str = 'records',
        indent: int = 2,
        date_format: str = 'iso',
        **kwargs
    ) -> str:
        """
        Export data to JSON format.

        Args:
            data: DataFrame to export
            filename: Output filename
            orient: JSON structure ('records', 'index', 'columns', 'values', 'table')
            indent: Indentation level
            date_format: Date format ('iso', 'epoch')
            **kwargs: Additional pandas to_json parameters

        Returns:
            Path to exported file
        """
        filepath = self.output_dir / f"{filename}.json"

        # Export
        data.to_json(
            filepath,
            orient=orient,
            indent=indent,
            date_format=date_format,
            **kwargs
        )

        return str(filepath)

    def export_to_hdf5(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        filename: str,
        key: str = 'data',
        mode: str = 'w',
        complevel: int = 9,
        **kwargs
    ) -> str:
        """
        Export data to HDF5 format.

        Args:
            data: DataFrame or dict of DataFrames
            filename: Output filename
            key: Key for data storage (if single DataFrame)
            mode: File mode ('w', 'a', 'r+')
            complevel: Compression level (0-9)
            **kwargs: Additional pandas to_hdf parameters

        Returns:
            Path to exported file
        """
        filepath = self.output_dir / f"{filename}.h5"

        # Export
        if isinstance(data, dict):
            # Multiple datasets
            for name, df in data.items():
                df.to_hdf(
                    filepath,
                    key=name,
                    mode='a' if name != list(data.keys())[0] else mode,
                    complevel=complevel,
                    **kwargs
                )
        else:
            # Single dataset
            data.to_hdf(filepath, key=key, mode=mode, complevel=complevel, **kwargs)

        return str(filepath)

    def export_to_feather(
        self,
        data: pd.DataFrame,
        filename: str,
        compression: str = 'lz4',
        **kwargs
    ) -> str:
        """
        Export data to Feather format.

        Args:
            data: DataFrame to export
            filename: Output filename
            compression: Compression algorithm ('lz4', 'zstd', 'uncompressed')
            **kwargs: Additional pandas to_feather parameters

        Returns:
            Path to exported file
        """
        filepath = self.output_dir / f"{filename}.feather"

        # Export
        data.to_feather(filepath, compression=compression, **kwargs)

        return str(filepath)

    def export_portfolio_data(
        self,
        data: pd.DataFrame,
        filename: str,
        format: ExportFormat = ExportFormat.EXCEL,
        include_summary: bool = True
    ) -> str:
        """
        Export portfolio data with optional summary statistics.

        Args:
            data: Portfolio data
            filename: Output filename
            format: Export format
            include_summary: Include summary sheet/section

        Returns:
            Path to exported file
        """
        if include_summary and format == ExportFormat.EXCEL:
            # Create summary statistics
            summary = self._create_portfolio_summary(data)

            # Export both data and summary
            return self.export_to_excel(
                data={
                    'Data': data,
                    'Summary': summary
                },
                filename=filename
            )
        else:
            return self.export(data, filename, format)

    def _create_portfolio_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics for portfolio data."""
        summary_data = []

        # Basic statistics
        summary_data.append({'Metric': 'Total Records', 'Value': len(data)})

        if 'symbol' in data.columns:
            summary_data.append({
                'Metric': 'Unique Symbols',
                'Value': data['symbol'].nunique()
            })

        if 'date' in data.columns:
            summary_data.append({
                'Metric': 'Date Range',
                'Value': f"{data['date'].min()} to {data['date'].max()}"
            })

        # Price statistics
        if 'close' in data.columns:
            summary_data.append({
                'Metric': 'Average Close Price',
                'Value': f"{data['close'].mean():.2f}"
            })
            summary_data.append({
                'Metric': 'Min Close Price',
                'Value': f"{data['close'].min():.2f}"
            })
            summary_data.append({
                'Metric': 'Max Close Price',
                'Value': f"{data['close'].max():.2f}"
            })

        # Volume statistics
        if 'volume' in data.columns:
            summary_data.append({
                'Metric': 'Average Volume',
                'Value': f"{data['volume'].mean():.0f}"
            })
            summary_data.append({
                'Metric': 'Total Volume',
                'Value': f"{data['volume'].sum():.0f}"
            })

        return pd.DataFrame(summary_data)

    def export_batch(
        self,
        data_dict: Dict[str, pd.DataFrame],
        base_filename: str,
        format: ExportFormat = ExportFormat.CSV
    ) -> List[str]:
        """
        Export multiple DataFrames to separate files.

        Args:
            data_dict: Dictionary mapping names to DataFrames
            base_filename: Base filename for exports
            format: Export format

        Returns:
            List of paths to exported files
        """
        exported_files = []

        for name, data in data_dict.items():
            filename = f"{base_filename}_{name}"
            filepath = self.export(data, filename, format)
            exported_files.append(filepath)

        return exported_files

    def export_with_metadata(
        self,
        data: pd.DataFrame,
        filename: str,
        metadata: Dict[str, Any],
        format: ExportFormat = ExportFormat.PARQUET
    ) -> str:
        """
        Export data with metadata.

        Args:
            data: DataFrame to export
            filename: Output filename
            metadata: Metadata dictionary
            format: Export format

        Returns:
            Path to exported file
        """
        if format == ExportFormat.PARQUET:
            # Parquet supports metadata
            filepath = self.output_dir / f"{filename}.parquet"
            data.to_parquet(filepath, index=False)

            # Write metadata to separate JSON file
            metadata_path = self.output_dir / f"{filename}_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            return str(filepath)

        elif format == ExportFormat.HDF5:
            # HDF5 supports attributes
            filepath = self.output_dir / f"{filename}.h5"

            with pd.HDFStore(filepath, mode='w') as store:
                store.put('data', data)
                store.get_storer('data').attrs.metadata = metadata

            return str(filepath)

        else:
            # For other formats, export data normally and metadata separately
            filepath = self.export(data, filename, format)

            metadata_path = self.output_dir / f"{filename}_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            return filepath

    def export_time_series(
        self,
        data: pd.DataFrame,
        filename: str,
        symbol_column: str = 'symbol',
        date_column: str = 'date',
        value_columns: Optional[List[str]] = None,
        format: ExportFormat = ExportFormat.PARQUET
    ) -> str:
        """
        Export time series data in optimized format.

        Args:
            data: Time series data
            filename: Output filename
            symbol_column: Column containing symbols
            date_column: Column containing dates
            value_columns: Columns to include (None = all numeric)
            format: Export format

        Returns:
            Path to exported file
        """
        # Select columns
        if value_columns is None:
            value_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        columns_to_export = [symbol_column, date_column] + value_columns
        export_data = data[columns_to_export].copy()

        # Sort by date for efficient querying
        export_data = export_data.sort_values([symbol_column, date_column])

        # Export
        return self.export(export_data, filename, format)

    def get_export_info(self) -> Dict[str, Any]:
        """
        Get information about export directory and files.

        Returns:
            Dictionary with export information
        """
        files = list(self.output_dir.glob('*'))

        file_info = []
        total_size = 0

        for file in files:
            if file.is_file():
                size = file.stat().st_size
                total_size += size

                file_info.append({
                    'name': file.name,
                    'size': size,
                    'size_mb': size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(file.stat().st_mtime)
                })

        return {
            'export_directory': str(self.output_dir),
            'total_files': len(file_info),
            'total_size_mb': total_size / (1024 * 1024),
            'files': file_info
        }


class DataImporter:
    """
    Import data from various file formats.

    Companion to DataExporter for reading exported data.
    """

    def __init__(self, input_dir: str = "data/exports"):
        """
        Initialize data importer.

        Args:
            input_dir: Directory containing files to import
        """
        self.input_dir = Path(input_dir)

    def import_data(
        self,
        filename: str,
        format: Optional[ExportFormat] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Import data from file.

        Args:
            filename: File to import
            format: File format (auto-detected if None)
            **kwargs: Format-specific options

        Returns:
            Imported DataFrame
        """
        filepath = self.input_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Auto-detect format from extension
        if format is None:
            ext = filepath.suffix.lower()
            format_map = {
                '.csv': ExportFormat.CSV,
                '.xlsx': ExportFormat.EXCEL,
                '.xls': ExportFormat.EXCEL,
                '.parquet': ExportFormat.PARQUET,
                '.json': ExportFormat.JSON,
                '.h5': ExportFormat.HDF5,
                '.hdf5': ExportFormat.HDF5,
                '.feather': ExportFormat.FEATHER,
            }
            format = format_map.get(ext)

        # Import based on format
        if format == ExportFormat.CSV:
            return pd.read_csv(filepath, **kwargs)
        elif format == ExportFormat.EXCEL:
            return pd.read_excel(filepath, **kwargs)
        elif format == ExportFormat.PARQUET:
            return pd.read_parquet(filepath, **kwargs)
        elif format == ExportFormat.JSON:
            return pd.read_json(filepath, **kwargs)
        elif format == ExportFormat.HDF5:
            return pd.read_hdf(filepath, **kwargs)
        elif format == ExportFormat.FEATHER:
            return pd.read_feather(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def import_with_metadata(
        self,
        filename: str,
        format: Optional[ExportFormat] = None
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Import data with metadata.

        Args:
            filename: Base filename
            format: File format

        Returns:
            Tuple of (data, metadata)
        """
        # Import data
        data = self.import_data(filename, format)

        # Try to import metadata
        metadata = {}
        base_name = Path(filename).stem
        metadata_file = self.input_dir / f"{base_name}_metadata.json"

        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        return data, metadata
