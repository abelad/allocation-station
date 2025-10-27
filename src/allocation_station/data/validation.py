"""Data validation and cleaning pipelines for market data."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationIssue(BaseModel):
    """Represents a data validation issue."""
    severity: ValidationSeverity
    issue_type: str
    description: str
    affected_rows: Optional[List[int]] = None
    affected_columns: Optional[List[str]] = None
    count: int = 0
    details: Optional[Dict[str, Any]] = None


class ValidationReport(BaseModel):
    """Comprehensive validation report."""
    timestamp: datetime = Field(default_factory=datetime.now)
    total_rows: int
    total_columns: int
    issues: List[ValidationIssue] = Field(default_factory=list)
    passed: bool = True
    summary: Dict[str, int] = Field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue to the report."""
        self.issues.append(issue)
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.passed = False

    def get_summary(self) -> Dict[str, int]:
        """Get summary of issues by severity."""
        summary = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'critical': 0
        }
        for issue in self.issues:
            summary[issue.severity.value] += 1
        return summary


class DataValidator:
    """
    Comprehensive data validation for market data.

    Performs various checks to ensure data quality and integrity.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize data validator.

        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode

    def validate(self, data: pd.DataFrame) -> ValidationReport:
        """
        Run comprehensive validation on market data.

        Args:
            data: DataFrame to validate

        Returns:
            ValidationReport with all findings
        """
        report = ValidationReport(
            total_rows=len(data),
            total_columns=len(data.columns)
        )

        # Run all validation checks
        self._check_missing_values(data, report)
        self._check_duplicates(data, report)
        self._check_data_types(data, report)
        self._check_price_anomalies(data, report)
        self._check_volume_anomalies(data, report)
        self._check_date_gaps(data, report)
        self._check_negative_prices(data, report)
        self._check_outliers(data, report)
        self._check_constant_values(data, report)
        self._check_correlation_breaks(data, report)

        report.summary = report.get_summary()

        return report

    def _check_missing_values(self, data: pd.DataFrame, report: ValidationReport):
        """Check for missing values."""
        missing = data.isnull().sum()
        missing = missing[missing > 0]

        if not missing.empty:
            for col, count in missing.items():
                pct = (count / len(data)) * 100
                severity = ValidationSeverity.CRITICAL if pct > 50 else \
                          ValidationSeverity.ERROR if pct > 10 else \
                          ValidationSeverity.WARNING

                report.add_issue(ValidationIssue(
                    severity=severity,
                    issue_type="missing_values",
                    description=f"Column '{col}' has {count} missing values ({pct:.2f}%)",
                    affected_columns=[col],
                    count=count,
                    details={'percentage': pct}
                ))

    def _check_duplicates(self, data: pd.DataFrame, report: ValidationReport):
        """Check for duplicate rows."""
        if 'date' in data.columns and 'symbol' in data.columns:
            duplicates = data.duplicated(subset=['date', 'symbol'], keep=False)
            dup_count = duplicates.sum()

            if dup_count > 0:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    issue_type="duplicates",
                    description=f"Found {dup_count} duplicate date-symbol combinations",
                    affected_rows=data[duplicates].index.tolist()[:100],  # Limit to 100
                    count=dup_count
                ))

    def _check_data_types(self, data: pd.DataFrame, report: ValidationReport):
        """Check for incorrect data types."""
        expected_numeric = ['open', 'high', 'low', 'close', 'volume', 'price']

        for col in data.columns:
            if any(expected in col.lower() for expected in expected_numeric):
                if not pd.api.types.is_numeric_dtype(data[col]):
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        issue_type="data_type",
                        description=f"Column '{col}' should be numeric but is {data[col].dtype}",
                        affected_columns=[col]
                    ))

    def _check_price_anomalies(self, data: pd.DataFrame, report: ValidationReport):
        """Check for price anomalies (OHLC relationships)."""
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= Open, Close, Low
            high_low = data['high'] < data['low']
            high_open = data['high'] < data['open']
            high_close = data['high'] < data['close']
            low_high = data['low'] > data['high']

            anomalies = high_low | high_open | high_close | low_high
            anomaly_count = anomalies.sum()

            if anomaly_count > 0:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    issue_type="price_anomaly",
                    description=f"Found {anomaly_count} rows with invalid OHLC relationships",
                    affected_rows=data[anomalies].index.tolist()[:100],
                    count=anomaly_count
                ))

    def _check_volume_anomalies(self, data: pd.DataFrame, report: ValidationReport):
        """Check for unusual volume patterns."""
        if 'volume' in data.columns:
            # Check for zero volume
            zero_volume = (data['volume'] == 0)
            zero_count = zero_volume.sum()

            if zero_count > 0:
                pct = (zero_count / len(data)) * 100
                severity = ValidationSeverity.WARNING if pct < 5 else ValidationSeverity.ERROR

                report.add_issue(ValidationIssue(
                    severity=severity,
                    issue_type="zero_volume",
                    description=f"Found {zero_count} rows with zero volume ({pct:.2f}%)",
                    count=zero_count,
                    details={'percentage': pct}
                ))

    def _check_date_gaps(self, data: pd.DataFrame, report: ValidationReport):
        """Check for gaps in date sequence."""
        if 'date' not in data.columns:
            return

        # Convert date to datetime if necessary
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            try:
                dates = pd.to_datetime(data['date'])
            except:
                return
        else:
            dates = data['date']

        # Check for each symbol separately
        if 'symbol' in data.columns:
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol].copy()
                symbol_dates = pd.to_datetime(symbol_data['date'])

                if len(symbol_dates) > 1:
                    symbol_dates_sorted = symbol_dates.sort_values()
                    expected_dates = pd.bdate_range(
                        start=symbol_dates_sorted.min(),
                        end=symbol_dates_sorted.max()
                    )

                    missing_dates = expected_dates.difference(symbol_dates_sorted)

                    if len(missing_dates) > 0:
                        # Only report if missing more than 5 business days
                        if len(missing_dates) > 5:
                            report.add_issue(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                issue_type="date_gaps",
                                description=f"Symbol '{symbol}' has {len(missing_dates)} missing business days",
                                count=len(missing_dates),
                                details={'symbol': symbol, 'sample_missing_dates': [str(d.date()) for d in missing_dates[:5]]}
                            ))

    def _check_negative_prices(self, data: pd.DataFrame, report: ValidationReport):
        """Check for negative prices."""
        price_cols = ['open', 'high', 'low', 'close', 'price']

        for col in price_cols:
            if col in data.columns:
                negative = data[col] < 0
                negative_count = negative.sum()

                if negative_count > 0:
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        issue_type="negative_price",
                        description=f"Column '{col}' has {negative_count} negative values",
                        affected_rows=data[negative].index.tolist(),
                        affected_columns=[col],
                        count=negative_count
                    ))

    def _check_outliers(self, data: pd.DataFrame, report: ValidationReport):
        """Check for statistical outliers using IQR method."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['volume']:  # Skip volume as it naturally has high variance
                continue

            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            outliers = (data[col] < Q1 - 3 * IQR) | (data[col] > Q3 + 3 * IQR)
            outlier_count = outliers.sum()

            if outlier_count > 0:
                pct = (outlier_count / len(data)) * 100

                # Only report if outliers are significant
                if pct > 1:
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        issue_type="outliers",
                        description=f"Column '{col}' has {outlier_count} statistical outliers ({pct:.2f}%)",
                        affected_columns=[col],
                        count=outlier_count,
                        details={'percentage': pct, 'Q1': Q1, 'Q3': Q3, 'IQR': IQR}
                    ))

    def _check_constant_values(self, data: pd.DataFrame, report: ValidationReport):
        """Check for columns with constant values."""
        for col in data.columns:
            if data[col].dtype in [np.number, np.float64, np.int64]:
                if data[col].nunique() == 1:
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        issue_type="constant_values",
                        description=f"Column '{col}' has only one unique value: {data[col].iloc[0]}",
                        affected_columns=[col]
                    ))

    def _check_correlation_breaks(self, data: pd.DataFrame, report: ValidationReport):
        """Check for correlation structure breaks."""
        if 'symbol' in data.columns and len(data['symbol'].unique()) > 1:
            # Check if we have price data
            if 'close' in data.columns:
                # Pivot to get prices by symbol
                try:
                    price_pivot = data.pivot(index='date', columns='symbol', values='close')

                    if price_pivot.shape[1] > 1:
                        # Calculate correlation
                        corr = price_pivot.corr()

                        # Check for suspiciously low correlations between what should be related assets
                        low_corr = (corr < 0.3) & (corr > -0.3)
                        low_corr_count = (low_corr.sum().sum() - len(corr)) / 2  # Exclude diagonal

                        if low_corr_count > 0:
                            report.add_issue(ValidationIssue(
                                severity=ValidationSeverity.INFO,
                                issue_type="low_correlation",
                                description=f"Found {int(low_corr_count)} asset pairs with unusually low correlation",
                                count=int(low_corr_count)
                            ))
                except:
                    pass  # Skip if pivot fails


class DataCleaner:
    """
    Data cleaning and preprocessing for market data.

    Provides methods to clean and fix common data quality issues.
    """

    def __init__(self):
        """Initialize data cleaner."""
        pass

    def clean(
        self,
        data: pd.DataFrame,
        validation_report: Optional[ValidationReport] = None,
        methods: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean data based on validation report or specified methods.

        Args:
            data: DataFrame to clean
            validation_report: Optional validation report to guide cleaning
            methods: List of cleaning methods to apply

        Returns:
            Tuple of (cleaned_data, cleaning_log)
        """
        cleaned_data = data.copy()
        cleaning_log = {
            'timestamp': datetime.now(),
            'actions': []
        }

        # Determine which cleaning methods to apply
        if methods is None:
            methods = [
                'remove_duplicates',
                'fill_missing_values',
                'fix_data_types',
                'remove_negative_prices',
                'handle_outliers',
                'sort_by_date'
            ]

        # Apply cleaning methods
        for method in methods:
            if method == 'remove_duplicates':
                cleaned_data, log = self._remove_duplicates(cleaned_data)
                cleaning_log['actions'].append(log)
            elif method == 'fill_missing_values':
                cleaned_data, log = self._fill_missing_values(cleaned_data)
                cleaning_log['actions'].append(log)
            elif method == 'fix_data_types':
                cleaned_data, log = self._fix_data_types(cleaned_data)
                cleaning_log['actions'].append(log)
            elif method == 'remove_negative_prices':
                cleaned_data, log = self._remove_negative_prices(cleaned_data)
                cleaning_log['actions'].append(log)
            elif method == 'handle_outliers':
                cleaned_data, log = self._handle_outliers(cleaned_data)
                cleaning_log['actions'].append(log)
            elif method == 'sort_by_date':
                cleaned_data, log = self._sort_by_date(cleaned_data)
                cleaning_log['actions'].append(log)

        return cleaned_data, cleaning_log

    def _remove_duplicates(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove duplicate rows."""
        initial_count = len(data)

        if 'date' in data.columns and 'symbol' in data.columns:
            data = data.drop_duplicates(subset=['date', 'symbol'], keep='first')
        else:
            data = data.drop_duplicates()

        removed_count = initial_count - len(data)

        return data, {
            'method': 'remove_duplicates',
            'rows_removed': removed_count,
            'rows_remaining': len(data)
        }

    def _fill_missing_values(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fill missing values using appropriate methods."""
        filled_cols = []

        # For price columns, use forward fill then backward fill
        price_cols = ['open', 'high', 'low', 'close', 'price']
        for col in price_cols:
            if col in data.columns:
                missing_before = data[col].isnull().sum()
                if missing_before > 0:
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                    filled_cols.append(col)

        # For volume, fill with 0 or median
        if 'volume' in data.columns:
            missing_before = data['volume'].isnull().sum()
            if missing_before > 0:
                data['volume'] = data['volume'].fillna(0)
                filled_cols.append('volume')

        return data, {
            'method': 'fill_missing_values',
            'columns_filled': filled_cols
        }

    def _fix_data_types(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fix data types for common columns."""
        fixed_cols = []

        # Convert date columns
        if 'date' in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])
                fixed_cols.append('date')

        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'price']
        for col in numeric_cols:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    fixed_cols.append(col)

        return data, {
            'method': 'fix_data_types',
            'columns_fixed': fixed_cols
        }

    def _remove_negative_prices(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove rows with negative prices."""
        initial_count = len(data)

        price_cols = ['open', 'high', 'low', 'close', 'price']
        for col in price_cols:
            if col in data.columns:
                data = data[data[col] >= 0]

        removed_count = initial_count - len(data)

        return data, {
            'method': 'remove_negative_prices',
            'rows_removed': removed_count
        }

    def _handle_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'winsorize',
        threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle outliers using specified method."""
        handled_cols = []

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['volume']:  # Skip volume
                continue

            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)

            if outliers.sum() > 0:
                if method == 'winsorize':
                    # Cap outliers at bounds
                    data.loc[data[col] < lower_bound, col] = lower_bound
                    data.loc[data[col] > upper_bound, col] = upper_bound
                    handled_cols.append(col)
                elif method == 'remove':
                    # Remove outliers
                    data = data[~outliers]
                    handled_cols.append(col)

        return data, {
            'method': f'handle_outliers ({method})',
            'columns_handled': handled_cols
        }

    def _sort_by_date(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Sort data by date."""
        if 'date' in data.columns:
            data = data.sort_values('date')
            return data, {
                'method': 'sort_by_date',
                'sorted': True
            }

        return data, {
            'method': 'sort_by_date',
            'sorted': False,
            'reason': 'No date column found'
        }

    def interpolate_missing_data(
        self,
        data: pd.DataFrame,
        method: str = 'linear',
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Interpolate missing data points.

        Args:
            data: DataFrame with missing values
            method: Interpolation method ('linear', 'polynomial', 'spline')
            limit: Maximum number of consecutive NaNs to fill

        Returns:
            DataFrame with interpolated values
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if data[col].isnull().any():
                data[col] = data[col].interpolate(method=method, limit=limit)

        return data

    def resample_data(
        self,
        data: pd.DataFrame,
        frequency: str = 'D',
        aggregation: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Resample data to different frequency.

        Args:
            data: DataFrame to resample
            frequency: Target frequency ('D', 'W', 'M', 'Q', 'Y')
            aggregation: Dictionary mapping columns to aggregation functions

        Returns:
            Resampled DataFrame
        """
        if 'date' not in data.columns:
            raise ValueError("Data must have a 'date' column")

        # Set date as index
        data = data.set_index('date')

        # Default aggregation
        if aggregation is None:
            aggregation = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }

        # Apply aggregation only to existing columns
        agg_dict = {col: func for col, func in aggregation.items() if col in data.columns}

        # Resample
        resampled = data.resample(frequency).agg(agg_dict)

        return resampled.reset_index()
