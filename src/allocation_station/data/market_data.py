"""Market data fetching and management."""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path


class DataSource(str, Enum):
    """Available data sources."""

    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    IEX = "iex"
    FRED = "fred"
    CSV = "csv"
    DATABASE = "database"


class DataFrequency(str, Enum):
    """Data frequency options."""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"


class MarketDataConfig(BaseModel):
    """Configuration for market data fetching."""

    source: DataSource = Field(DataSource.YAHOO, description="Data source")
    api_key: Optional[str] = Field(None, description="API key for data source")
    cache_dir: str = Field("data/cache", description="Cache directory")
    cache_expiry: int = Field(3600, description="Cache expiry in seconds")
    retry_count: int = Field(3, description="Number of retries on failure")
    timeout: int = Field(30, description="Request timeout in seconds")


class MarketDataProvider:
    """
    Manages market data fetching from various sources.

    This class provides a unified interface for fetching market data
    from different providers and manages caching for efficiency.
    """

    def __init__(self, config: Optional[MarketDataConfig] = None):
        """Initialize market data provider."""
        self.config = config or MarketDataConfig()
        self._setup_cache_dir()
        self._data_cache = {}

    def _setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def fetch_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: DataFrequency = DataFrequency.DAILY,
        adjust_for_splits: bool = True,
        adjust_for_dividends: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical price data for given symbols.

        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency
            adjust_for_splits: Adjust prices for stock splits
            adjust_for_dividends: Adjust prices for dividends

        Returns:
            DataFrame with historical price data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Check cache first
        cache_key = self._get_cache_key(symbols, start_date, end_date, frequency)
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch based on configured source
        if self.config.source == DataSource.YAHOO:
            data = self._fetch_yahoo_data(symbols, start_date, end_date, frequency)
        elif self.config.source == DataSource.CSV:
            data = self._fetch_csv_data(symbols)
        else:
            raise NotImplementedError(f"Data source {self.config.source} not implemented")

        # Cache the data
        self._cache_data(cache_key, data)

        return data

    def _fetch_yahoo_data(
        self,
        symbols: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        frequency: DataFrequency
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        # Set default date range if not provided
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=365*5))  # 5 years default

        # Map frequency to yfinance format
        freq_map = {
            DataFrequency.MINUTE_1: "1m",
            DataFrequency.MINUTE_5: "5m",
            DataFrequency.MINUTE_15: "15m",
            DataFrequency.MINUTE_30: "30m",
            DataFrequency.HOURLY: "1h",
            DataFrequency.DAILY: "1d",
            DataFrequency.WEEKLY: "1wk",
            DataFrequency.MONTHLY: "1mo",
        }

        interval = freq_map.get(frequency, "1d")

        # Fetch data for each symbol
        data_frames = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True
                )

                if not df.empty:
                    # Add symbol column
                    df['symbol'] = symbol
                    # Standardize column names
                    df.columns = [col.lower() for col in df.columns]
                    data_frames.append(df)

            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue

        if data_frames:
            # Combine all dataframes
            combined = pd.concat(data_frames)
            return combined

        return pd.DataFrame()

    def _fetch_csv_data(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch data from CSV files."""
        data_frames = []

        for symbol in symbols:
            file_path = Path(self.config.cache_dir) / f"{symbol}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df['symbol'] = symbol
                data_frames.append(df)

        if data_frames:
            return pd.concat(data_frames)

        return pd.DataFrame()

    def fetch_real_time_data(self, symbols: Union[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Fetch real-time market data.

        Args:
            symbols: Single symbol or list of symbols

        Returns:
            Dictionary with real-time data for each symbol
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        real_time_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                real_time_data[symbol] = {
                    'price': info.get('regularMarketPrice', 0),
                    'previous_close': info.get('previousClose', 0),
                    'volume': info.get('volume', 0),
                    'bid': info.get('bid', 0),
                    'ask': info.get('ask', 0),
                    'day_high': info.get('dayHigh', 0),
                    'day_low': info.get('dayLow', 0),
                    'market_cap': info.get('marketCap', 0),
                }

            except Exception as e:
                print(f"Error fetching real-time data for {symbol}: {e}")
                real_time_data[symbol] = {}

        return real_time_data

    def fetch_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with fundamental data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'pe_ratio': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'market_cap': info.get('marketCap'),
                'revenue': info.get('totalRevenue'),
                'earnings': info.get('grossProfits'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'profit_margin': info.get('profitMargins'),
            }

        except Exception as e:
            print(f"Error fetching fundamental data for {symbol}: {e}")
            return {}

    def fetch_options_data(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch options chain data.

        Args:
            symbol: Stock symbol
            expiry: Option expiry date

        Returns:
            DataFrame with options data
        """
        try:
            ticker = yf.Ticker(symbol)

            if expiry:
                options = ticker.option_chain(expiry)
            else:
                # Get next expiry
                expiries = ticker.options
                if expiries:
                    options = ticker.option_chain(expiries[0])
                else:
                    return pd.DataFrame()

            # Combine calls and puts
            calls = options.calls
            calls['type'] = 'call'
            puts = options.puts
            puts['type'] = 'put'

            return pd.concat([calls, puts])

        except Exception as e:
            print(f"Error fetching options data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_returns(
        self,
        price_data: pd.DataFrame,
        method: str = 'simple'
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.

        Args:
            price_data: DataFrame with price data
            method: 'simple' or 'log' returns

        Returns:
            DataFrame with returns
        """
        if 'close' not in price_data.columns:
            raise ValueError("Price data must have 'close' column")

        if method == 'simple':
            returns = price_data['close'].pct_change()
        elif method == 'log':
            returns = np.log(price_data['close'] / price_data['close'].shift(1))
        else:
            raise ValueError(f"Unknown return calculation method: {method}")

        return returns

    def calculate_volatility(
        self,
        returns: pd.Series,
        window: int = 30,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            returns: Series of returns
            window: Rolling window size
            annualize: Whether to annualize volatility

        Returns:
            Series of volatility values
        """
        volatility = returns.rolling(window).std()

        if annualize:
            # Assume 252 trading days
            volatility = volatility * np.sqrt(252)

        return volatility

    def _get_cache_key(
        self,
        symbols: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        frequency: DataFrequency
    ) -> str:
        """Generate cache key for data request."""
        key_parts = [
            ','.join(sorted(symbols)),
            str(start_date.date()) if start_date else 'None',
            str(end_date.date()) if end_date else 'None',
            frequency.value
        ]
        return '_'.join(key_parts)

    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve data from cache."""
        cache_file = Path(self.config.cache_dir) / f"{cache_key}.parquet"

        if cache_file.exists():
            # Check if cache is still valid
            file_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if file_age < self.config.cache_expiry:
                try:
                    return pd.read_parquet(cache_file)
                except Exception:
                    pass

        return None

    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        if data.empty:
            return

        cache_file = Path(self.config.cache_dir) / f"{cache_key}.parquet"

        try:
            data.to_parquet(cache_file)
        except Exception as e:
            print(f"Error caching data: {e}")

    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate market data for quality issues.

        Args:
            data: DataFrame with market data

        Returns:
            Dictionary with validation results
        """
        issues = {
            'missing_values': {},
            'outliers': {},
            'data_gaps': [],
            'suspicious_values': []
        }

        # Check for missing values
        missing_counts = data.isnull().sum()
        issues['missing_values'] = missing_counts[missing_counts > 0].to_dict()

        # Check for outliers (using IQR method)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)]
            if not outliers.empty:
                issues['outliers'][col] = len(outliers)

        # Check for data gaps (assuming daily data)
        if not data.empty and isinstance(data.index, pd.DatetimeIndex):
            expected_days = pd.bdate_range(start=data.index.min(), end=data.index.max())
            missing_days = expected_days.difference(data.index)
            if len(missing_days) > 0:
                issues['data_gaps'] = [str(d.date()) for d in missing_days[:10]]  # Limit to 10

        return issues