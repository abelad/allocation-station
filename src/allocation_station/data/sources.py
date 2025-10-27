"""Data source implementations for various market data providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel
from pathlib import Path

try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

try:
    import nasdaqdatalink
    NASDAQ_DATALINK_AVAILABLE = True
except ImportError:
    NASDAQ_DATALINK_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

import yfinance as yf


class DataSourceInterface(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def fetch_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """Fetch historical price data."""
        pass

    @abstractmethod
    def fetch_real_time_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """Fetch real-time market data."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is available and configured."""
        pass


class AlphaVantageSource(DataSourceInterface):
    """Alpha Vantage data source implementation."""

    def __init__(self, api_key: str):
        """
        Initialize Alpha Vantage data source.

        Args:
            api_key: Alpha Vantage API key
        """
        if not ALPHA_VANTAGE_AVAILABLE:
            raise ImportError(
                "alpha-vantage package is not installed. "
                "Install it with: pip install alpha-vantage"
            )

        self.api_key = api_key
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.fd = FundamentalData(key=api_key, output_format='pandas')

    def fetch_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """
        Fetch historical data from Alpha Vantage.

        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency ('daily', 'weekly', 'monthly', 'intraday')

        Returns:
            DataFrame with historical data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        all_data = []

        for symbol in symbols:
            try:
                if frequency == "daily":
                    data, meta = self.ts.get_daily_adjusted(symbol=symbol, outputsize='full')
                elif frequency == "weekly":
                    data, meta = self.ts.get_weekly_adjusted(symbol=symbol)
                elif frequency == "monthly":
                    data, meta = self.ts.get_monthly_adjusted(symbol=symbol)
                elif frequency == "intraday":
                    data, meta = self.ts.get_intraday(
                        symbol=symbol,
                        interval='5min',
                        outputsize='full'
                    )
                else:
                    raise ValueError(f"Unknown frequency: {frequency}")

                # Standardize column names
                data.columns = [col.split('. ')[1].lower() for col in data.columns]

                # Filter by date range if provided
                if start_date:
                    data = data[data.index >= start_date]
                if end_date:
                    data = data[data.index <= end_date]

                # Add symbol column
                data['symbol'] = symbol
                data = data.reset_index()
                data.columns = ['date' if col == 'index' else col for col in data.columns]

                all_data.append(data)

            except Exception as e:
                print(f"Error fetching Alpha Vantage data for {symbol}: {e}")
                continue

        if all_data:
            return pd.concat(all_data, ignore_index=True)

        return pd.DataFrame()

    def fetch_real_time_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Fetch real-time quote data.

        Args:
            symbols: Single symbol or list of symbols

        Returns:
            Dictionary with real-time data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        real_time_data = {}

        for symbol in symbols:
            try:
                data, meta = self.ts.get_quote_endpoint(symbol=symbol)

                real_time_data[symbol] = {
                    'price': float(data['05. price'][0]),
                    'volume': float(data['06. volume'][0]),
                    'previous_close': float(data['08. previous close'][0]),
                    'change': float(data['09. change'][0]),
                    'change_percent': float(data['10. change percent'][0].rstrip('%'))
                }

            except Exception as e:
                print(f"Error fetching Alpha Vantage real-time data for {symbol}: {e}")
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
            overview, _ = self.fd.get_company_overview(symbol=symbol)

            return {
                'name': overview.get('Name', [''])[0],
                'sector': overview.get('Sector', [''])[0],
                'industry': overview.get('Industry', [''])[0],
                'market_cap': float(overview.get('MarketCapitalization', [0])[0]),
                'pe_ratio': float(overview.get('PERatio', [0])[0]) if overview.get('PERatio', [0])[0] != 'None' else None,
                'pb_ratio': float(overview.get('PriceToBookRatio', [0])[0]) if overview.get('PriceToBookRatio', [0])[0] != 'None' else None,
                'dividend_yield': float(overview.get('DividendYield', [0])[0]) if overview.get('DividendYield', [0])[0] != 'None' else None,
                'eps': float(overview.get('EPS', [0])[0]) if overview.get('EPS', [0])[0] != 'None' else None,
                'beta': float(overview.get('Beta', [0])[0]) if overview.get('Beta', [0])[0] != 'None' else None,
            }

        except Exception as e:
            print(f"Error fetching Alpha Vantage fundamental data for {symbol}: {e}")
            return {}

    def is_available(self) -> bool:
        """Check if Alpha Vantage is available."""
        return ALPHA_VANTAGE_AVAILABLE and self.api_key is not None


class NasdaqDataLinkSource(DataSourceInterface):
    """NASDAQ Data Link (formerly Quandl) data source implementation."""

    def __init__(self, api_key: str):
        """
        Initialize NASDAQ Data Link source.

        Args:
            api_key: NASDAQ Data Link API key
        """
        if not NASDAQ_DATALINK_AVAILABLE:
            raise ImportError(
                "nasdaq-data-link package is not installed. "
                "Install it with: pip install nasdaq-data-link"
            )

        self.api_key = api_key
        nasdaqdatalink.ApiConfig.api_key = api_key

    def fetch_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """
        Fetch historical data from NASDAQ Data Link.

        Args:
            symbols: Database/dataset codes (e.g., 'WIKI/AAPL')
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency

        Returns:
            DataFrame with historical data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        all_data = []

        for symbol in symbols:
            try:
                # Fetch data
                data = nasdaqdatalink.get(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                    collapse=self._map_frequency(frequency)
                )

                # Standardize column names
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]

                # Add symbol column
                data['symbol'] = symbol
                data = data.reset_index()

                all_data.append(data)

            except Exception as e:
                print(f"Error fetching NASDAQ Data Link data for {symbol}: {e}")
                continue

        if all_data:
            return pd.concat(all_data, ignore_index=True)

        return pd.DataFrame()

    def fetch_real_time_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Fetch most recent data point (NASDAQ Data Link doesn't provide real-time quotes).

        Args:
            symbols: Database/dataset codes

        Returns:
            Dictionary with latest data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        real_time_data = {}

        for symbol in symbols:
            try:
                # Get the most recent row
                data = nasdaqdatalink.get(symbol, rows=1)

                if not data.empty:
                    row = data.iloc[-1]
                    real_time_data[symbol] = {
                        'price': row.get('Close', row.get('close', 0)),
                        'volume': row.get('Volume', row.get('volume', 0)),
                        'date': str(data.index[-1].date())
                    }

            except Exception as e:
                print(f"Error fetching NASDAQ Data Link latest data for {symbol}: {e}")
                real_time_data[symbol] = {}

        return real_time_data

    def _map_frequency(self, frequency: str) -> Optional[str]:
        """Map frequency to NASDAQ Data Link format."""
        freq_map = {
            'daily': 'daily',
            'weekly': 'weekly',
            'monthly': 'monthly',
            'quarterly': 'quarterly',
            'annual': 'annual'
        }
        return freq_map.get(frequency.lower())

    def is_available(self) -> bool:
        """Check if NASDAQ Data Link is available."""
        return NASDAQ_DATALINK_AVAILABLE and self.api_key is not None


class FREDSource(DataSourceInterface):
    """Federal Reserve Economic Data (FRED) source implementation."""

    def __init__(self, api_key: str):
        """
        Initialize FRED data source.

        Args:
            api_key: FRED API key
        """
        if not FRED_AVAILABLE:
            raise ImportError(
                "fredapi package is not installed. "
                "Install it with: pip install fredapi"
            )

        self.api_key = api_key
        self.fred = Fred(api_key=api_key)

    def fetch_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """
        Fetch economic data from FRED.

        Args:
            symbols: FRED series IDs (e.g., 'GDP', 'UNRATE', 'DGS10')
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency

        Returns:
            DataFrame with economic data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        all_data = []

        for symbol in symbols:
            try:
                # Fetch series data
                data = self.fred.get_series(
                    symbol,
                    observation_start=start_date,
                    observation_end=end_date
                )

                # Convert to DataFrame
                df = pd.DataFrame({
                    'date': data.index,
                    'value': data.values,
                    'symbol': symbol
                })

                all_data.append(df)

            except Exception as e:
                print(f"Error fetching FRED data for {symbol}: {e}")
                continue

        if all_data:
            return pd.concat(all_data, ignore_index=True)

        return pd.DataFrame()

    def fetch_real_time_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Fetch latest economic data.

        Args:
            symbols: FRED series IDs

        Returns:
            Dictionary with latest data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        real_time_data = {}

        for symbol in symbols:
            try:
                # Get the most recent observation
                data = self.fred.get_series(symbol, limit=1)

                if not data.empty:
                    real_time_data[symbol] = {
                        'value': float(data.iloc[-1]),
                        'date': str(data.index[-1].date())
                    }

            except Exception as e:
                print(f"Error fetching FRED latest data for {symbol}: {e}")
                real_time_data[symbol] = {}

        return real_time_data

    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """
        Get information about a FRED series.

        Args:
            series_id: FRED series ID

        Returns:
            Dictionary with series information
        """
        try:
            info = self.fred.get_series_info(series_id)
            return {
                'id': info.get('id'),
                'title': info.get('title'),
                'units': info.get('units'),
                'frequency': info.get('frequency'),
                'seasonal_adjustment': info.get('seasonal_adjustment'),
                'last_updated': info.get('last_updated'),
            }
        except Exception as e:
            print(f"Error fetching FRED series info for {series_id}: {e}")
            return {}

    def search_series(self, search_text: str, limit: int = 10) -> pd.DataFrame:
        """
        Search for FRED series by keyword.

        Args:
            search_text: Search query
            limit: Maximum number of results

        Returns:
            DataFrame with search results
        """
        try:
            results = self.fred.search(search_text, limit=limit)
            return results
        except Exception as e:
            print(f"Error searching FRED for '{search_text}': {e}")
            return pd.DataFrame()

    def is_available(self) -> bool:
        """Check if FRED is available."""
        return FRED_AVAILABLE and self.api_key is not None


class YahooFinanceSource(DataSourceInterface):
    """Yahoo Finance data source implementation (wrapper for existing functionality)."""

    def __init__(self):
        """Initialize Yahoo Finance source."""
        pass

    def fetch_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.

        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            frequency: Data frequency

        Returns:
            DataFrame with historical data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Set default date range
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=365*5))

        # Map frequency
        freq_map = {
            'intraday_1m': '1m',
            'intraday_5m': '5m',
            'intraday_15m': '15m',
            'intraday_30m': '30m',
            'hourly': '1h',
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo'
        }
        interval = freq_map.get(frequency, '1d')

        all_data = []

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
                    df['symbol'] = symbol
                    df.columns = [col.lower() for col in df.columns]
                    df = df.reset_index()
                    all_data.append(df)

            except Exception as e:
                print(f"Error fetching Yahoo Finance data for {symbol}: {e}")
                continue

        if all_data:
            return pd.concat(all_data, ignore_index=True)

        return pd.DataFrame()

    def fetch_real_time_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Fetch real-time market data.

        Args:
            symbols: Single symbol or list of symbols

        Returns:
            Dictionary with real-time data
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
                }

            except Exception as e:
                print(f"Error fetching Yahoo Finance real-time data for {symbol}: {e}")
                real_time_data[symbol] = {}

        return real_time_data

    def is_available(self) -> bool:
        """Check if Yahoo Finance is available."""
        return True
