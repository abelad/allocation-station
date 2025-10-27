"""Portfolio Import Module - Import portfolios from various brokers and formats."""

from typing import Dict, List
import pandas as pd
from .broker_api import BrokerConnection, Position


class PortfolioImporter:
    """Import portfolios from brokers and file formats."""

    def import_from_broker(self, broker: BrokerConnection) -> pd.DataFrame:
        """Import portfolio from broker connection."""
        positions = broker.get_positions()

        data = [{
            'symbol': p.symbol,
            'quantity': p.quantity,
            'avg_cost': p.avg_cost,
            'current_price': p.current_price,
            'market_value': p.market_value,
            'unrealized_pnl': p.unrealized_pnl
        } for p in positions]

        return pd.DataFrame(data)

    def import_from_csv(self, filepath: str) -> pd.DataFrame:
        """Import portfolio from CSV file."""
        return pd.read_csv(filepath)

    def import_from_excel(self, filepath: str, sheet_name: str = 'Sheet1') -> pd.DataFrame:
        """Import portfolio from Excel file."""
        return pd.read_excel(filepath, sheet_name=sheet_name)
