"""External Integrations Examples - Broker APIs, REST API, Cloud Storage, Databases."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from allocation_station.integrations import (
    IBKRClient, TDAClient, PortfolioImporter, AllocationStationAPI,
    WebhookManager, CloudStorageManager, DatabaseConnector,
    BrokerType, Order, OrderAction, OrderType
)
from datetime import datetime, timedelta


def example_ibkr_integration():
    """Example: Interactive Brokers integration."""
    print("\n" + "=" * 80)
    print(" Interactive Brokers (IBKR) Integration")
    print("=" * 80 + "\n")

    # Create IBKR client
    ibkr = IBKRClient(host='127.0.0.1', port=7497)

    # Connect
    if ibkr.connect():
        # Get positions
        positions = ibkr.get_positions()
        print(f"Current Positions: {len(positions)}")
        for pos in positions:
            print(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.current_price:.2f}")

        # Get account value
        account_value = ibkr.get_account_value()
        print(f"\nTotal Account Value: ${account_value:,.2f}")

        # Place order
        order = Order(
            symbol="SPY",
            action=OrderAction.BUY,
            order_type=OrderType.LIMIT,
            quantity=10,
            limit_price=440.00
        )
        order_id = ibkr.place_order(order)
        print(f"\nOrder ID: {order_id}")

        # Get options chain
        options = ibkr.get_option_chain("SPY")
        print(f"\nOptions Chain: {len(options)} strikes")

        ibkr.disconnect()


def example_rest_api():
    """Example: REST API server."""
    print("\n" + "=" * 80)
    print(" REST API Server")
    print("=" * 80 + "\n")

    print("Starting Allocation Station REST API...")
    print("Endpoints:")
    print("  GET  /           - API info")
    print("  POST /analyze    - Analyze portfolio")
    print("  GET  /health     - Health check")
    print("\nTo start server: api = AllocationStationAPI(); api.run()")


def example_cloud_storage():
    """Example: Cloud storage integration."""
    print("\n" + "=" * 80)
    print(" Cloud Storage Integration")
    print("=" * 80 + "\n")

    storage = CloudStorageManager(provider='aws')
    storage.connect({'access_key': 'xxx', 'secret_key': 'yyy'})

    # Upload portfolio data
    storage.upload_file('portfolio.json', 's3://bucket/portfolio.json')

    # List files
    files = storage.list_files('reports/')
    print(f"Files in cloud: {files}")


def main():
    """Run all integration examples."""
    print("\nALLOCATION STATION - EXTERNAL INTEGRATIONS EXAMPLES\n")

    example_ibkr_integration()
    example_rest_api()
    example_cloud_storage()

    print("\n" + "=" * 80)
    print(" Examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
