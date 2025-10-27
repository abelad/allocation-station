"""Export & Compatibility Examples - FIX, QuantLib, CLI, GraphQL, Multi-language."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from allocation_station.export import (
    FIXProtocolHandler, QuantLibIntegration, AllocationStationCLI, GraphQLAPI
)


def example_fix_protocol():
    """Example: FIX protocol for trading."""
    print("\n" + "=" * 80)
    print(" FIX Protocol Integration")
    print("=" * 80 + "\n")

    fix = FIXProtocolHandler(sender_comp_id="ALLOC_STATION", target_comp_id="BROKER")

    # Create logon message
    logon = fix.create_logon_message()
    print(f"Logon Message:\n{logon}\n")

    # Create new order
    order = fix.create_new_order(symbol="SPY", side="1", quantity=100, price=440.00)
    print(f"New Order Message:\n{order}\n")


def example_quantlib():
    """Example: QuantLib integration."""
    print("\n" + "=" * 80)
    print(" QuantLib Integration")
    print("=" * 80 + "\n")

    ql = QuantLibIntegration()

    # Price a bond
    bond_price = ql.price_bond(
        face_value=1000,
        coupon_rate=0.05,
        maturity_years=10,
        yield_rate=0.04
    )
    print(f"Bond Price: ${bond_price:.2f}\n")

    # Black-Scholes option pricing
    option = ql.black_scholes_option(
        spot=442.15,
        strike=440.00,
        rate=0.05,
        volatility=0.20,
        time=1.0,
        option_type='call'
    )
    print("Call Option:")
    print(f"  Price: ${option['price']:.2f}")
    print(f"  Delta: {option['delta']:.4f}")
    print(f"  Gamma: {option['gamma']:.4f}")
    print(f"  Vega: {option['vega']:.4f}\n")


def example_cli():
    """Example: Command-line interface."""
    print("\n" + "=" * 80)
    print(" Command-Line Interface")
    print("=" * 80 + "\n")

    print("CLI Commands:")
    print("  allocation-station analyze portfolio.json")
    print("  allocation-station serve --port 8000")
    print("  allocation-station report portfolio.json --format pdf -o report.pdf")
    print("  allocation-station version\n")


def example_graphql():
    """Example: GraphQL API."""
    print("\n" + "=" * 80)
    print(" GraphQL API")
    print("=" * 80 + "\n")

    api = GraphQLAPI()
    schema = api.get_schema()

    print("GraphQL Schema initialized")
    print("\nExample Queries:")
    print("""
    query {
      portfolio(portfolioId: "123") {
        name
        totalValue
        positions {
          symbol
          quantity
          value
        }
      }
    }
    """)


def example_multi_language():
    """Example: Multi-language bindings."""
    print("\n" + "=" * 80)
    print(" Multi-Language Bindings")
    print("=" * 80 + "\n")

    print("R Interface:")
    print("  library(allocationstation)")
    print("  results <- analyze_portfolio(holdings)\n")

    print("MATLAB Interface:")
    print("  as = AllocationStation()")
    print("  results = as.analyzePortfolio(holdings)\n")

    print("Julia Interface:")
    print("  using AllocationStation")
    print("  results = analyze_portfolio(holdings)\n")


def main():
    """Run all export/compatibility examples."""
    print("\nALLOCATION STATION - EXPORT & COMPATIBILITY EXAMPLES\n")

    example_fix_protocol()
    example_quantlib()
    example_cli()
    example_graphql()
    example_multi_language()

    print("\n" + "=" * 80)
    print(" Examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
