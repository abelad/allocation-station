"""GraphQL API for Allocation Station."""

from typing import List, Optional
import strawberry
from datetime import datetime


@strawberry.type
class Position:
    """GraphQL Position type."""
    symbol: str
    quantity: float
    price: float
    value: float


@strawberry.type
class Portfolio:
    """GraphQL Portfolio type."""
    id: str
    name: str
    total_value: float
    positions: List[Position]
    created_at: datetime


@strawberry.type
class PerformanceMetrics:
    """GraphQL Performance Metrics type."""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float


@strawberry.type
class Query:
    """GraphQL Query type."""

    @strawberry.field
    def portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Get portfolio by ID."""
        # Mock data
        return Portfolio(
            id=portfolio_id,
            name="Sample Portfolio",
            total_value=1000000.0,
            positions=[
                Position(symbol="SPY", quantity=100, price=442.15, value=44215),
                Position(symbol="TLT", quantity=150, price=92.30, value=13845)
            ],
            created_at=datetime.now()
        )

    @strawberry.field
    def portfolios(self) -> List[Portfolio]:
        """Get all portfolios."""
        return []

    @strawberry.field
    def performance(self, portfolio_id: str) -> PerformanceMetrics:
        """Get performance metrics."""
        return PerformanceMetrics(
            total_return=18.5,
            annual_return=15.2,
            volatility=12.3,
            sharpe_ratio=1.45,
            max_drawdown=-8.3
        )


@strawberry.type
class Mutation:
    """GraphQL Mutation type."""

    @strawberry.mutation
    def create_portfolio(self, name: str) -> Portfolio:
        """Create a new portfolio."""
        return Portfolio(
            id="new_portfolio",
            name=name,
            total_value=0.0,
            positions=[],
            created_at=datetime.now()
        )


class GraphQLAPI:
    """GraphQL API wrapper."""

    def __init__(self):
        """Initialize GraphQL API."""
        self.schema = strawberry.Schema(query=Query, mutation=Mutation)

    def get_schema(self):
        """Get GraphQL schema."""
        return self.schema
