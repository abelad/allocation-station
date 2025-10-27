"""REST API for Allocation Station framework."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn


class PortfolioRequest(BaseModel):
    """Portfolio analysis request."""
    holdings: Dict[str, float]
    initial_value: float


class PortfolioResponse(BaseModel):
    """Portfolio analysis response."""
    total_value: float
    allocation: Dict[str, float]
    metrics: Dict[str, float]


class AllocationStationAPI:
    """REST API for the framework."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """Initialize API."""
        self.app = FastAPI(title="Allocation Station API", version="1.0.0")
        self.host = host
        self.port = port
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            return {"message": "Allocation Station API", "version": "1.0.0"}

        @self.app.post("/analyze", response_model=PortfolioResponse)
        async def analyze_portfolio(request: PortfolioRequest):
            """Analyze portfolio."""
            total_value = sum(request.holdings.values())

            allocation = {
                symbol: (value / total_value * 100)
                for symbol, value in request.holdings.items()
            }

            metrics = {
                "total_value": total_value,
                "num_holdings": len(request.holdings),
                "concentration": max(allocation.values()) if allocation else 0
            }

            return PortfolioResponse(
                total_value=total_value,
                allocation=allocation,
                metrics=metrics
            )

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}

    def run(self):
        """Run the API server."""
        uvicorn.run(self.app, host=self.host, port=self.port)
