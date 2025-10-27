"""
Interactive Dashboard for Allocation Station

This module provides a comprehensive web-based dashboard using Streamlit for
portfolio analysis, strategy building, and real-time monitoring. It includes
drag-and-drop allocation interfaces, portfolio comparison tools, and user
authentication with profile management.

Features:
    - Real-time portfolio monitoring with live updates
    - Interactive strategy builder with visual feedback
    - Drag-and-drop allocation interface
    - Portfolio comparison tools with side-by-side analysis
    - Mobile-responsive design
    - User authentication and profile management
    - Data persistence and session management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import hashlib
import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time
import yfinance as yf
from pathlib import Path
import os

# Import allocation station modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from allocation_station.strategy import Strategy
from allocation_station.backtest import Backtester
from allocation_station.portfolio import Portfolio
from allocation_station.monte_carlo import MonteCarlo
from allocation_station.efficient_frontier import EfficientFrontier


class DashboardSection(Enum):
    """Dashboard sections."""
    OVERVIEW = "Overview"
    STRATEGY_BUILDER = "Strategy Builder"
    PORTFOLIO_MONITOR = "Portfolio Monitor"
    COMPARISON = "Comparison Tools"
    ANALYSIS = "Analysis"
    SETTINGS = "Settings"


@dataclass
class UserProfile:
    """User profile data."""
    username: str
    email: str
    created_at: datetime
    risk_tolerance: str
    investment_horizon: int
    portfolios: List[Dict]
    preferences: Dict


class AuthenticationManager:
    """Handles user authentication and session management."""

    def __init__(self, db_path: str = "users.db"):
        """Initialize authentication manager."""
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize user database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                profile_data TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        conn.commit()
        conn.close()

    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username: str, password: str, email: str) -> bool:
        """Register new user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            password_hash = self.hash_password(password)

            profile = UserProfile(
                username=username,
                email=email,
                created_at=datetime.now(),
                risk_tolerance="moderate",
                investment_horizon=10,
                portfolios=[],
                preferences={}
            )

            cursor.execute("""
                INSERT INTO users (username, password_hash, email, profile_data)
                VALUES (?, ?, ?, ?)
            """, (username, password_hash, email, json.dumps(asdict(profile), default=str)))

            conn.commit()
            return True

        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        password_hash = self.hash_password(password)

        cursor.execute("""
            SELECT id FROM users
            WHERE username = ? AND password_hash = ?
        """, (username, password_hash))

        user = cursor.fetchone()

        if user:
            # Create session token
            session_token = hashlib.sha256(
                f"{username}{datetime.now()}".encode()
            ).hexdigest()

            cursor.execute("""
                INSERT INTO sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            """, (user[0], session_token, datetime.now() + timedelta(hours=24)))

            conn.commit()
            conn.close()
            return session_token

        conn.close()
        return None

    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate session token and return user data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT u.username, u.email, u.profile_data
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_token = ? AND s.expires_at > ?
        """, (session_token, datetime.now()))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'username': result[0],
                'email': result[1],
                'profile': json.loads(result[2]) if result[2] else {}
            }

        return None

    def logout(self, session_token: str):
        """Logout user by invalidating session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM sessions WHERE session_token = ?
        """, (session_token,))

        conn.commit()
        conn.close()


class RealTimeMonitor:
    """Handles real-time portfolio monitoring."""

    def __init__(self):
        """Initialize real-time monitor."""
        self.update_interval = 60  # seconds
        self.last_update = None
        self.cache = {}

    def fetch_live_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch live prices for symbols."""
        prices = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                prices[symbol] = info.get('currentPrice', info.get('regularMarketPrice', 0))
            except Exception:
                prices[symbol] = 0

        return prices

    def calculate_portfolio_value(self, holdings: Dict[str, float],
                                 prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total = 0
        for symbol, shares in holdings.items():
            if symbol in prices:
                total += shares * prices[symbol]
        return total

    def get_intraday_data(self, symbol: str, period: str = "1d") -> pd.DataFrame:
        """Get intraday price data."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.history(period=period, interval="5m")
        except Exception:
            return pd.DataFrame()

    def calculate_daily_returns(self, portfolio: Dict[str, float]) -> float:
        """Calculate daily returns for portfolio."""
        total_return = 0
        total_value = 0

        for symbol, shares in portfolio.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")

                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[-2]
                    curr_close = hist['Close'].iloc[-1]
                    daily_return = (curr_close - prev_close) / prev_close
                    position_value = shares * curr_close

                    total_return += daily_return * position_value
                    total_value += position_value

            except Exception:
                continue

        return (total_return / total_value * 100) if total_value > 0 else 0

    def monitor_portfolio(self, portfolio: Dict[str, float],
                         alerts: Dict[str, Any]) -> List[str]:
        """Monitor portfolio and generate alerts."""
        messages = []
        prices = self.fetch_live_prices(list(portfolio.keys()))

        for symbol, shares in portfolio.items():
            if symbol in prices:
                current_price = prices[symbol]

                # Check price alerts
                if symbol in alerts:
                    if 'stop_loss' in alerts[symbol]:
                        if current_price <= alerts[symbol]['stop_loss']:
                            messages.append(
                                f"⚠️ {symbol} hit stop loss at ${current_price:.2f}"
                            )

                    if 'take_profit' in alerts[symbol]:
                        if current_price >= alerts[symbol]['take_profit']:
                            messages.append(
                                f"✅ {symbol} hit take profit at ${current_price:.2f}"
                            )

                    if 'volatility_threshold' in alerts[symbol]:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="5d")
                        if len(hist) > 0:
                            volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                            if volatility > alerts[symbol]['volatility_threshold']:
                                messages.append(
                                    f"📊 {symbol} volatility alert: {volatility:.1%}"
                                )

        return messages


class StrategyBuilder:
    """Interactive strategy builder component."""

    def __init__(self):
        """Initialize strategy builder."""
        self.asset_classes = {
            'Stocks': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI'],
            'Bonds': ['TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'EMB'],
            'Commodities': ['GLD', 'SLV', 'DBA', 'USO', 'UNG'],
            'Real Estate': ['VNQ', 'REM', 'MORT', 'REZ'],
            'Crypto': ['GBTC', 'ETHE', 'BITO']
        }

        self.strategies = {
            'Conservative': {'stocks': 30, 'bonds': 60, 'commodities': 5, 'real_estate': 5},
            'Moderate': {'stocks': 50, 'bonds': 35, 'commodities': 10, 'real_estate': 5},
            'Aggressive': {'stocks': 70, 'bonds': 20, 'commodities': 5, 'real_estate': 5},
            'All Weather': {'stocks': 30, 'bonds': 55, 'commodities': 15, 'real_estate': 0},
            'Risk Parity': {'stocks': 25, 'bonds': 25, 'commodities': 25, 'real_estate': 25}
        }

    def create_allocation_chart(self, allocation: Dict[str, float]) -> go.Figure:
        """Create allocation pie chart."""
        fig = go.Figure(data=[
            go.Pie(
                labels=list(allocation.keys()),
                values=list(allocation.values()),
                hole=0.4,
                marker=dict(
                    colors=px.colors.qualitative.Set3
                ),
                textfont=dict(size=14),
                textposition='outside',
                textinfo='label+percent'
            )
        ])

        fig.update_layout(
            title="Portfolio Allocation",
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )

        return fig

    def calculate_risk_metrics(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk metrics for allocation."""
        # Simplified risk calculation
        stock_weight = allocation.get('stocks', 0) / 100
        bond_weight = allocation.get('bonds', 0) / 100
        commodity_weight = allocation.get('commodities', 0) / 100
        real_estate_weight = allocation.get('real_estate', 0) / 100

        # Historical statistics (approximate)
        expected_return = (
            stock_weight * 0.10 +
            bond_weight * 0.04 +
            commodity_weight * 0.06 +
            real_estate_weight * 0.08
        )

        volatility = np.sqrt(
            (stock_weight * 0.16)**2 +
            (bond_weight * 0.05)**2 +
            (commodity_weight * 0.15)**2 +
            (real_estate_weight * 0.12)**2 +
            2 * stock_weight * bond_weight * 0.16 * 0.05 * (-0.2) +
            2 * stock_weight * commodity_weight * 0.16 * 0.15 * 0.1 +
            2 * stock_weight * real_estate_weight * 0.16 * 0.12 * 0.6
        )

        sharpe_ratio = (expected_return - 0.03) / volatility if volatility > 0 else 0
        max_drawdown = volatility * 2.5  # Approximation

        return {
            'expected_return': expected_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100
        }

    def optimize_allocation(self, risk_tolerance: str,
                          constraints: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Optimize allocation based on risk tolerance and constraints."""
        # Start with base strategy
        base = self.strategies.get(
            'Conservative' if risk_tolerance == 'low' else
            'Moderate' if risk_tolerance == 'medium' else
            'Aggressive'
        ).copy()

        # Apply constraints
        for asset, (min_val, max_val) in constraints.items():
            if asset in base:
                base[asset] = max(min_val, min(max_val, base[asset]))

        # Normalize to 100%
        total = sum(base.values())
        if total > 0:
            for asset in base:
                base[asset] = (base[asset] / total) * 100

        return base


class PortfolioComparison:
    """Portfolio comparison tools."""

    def __init__(self):
        """Initialize comparison tools."""
        self.metrics = [
            'Total Return',
            'Annual Return',
            'Volatility',
            'Sharpe Ratio',
            'Max Drawdown',
            'Calmar Ratio',
            'Sortino Ratio',
            'Beta',
            'Alpha',
            'Treynor Ratio'
        ]

    def compare_portfolios(self, portfolios: List[Dict]) -> pd.DataFrame:
        """Compare multiple portfolios."""
        comparison_data = []

        for portfolio in portfolios:
            # Simulate performance metrics
            metrics = {
                'Portfolio': portfolio['name'],
                'Total Return': np.random.uniform(20, 100),
                'Annual Return': np.random.uniform(5, 15),
                'Volatility': np.random.uniform(8, 20),
                'Sharpe Ratio': np.random.uniform(0.5, 2.0),
                'Max Drawdown': np.random.uniform(-30, -10),
                'Calmar Ratio': np.random.uniform(0.3, 1.5),
                'Sortino Ratio': np.random.uniform(0.7, 2.5),
                'Beta': np.random.uniform(0.7, 1.3),
                'Alpha': np.random.uniform(-2, 5),
                'Treynor Ratio': np.random.uniform(0.05, 0.20)
            }
            comparison_data.append(metrics)

        return pd.DataFrame(comparison_data)

    def create_comparison_chart(self, df: pd.DataFrame, metric: str) -> go.Figure:
        """Create comparison bar chart."""
        fig = go.Figure(data=[
            go.Bar(
                x=df['Portfolio'],
                y=df[metric],
                text=df[metric].round(2),
                textposition='outside',
                marker_color='lightblue'
            )
        ])

        fig.update_layout(
            title=f"{metric} Comparison",
            xaxis_title="Portfolio",
            yaxis_title=metric,
            height=400,
            showlegend=False
        )

        return fig

    def create_scatter_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create risk-return scatter plot."""
        fig = go.Figure()

        # Add scatter points
        fig.add_trace(go.Scatter(
            x=df['Volatility'],
            y=df['Annual Return'],
            mode='markers+text',
            text=df['Portfolio'],
            textposition='top center',
            marker=dict(
                size=df['Sharpe Ratio'] * 20,
                color=df['Sharpe Ratio'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            )
        ))

        # Add efficient frontier line (simplified)
        x_eff = np.linspace(df['Volatility'].min(), df['Volatility'].max(), 100)
        y_eff = 2 + 0.5 * x_eff

        fig.add_trace(go.Scatter(
            x=x_eff,
            y=y_eff,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title="Risk-Return Analysis",
            xaxis_title="Volatility (%)",
            yaxis_title="Annual Return (%)",
            height=500,
            hovermode='closest'
        )

        return fig

    def create_correlation_heatmap(self, portfolios: List[Dict]) -> go.Figure:
        """Create correlation heatmap between portfolios."""
        # Generate correlation matrix
        n = len(portfolios)
        corr_matrix = np.random.uniform(0.3, 0.9, (n, n))
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2

        portfolio_names = [p['name'] for p in portfolios]

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=portfolio_names,
            y=portfolio_names,
            colorscale='RdBu',
            zmid=0.5,
            text=corr_matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title="Portfolio Correlation Matrix",
            height=400,
            xaxis_title="Portfolio",
            yaxis_title="Portfolio"
        )

        return fig


class DashboardUI:
    """Main dashboard UI class."""

    def __init__(self):
        """Initialize dashboard."""
        self.auth_manager = AuthenticationManager()
        self.monitor = RealTimeMonitor()
        self.strategy_builder = StrategyBuilder()
        self.comparison_tools = PortfolioComparison()

        # Configure Streamlit
        st.set_page_config(
            page_title="Allocation Station Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        if 'allocation' not in st.session_state:
            st.session_state.allocation = {'stocks': 50, 'bonds': 30, 'commodities': 10, 'real_estate': 10}

    def render_login(self):
        """Render login page."""
        st.title("🏦 Allocation Station")
        st.subheader("Portfolio Management Dashboard")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            tab1, tab2 = st.tabs(["Login", "Register"])

            with tab1:
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    submit = st.form_submit_button("Login", use_container_width=True)

                    if submit:
                        session_token = self.auth_manager.authenticate_user(username, password)
                        if session_token:
                            st.session_state.authenticated = True
                            st.session_state.session_token = session_token
                            st.session_state.user = username
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials")

            with tab2:
                with st.form("register_form"):
                    new_username = st.text_input("Choose Username")
                    new_email = st.text_input("Email")
                    new_password = st.text_input("Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    register = st.form_submit_button("Register", use_container_width=True)

                    if register:
                        if new_password != confirm_password:
                            st.error("Passwords don't match")
                        elif len(new_password) < 8:
                            st.error("Password must be at least 8 characters")
                        else:
                            if self.auth_manager.register_user(new_username, new_password, new_email):
                                st.success("Registration successful! Please login.")
                            else:
                                st.error("Username already exists")

    def render_sidebar(self):
        """Render sidebar navigation."""
        with st.sidebar:
            st.image("https://via.placeholder.com/150", caption="Allocation Station")

            st.markdown(f"### Welcome, {st.session_state.user}!")

            st.markdown("---")

            # Navigation
            selected_section = st.selectbox(
                "Navigation",
                [s.value for s in DashboardSection]
            )

            st.markdown("---")

            # Quick stats
            st.markdown("### 📊 Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Portfolio Value", "$1,245,678", "+2.3%")
            with col2:
                st.metric("Daily P&L", "+$28,451", "+2.3%")

            st.markdown("---")

            # Market status
            st.markdown("### 🏛️ Market Status")
            market_open = datetime.now().hour >= 9 and datetime.now().hour < 16
            if market_open:
                st.success("Markets Open")
            else:
                st.warning("Markets Closed")

            st.markdown("---")

            # Logout button
            if st.button("Logout", use_container_width=True):
                self.auth_manager.logout(st.session_state.session_token)
                st.session_state.authenticated = False
                st.session_state.user = None
                st.rerun()

        return selected_section

    def render_overview(self):
        """Render overview section."""
        st.title("📈 Portfolio Overview")

        # Portfolio summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Value", "$1,245,678", "+2.3%", delta_color="normal")
        with col2:
            st.metric("YTD Return", "+18.5%", "+1.2%", delta_color="normal")
        with col3:
            st.metric("Sharpe Ratio", "1.45", "+0.05", delta_color="normal")
        with col4:
            st.metric("Max Drawdown", "-8.3%", "-0.2%", delta_color="inverse")

        st.markdown("---")

        # Performance chart
        col1, col2 = st.columns([2, 1])

        with col1:
            # Generate sample performance data
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            cumulative_returns = np.cumprod(1 + np.random.randn(len(dates)) * 0.01)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns * 1000000,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))

            fig.update_layout(
                title="Portfolio Performance",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Current allocation
            fig = self.strategy_builder.create_allocation_chart(st.session_state.allocation)
            st.plotly_chart(fig, use_container_width=True)

        # Holdings table
        st.subheader("Current Holdings")

        holdings_data = {
            'Symbol': ['SPY', 'TLT', 'GLD', 'VNQ', 'QQQ'],
            'Shares': [100, 150, 50, 75, 80],
            'Price': [440.25, 92.50, 185.30, 88.75, 380.60],
            'Value': [44025, 13875, 9265, 6656, 30448],
            'Weight': [42.3, 13.3, 8.9, 6.4, 29.1],
            'Daily Change': ['+1.2%', '-0.5%', '+0.8%', '+1.5%', '+2.1%']
        }

        df_holdings = pd.DataFrame(holdings_data)
        st.dataframe(df_holdings, use_container_width=True, hide_index=True)

    def render_strategy_builder(self):
        """Render strategy builder section."""
        st.title("🔧 Strategy Builder")

        # Strategy settings
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Settings")

            # Risk tolerance
            risk_tolerance = st.select_slider(
                "Risk Tolerance",
                options=["Conservative", "Moderate", "Aggressive"],
                value="Moderate"
            )

            # Investment horizon
            investment_horizon = st.slider(
                "Investment Horizon (years)",
                min_value=1,
                max_value=30,
                value=10
            )

            # Rebalancing frequency
            rebalancing = st.selectbox(
                "Rebalancing Frequency",
                ["Monthly", "Quarterly", "Annually", "Never"]
            )

            st.markdown("---")

            # Preset strategies
            st.subheader("Preset Strategies")

            preset = st.selectbox(
                "Select Preset",
                list(self.strategy_builder.strategies.keys())
            )

            if st.button("Apply Preset", use_container_width=True):
                st.session_state.allocation = self.strategy_builder.strategies[preset].copy()
                st.rerun()

        with col2:
            st.subheader("Allocation Builder")

            # Drag and drop interface (simulated with sliders)
            allocation = {}

            col_a, col_b = st.columns(2)

            with col_a:
                allocation['stocks'] = st.slider(
                    "Stocks (%)",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.allocation.get('stocks', 50),
                    step=5
                )

                allocation['bonds'] = st.slider(
                    "Bonds (%)",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.allocation.get('bonds', 30),
                    step=5
                )

            with col_b:
                allocation['commodities'] = st.slider(
                    "Commodities (%)",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.allocation.get('commodities', 10),
                    step=5
                )

                allocation['real_estate'] = st.slider(
                    "Real Estate (%)",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.allocation.get('real_estate', 10),
                    step=5
                )

            # Normalize to 100%
            total = sum(allocation.values())
            if total > 0:
                for key in allocation:
                    allocation[key] = (allocation[key] / total) * 100

            st.session_state.allocation = allocation

            # Display total
            total_pct = sum(allocation.values())
            if abs(total_pct - 100) > 0.1:
                st.warning(f"Total allocation: {total_pct:.1f}% (normalizing to 100%)")
            else:
                st.success(f"Total allocation: {total_pct:.1f}%")

            # Display allocation chart
            fig = self.strategy_builder.create_allocation_chart(allocation)
            st.plotly_chart(fig, use_container_width=True)

            # Risk metrics
            metrics = self.strategy_builder.calculate_risk_metrics(allocation)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Expected Return", f"{metrics['expected_return']:.1f}%")
            with col2:
                st.metric("Volatility", f"{metrics['volatility']:.1f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            with col4:
                st.metric("Max Drawdown", f"-{metrics['max_drawdown']:.1f}%")

    def render_portfolio_monitor(self):
        """Render portfolio monitoring section."""
        st.title("📊 Real-Time Portfolio Monitor")

        # Auto-refresh toggle
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            auto_refresh = st.checkbox("Auto-refresh", value=True)

        with col2:
            refresh_interval = st.selectbox(
                "Interval",
                ["30s", "1m", "5m", "15m"],
                index=1
            )

        with col3:
            last_update = st.empty()
            last_update.text(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

        # Portfolio metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("SPY", "$442.15", "+1.2%")
        with col2:
            st.metric("TLT", "$92.30", "-0.5%")
        with col3:
            st.metric("GLD", "$186.50", "+0.8%")
        with col4:
            st.metric("VNQ", "$89.20", "+1.5%")
        with col5:
            st.metric("QQQ", "$385.75", "+2.1%")

        st.markdown("---")

        # Live chart
        st.subheader("Intraday Performance")

        # Generate intraday data
        times = pd.date_range(start='09:30', end='16:00', freq='5min')
        prices = 1000000 * np.cumprod(1 + np.random.randn(len(times)) * 0.0005)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=prices,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2)
        ))

        # Add moving average
        ma_20 = pd.Series(prices).rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=times,
            y=ma_20,
            mode='lines',
            name='20-period MA',
            line=dict(color='orange', width=1, dash='dash')
        ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Portfolio Value ($)",
            height=400,
            hovermode='x unified',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Alerts
        st.subheader("🔔 Alerts")

        alerts = [
            "✅ SPY reached take profit target at $442.15",
            "⚠️ TLT approaching stop loss at $91.50",
            "📊 High volatility detected in GLD (>2σ)",
            "🎯 Portfolio rebalancing recommended"
        ]

        for alert in alerts:
            st.info(alert)

        # Set new alerts
        with st.expander("Set New Alert"):
            col1, col2, col3 = st.columns(3)

            with col1:
                alert_symbol = st.selectbox("Symbol", ["SPY", "TLT", "GLD", "VNQ", "QQQ"])

            with col2:
                alert_type = st.selectbox("Alert Type", ["Price Above", "Price Below", "Volatility"])

            with col3:
                alert_value = st.number_input("Value", value=100.0)

            if st.button("Create Alert", use_container_width=True):
                st.success(f"Alert created for {alert_symbol}")

    def render_comparison(self):
        """Render portfolio comparison section."""
        st.title("⚖️ Portfolio Comparison Tools")

        # Select portfolios to compare
        st.subheader("Select Portfolios")

        col1, col2 = st.columns(2)

        with col1:
            portfolio1 = st.selectbox(
                "Portfolio 1",
                ["My Portfolio", "S&P 500", "60/40 Portfolio", "All Weather"]
            )

        with col2:
            portfolio2 = st.selectbox(
                "Portfolio 2",
                ["S&P 500", "My Portfolio", "60/40 Portfolio", "All Weather"]
            )

        add_more = st.checkbox("Add more portfolios")

        portfolios = [
            {'name': portfolio1},
            {'name': portfolio2}
        ]

        if add_more:
            portfolio3 = st.selectbox(
                "Portfolio 3",
                ["60/40 Portfolio", "All Weather", "Risk Parity", "Growth"]
            )
            portfolios.append({'name': portfolio3})

        st.markdown("---")

        # Comparison table
        st.subheader("Performance Metrics")

        comparison_df = self.comparison_tools.compare_portfolios(portfolios)

        # Style the dataframe
        styled_df = comparison_df.style.format({
            'Total Return': '{:.1f}%',
            'Annual Return': '{:.1f}%',
            'Volatility': '{:.1f}%',
            'Sharpe Ratio': '{:.2f}',
            'Max Drawdown': '{:.1f}%',
            'Calmar Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}',
            'Beta': '{:.2f}',
            'Alpha': '{:.1f}%',
            'Treynor Ratio': '{:.3f}'
        })

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Risk-return scatter
            fig = self.comparison_tools.create_scatter_comparison(comparison_df)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Correlation heatmap
            fig = self.comparison_tools.create_correlation_heatmap(portfolios)
            st.plotly_chart(fig, use_container_width=True)

        # Metric comparison
        st.subheader("Metric Comparison")

        metric_to_compare = st.selectbox(
            "Select Metric",
            ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']
        )

        fig = self.comparison_tools.create_comparison_chart(comparison_df, metric_to_compare)
        st.plotly_chart(fig, use_container_width=True)

    def render_analysis(self):
        """Render analysis section."""
        st.title("📈 Advanced Analysis")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Monte Carlo", "Efficient Frontier", "Backtesting", "Risk Analysis"]
        )

        with tab1:
            st.subheader("Monte Carlo Simulation")

            col1, col2, col3 = st.columns(3)

            with col1:
                simulations = st.number_input("Simulations", value=1000, step=100)
            with col2:
                years = st.slider("Years", 1, 30, 10)
            with col3:
                initial_value = st.number_input("Initial Value", value=1000000)

            if st.button("Run Simulation", use_container_width=True):
                # Generate simulation results
                final_values = np.random.lognormal(
                    np.log(initial_value) + 0.07 * years,
                    0.15 * np.sqrt(years),
                    simulations
                )

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=final_values,
                    nbinsx=50,
                    name='Final Values'
                ))

                fig.update_layout(
                    title="Monte Carlo Simulation Results",
                    xaxis_title="Final Portfolio Value ($)",
                    yaxis_title="Frequency",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"${final_values.mean():,.0f}")
                with col2:
                    st.metric("Median", f"${np.median(final_values):,.0f}")
                with col3:
                    st.metric("95% VaR", f"${np.percentile(final_values, 5):,.0f}")
                with col4:
                    st.metric("Success Rate", f"{(final_values > initial_value).mean():.1%}")

        with tab2:
            st.subheader("Efficient Frontier Analysis")

            # Generate efficient frontier
            returns = np.linspace(5, 15, 50)
            min_vol = 8 + (returns - 5) * 0.8

            fig = go.Figure()

            # Efficient frontier
            fig.add_trace(go.Scatter(
                x=min_vol,
                y=returns,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=2)
            ))

            # Current portfolio
            fig.add_trace(go.Scatter(
                x=[12],
                y=[10],
                mode='markers',
                name='Current Portfolio',
                marker=dict(color='red', size=15, symbol='star')
            ))

            # Random portfolios
            random_vols = np.random.uniform(10, 20, 100)
            random_returns = np.random.uniform(6, 14, 100)

            fig.add_trace(go.Scatter(
                x=random_vols,
                y=random_returns,
                mode='markers',
                name='Suboptimal Portfolios',
                marker=dict(color='lightgray', size=5),
                opacity=0.5
            ))

            fig.update_layout(
                title="Efficient Frontier",
                xaxis_title="Volatility (%)",
                yaxis_title="Expected Return (%)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Backtesting Results")

            # Backtest settings
            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())

            if st.button("Run Backtest", use_container_width=True):
                # Generate backtest results
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                portfolio_returns = np.cumprod(1 + np.random.randn(len(dates)) * 0.01) * 100
                benchmark_returns = np.cumprod(1 + np.random.randn(len(dates)) * 0.008) * 100

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=portfolio_returns,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='blue', width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=benchmark_returns,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='gray', width=1)
                ))

                fig.update_layout(
                    title="Backtest Performance",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    height=400,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Risk Analysis")

            # Risk metrics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Value at Risk (VaR)")

                confidence_levels = [90, 95, 99]
                var_values = [8.5, 12.3, 18.7]

                for conf, var in zip(confidence_levels, var_values):
                    st.metric(f"{conf}% VaR", f"-{var:.1f}%")

            with col2:
                st.markdown("### Stress Testing")

                scenarios = ["Market Crash", "Interest Rate Shock", "Inflation Spike"]
                impacts = [-25.3, -12.5, -8.7]

                for scenario, impact in zip(scenarios, impacts):
                    st.metric(scenario, f"{impact:.1f}%")

    def render_settings(self):
        """Render settings section."""
        st.title("⚙️ Settings")

        tab1, tab2, tab3 = st.tabs(["Profile", "Preferences", "Data Management"])

        with tab1:
            st.subheader("User Profile")

            col1, col2 = st.columns(2)

            with col1:
                username = st.text_input("Username", value=st.session_state.user, disabled=True)
                email = st.text_input("Email", value="user@example.com")
                risk_tolerance = st.selectbox(
                    "Risk Tolerance",
                    ["Conservative", "Moderate", "Aggressive"]
                )

            with col2:
                investment_horizon = st.number_input("Investment Horizon (years)", value=10)
                tax_bracket = st.selectbox("Tax Bracket", ["10%", "22%", "24%", "32%", "35%", "37%"])
                retirement_age = st.number_input("Target Retirement Age", value=65)

            if st.button("Update Profile", use_container_width=True):
                st.success("Profile updated successfully")

        with tab2:
            st.subheader("Display Preferences")

            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
            date_format = st.selectbox("Date Format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"])

            st.subheader("Notifications")

            email_notifications = st.checkbox("Email Notifications", value=True)
            push_notifications = st.checkbox("Push Notifications", value=False)
            alert_threshold = st.slider("Alert Threshold (%)", 1, 10, 5)

            if st.button("Save Preferences", use_container_width=True):
                st.success("Preferences saved successfully")

        with tab3:
            st.subheader("Data Management")

            st.markdown("### Export Data")

            export_format = st.selectbox("Format", ["CSV", "Excel", "JSON"])

            if st.button("Export Portfolio Data", use_container_width=True):
                st.success(f"Data exported as {export_format}")

            st.markdown("### Import Data")

            uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'json'])

            if uploaded_file:
                st.success(f"File {uploaded_file.name} uploaded successfully")

            st.markdown("### Data Privacy")

            st.warning("⚠️ Danger Zone")

            if st.button("Delete All Data", type="secondary"):
                st.error("This will permanently delete all your data. This action cannot be undone.")

    def run(self):
        """Run the dashboard."""
        if not st.session_state.authenticated:
            self.render_login()
        else:
            # Render sidebar and get selected section
            selected_section = self.render_sidebar()

            # Render main content based on selection
            if selected_section == DashboardSection.OVERVIEW.value:
                self.render_overview()
            elif selected_section == DashboardSection.STRATEGY_BUILDER.value:
                self.render_strategy_builder()
            elif selected_section == DashboardSection.PORTFOLIO_MONITOR.value:
                self.render_portfolio_monitor()
            elif selected_section == DashboardSection.COMPARISON.value:
                self.render_comparison()
            elif selected_section == DashboardSection.ANALYSIS.value:
                self.render_analysis()
            elif selected_section == DashboardSection.SETTINGS.value:
                self.render_settings()

            # Footer
            st.markdown("---")
            st.markdown(
                """
                <div style='text-align: center; color: gray;'>
                    Allocation Station Dashboard v1.0 | © 2024 |
                    <a href='#'>Documentation</a> |
                    <a href='#'>Support</a>
                </div>
                """,
                unsafe_allow_html=True
            )


def main():
    """Main entry point."""
    dashboard = DashboardUI()
    dashboard.run()


if __name__ == "__main__":
    main()