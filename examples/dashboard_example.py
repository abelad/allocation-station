"""
Interactive Dashboard Example

This module demonstrates how to run and use the Allocation Station interactive
dashboard, including all features like real-time monitoring, strategy building,
portfolio comparison, and analysis tools.

Usage:
    streamlit run examples/dashboard_example.py

Features Demonstrated:
    1. User authentication and profile management
    2. Real-time portfolio monitoring
    3. Interactive strategy builder with drag-and-drop
    4. Portfolio comparison tools
    5. Monte Carlo simulation
    6. Efficient frontier analysis
    7. Backtesting capabilities
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import time
    import json
    from typing import Dict, List

    # Import dashboard components
    from allocation_station.ui.dashboard import (
        DashboardUI,
        AuthenticationManager,
        RealTimeMonitor,
        StrategyBuilder,
        PortfolioComparison
    )
    STREAMLIT_AVAILABLE = True
except ImportError as e:
    STREAMLIT_AVAILABLE = False
    IMPORT_ERROR = str(e)


def create_sample_data():
    """Create sample data for demonstration."""

    # Sample portfolio data
    portfolio = {
        'holdings': {
            'SPY': 100,  # S&P 500 ETF
            'TLT': 150,  # 20+ Year Treasury Bond ETF
            'GLD': 50,   # Gold ETF
            'VNQ': 75,   # Real Estate ETF
            'QQQ': 80    # Nasdaq 100 ETF
        },
        'cost_basis': {
            'SPY': 400.00,
            'TLT': 100.00,
            'GLD': 170.00,
            'VNQ': 85.00,
            'QQQ': 350.00
        },
        'purchase_dates': {
            'SPY': datetime(2023, 1, 15),
            'TLT': datetime(2023, 2, 10),
            'GLD': datetime(2023, 3, 5),
            'VNQ': datetime(2023, 4, 20),
            'QQQ': datetime(2023, 5, 1)
        }
    }

    # Sample allocation strategies
    strategies = {
        'conservative': {
            'name': 'Conservative Portfolio',
            'allocation': {
                'stocks': 30,
                'bonds': 60,
                'commodities': 5,
                'real_estate': 5
            },
            'risk_score': 3,
            'expected_return': 5.2,
            'volatility': 8.5
        },
        'moderate': {
            'name': 'Moderate Portfolio',
            'allocation': {
                'stocks': 50,
                'bonds': 35,
                'commodities': 10,
                'real_estate': 5
            },
            'risk_score': 5,
            'expected_return': 7.8,
            'volatility': 12.3
        },
        'aggressive': {
            'name': 'Aggressive Portfolio',
            'allocation': {
                'stocks': 70,
                'bonds': 20,
                'commodities': 5,
                'real_estate': 5
            },
            'risk_score': 7,
            'expected_return': 9.5,
            'volatility': 16.8
        },
        'all_weather': {
            'name': 'All Weather Portfolio',
            'allocation': {
                'stocks': 30,
                'bonds': 55,
                'commodities': 15,
                'real_estate': 0
            },
            'risk_score': 4,
            'expected_return': 6.5,
            'volatility': 10.2
        }
    }

    # Sample historical performance data
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    performance = pd.DataFrame({
        'date': dates,
        'portfolio_value': 1000000 * np.cumprod(1 + np.random.randn(len(dates)) * 0.01),
        'spy_value': 1000000 * np.cumprod(1 + np.random.randn(len(dates)) * 0.008),
        'daily_return': np.random.randn(len(dates)) * 0.02
    })

    # Sample alerts
    alerts = [
        {
            'type': 'price',
            'symbol': 'SPY',
            'condition': 'above',
            'threshold': 450,
            'active': True,
            'created': datetime.now() - timedelta(days=5)
        },
        {
            'type': 'volatility',
            'symbol': 'GLD',
            'condition': 'above',
            'threshold': 0.25,
            'active': True,
            'created': datetime.now() - timedelta(days=10)
        },
        {
            'type': 'drawdown',
            'symbol': 'portfolio',
            'condition': 'below',
            'threshold': -0.10,
            'active': True,
            'created': datetime.now() - timedelta(days=2)
        }
    ]

    return {
        'portfolio': portfolio,
        'strategies': strategies,
        'performance': performance,
        'alerts': alerts
    }


def demo_authentication():
    """Demonstrate authentication features."""
    st.header("🔐 Authentication Demo")

    auth_manager = AuthenticationManager()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Create Demo Users")

        # Create demo users
        demo_users = [
            ('demo_user', 'password123', 'demo@example.com'),
            ('investor1', 'secure456', 'investor1@example.com'),
            ('trader2', 'trading789', 'trader2@example.com')
        ]

        for username, password, email in demo_users:
            if auth_manager.register_user(username, password, email):
                st.success(f"Created user: {username}")
            else:
                st.info(f"User {username} already exists")

    with col2:
        st.subheader("Test Authentication")

        test_username = st.text_input("Test Username", value="demo_user")
        test_password = st.text_input("Test Password", value="password123", type="password")

        if st.button("Test Login"):
            session_token = auth_manager.authenticate_user(test_username, test_password)

            if session_token:
                st.success(f"Login successful! Token: {session_token[:10]}...")

                # Validate session
                user_data = auth_manager.validate_session(session_token)
                if user_data:
                    st.json(user_data)
            else:
                st.error("Login failed")


def demo_realtime_monitor():
    """Demonstrate real-time monitoring features."""
    st.header("📊 Real-Time Monitoring Demo")

    monitor = RealTimeMonitor()
    sample_data = create_sample_data()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Live Prices")

        # Note: This would fetch real prices in production
        symbols = list(sample_data['portfolio']['holdings'].keys())

        # Simulate live prices
        live_prices = {}
        for symbol in symbols:
            base_price = sample_data['portfolio']['cost_basis'][symbol]
            live_prices[symbol] = base_price * (1 + np.random.randn() * 0.05)

        price_df = pd.DataFrame({
            'Symbol': symbols,
            'Price': [f"${p:.2f}" for p in live_prices.values()],
            'Change': [f"{np.random.randn()*2:.2f}%" for _ in symbols]
        })

        st.dataframe(price_df, hide_index=True)

    with col2:
        st.subheader("Portfolio Metrics")

        total_value = monitor.calculate_portfolio_value(
            sample_data['portfolio']['holdings'],
            live_prices
        )

        daily_return = monitor.calculate_daily_returns(
            sample_data['portfolio']['holdings']
        )

        st.metric("Total Value", f"${total_value:,.2f}", f"{daily_return:.2f}%")

        # Generate alerts
        test_alerts = {
            'SPY': {'stop_loss': 420, 'take_profit': 460},
            'GLD': {'volatility_threshold': 0.20}
        }

        alert_messages = monitor.monitor_portfolio(
            sample_data['portfolio']['holdings'],
            test_alerts
        )

        if alert_messages:
            st.subheader("Active Alerts")
            for msg in alert_messages:
                st.warning(msg)


def demo_strategy_builder():
    """Demonstrate strategy builder features."""
    st.header("🔧 Strategy Builder Demo")

    builder = StrategyBuilder()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Allocation Interface")

        # Interactive allocation sliders
        allocation = {}
        allocation['stocks'] = st.slider("Stocks %", 0, 100, 50, 5)
        allocation['bonds'] = st.slider("Bonds %", 0, 100, 30, 5)
        allocation['commodities'] = st.slider("Commodities %", 0, 100, 10, 5)
        allocation['real_estate'] = st.slider("Real Estate %", 0, 100, 10, 5)

        # Normalize to 100%
        total = sum(allocation.values())
        if total > 0:
            for key in allocation:
                allocation[key] = (allocation[key] / total) * 100

        st.info(f"Total Allocation: {sum(allocation.values()):.0f}%")

        # Calculate metrics
        metrics = builder.calculate_risk_metrics(allocation)

        st.markdown("### Risk Metrics")
        st.metric("Expected Return", f"{metrics['expected_return']:.1f}%")
        st.metric("Volatility", f"{metrics['volatility']:.1f}%")
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")

    with col2:
        st.subheader("Visualization")

        # Display allocation chart
        import plotly.graph_objects as go

        fig = builder.create_allocation_chart(allocation)
        st.plotly_chart(fig, use_container_width=True)

        # Optimization
        st.subheader("Optimization")

        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["low", "medium", "high"],
            value="medium"
        )

        if st.button("Optimize Allocation"):
            constraints = {
                'stocks': (20, 80),
                'bonds': (10, 60),
                'commodities': (0, 20),
                'real_estate': (0, 20)
            }

            optimized = builder.optimize_allocation(risk_tolerance, constraints)

            st.success("Optimization Complete!")
            st.json(optimized)


def demo_portfolio_comparison():
    """Demonstrate portfolio comparison features."""
    st.header("⚖️ Portfolio Comparison Demo")

    comparison = PortfolioComparison()
    sample_data = create_sample_data()

    # Create sample portfolios
    portfolios = [
        {'name': 'My Portfolio'},
        {'name': 'S&P 500'},
        {'name': '60/40 Portfolio'},
        {'name': 'All Weather'}
    ]

    # Generate comparison data
    comparison_df = comparison.compare_portfolios(portfolios)

    st.subheader("Performance Comparison")
    st.dataframe(
        comparison_df.style.format({
            'Total Return': '{:.1f}%',
            'Annual Return': '{:.1f}%',
            'Volatility': '{:.1f}%',
            'Sharpe Ratio': '{:.2f}',
            'Max Drawdown': '{:.1f}%'
        }),
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk-Return Scatter")
        fig = comparison.create_scatter_comparison(comparison_df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Correlation Matrix")
        fig = comparison.create_correlation_heatmap(portfolios)
        st.plotly_chart(fig, use_container_width=True)

    # Metric comparison
    st.subheader("Individual Metric Comparison")

    metric = st.selectbox(
        "Select Metric",
        ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']
    )

    fig = comparison.create_comparison_chart(comparison_df, metric)
    st.plotly_chart(fig, use_container_width=True)


def demo_analysis_tools():
    """Demonstrate analysis tools."""
    st.header("📈 Analysis Tools Demo")

    tab1, tab2, tab3 = st.tabs(["Monte Carlo", "Efficient Frontier", "Backtesting"])

    with tab1:
        st.subheader("Monte Carlo Simulation")

        col1, col2, col3 = st.columns(3)

        with col1:
            simulations = st.number_input("Simulations", value=1000, step=100)
        with col2:
            years = st.slider("Years", 1, 30, 10)
        with col3:
            initial = st.number_input("Initial Value", value=1000000)

        if st.button("Run Monte Carlo"):
            # Generate simulation
            final_values = np.random.lognormal(
                np.log(initial) + 0.07 * years,
                0.15 * np.sqrt(years),
                simulations
            )

            import plotly.graph_objects as go

            fig = go.Figure(data=[
                go.Histogram(x=final_values, nbinsx=50)
            ])

            fig.update_layout(
                title="Monte Carlo Results",
                xaxis_title="Final Value ($)",
                yaxis_title="Frequency"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Mean", f"${final_values.mean():,.0f}")
            with col2:
                st.metric("Median", f"${np.median(final_values):,.0f}")
            with col3:
                success_rate = (final_values > initial).mean()
                st.metric("Success Rate", f"{success_rate:.1%}")

    with tab2:
        st.subheader("Efficient Frontier")

        # Generate efficient frontier
        import plotly.graph_objects as go

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

        # Current portfolio
        fig.add_trace(go.Scatter(
            x=[12],
            y=[10],
            mode='markers',
            name='Current Portfolio',
            marker=dict(color='red', size=15, symbol='star')
        ))

        fig.update_layout(
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Backtesting")

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())

        if st.button("Run Backtest"):
            # Generate backtest data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            portfolio = np.cumprod(1 + np.random.randn(len(dates)) * 0.01) * 100
            benchmark = np.cumprod(1 + np.random.randn(len(dates)) * 0.008) * 100

            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio,
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=1)
            ))

            fig.update_layout(
                title="Backtest Performance",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main demo application."""
    st.set_page_config(
        page_title="Allocation Station Dashboard Demo",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Allocation Station Dashboard Demo")
    st.markdown("---")

    # Sidebar for navigation
    with st.sidebar:
        st.header("Demo Navigation")

        demo_section = st.selectbox(
            "Select Demo",
            [
                "Full Dashboard",
                "Authentication",
                "Real-Time Monitoring",
                "Strategy Builder",
                "Portfolio Comparison",
                "Analysis Tools"
            ]
        )

        st.markdown("---")
        st.info(
            """
            ### How to Run Full Dashboard:

            ```bash
            streamlit run src/allocation_station/ui/dashboard.py
            ```

            The full dashboard includes:
            - User authentication
            - Portfolio monitoring
            - Strategy building
            - Comparison tools
            - Advanced analysis
            """
        )

    # Render selected demo
    if demo_section == "Full Dashboard":
        st.info("To run the full dashboard, use: `streamlit run src/allocation_station/ui/dashboard.py`")

        # Show dashboard features
        st.subheader("Dashboard Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **📈 Overview**
            - Portfolio summary
            - Performance charts
            - Holdings table
            - Quick stats
            """)

        with col2:
            st.markdown("""
            **🔧 Strategy Builder**
            - Drag-and-drop allocation
            - Risk metrics
            - Preset strategies
            - Optimization
            """)

        with col3:
            st.markdown("""
            **📊 Real-Time Monitor**
            - Live price updates
            - Intraday charts
            - Alert management
            - Auto-refresh
            """)

        col4, col5, col6 = st.columns(3)

        with col4:
            st.markdown("""
            **⚖️ Comparison**
            - Multi-portfolio analysis
            - Performance metrics
            - Correlation matrix
            - Risk-return scatter
            """)

        with col5:
            st.markdown("""
            **📈 Analysis**
            - Monte Carlo
            - Efficient Frontier
            - Backtesting
            - Risk analysis
            """)

        with col6:
            st.markdown("""
            **⚙️ Settings**
            - User profile
            - Preferences
            - Data export/import
            - Privacy settings
            """)

    elif demo_section == "Authentication":
        demo_authentication()

    elif demo_section == "Real-Time Monitoring":
        demo_realtime_monitor()

    elif demo_section == "Strategy Builder":
        demo_strategy_builder()

    elif demo_section == "Portfolio Comparison":
        demo_portfolio_comparison()

    elif demo_section == "Analysis Tools":
        demo_analysis_tools()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        ### 📝 Notes:
        - This is a demonstration of the dashboard capabilities
        - For production use, real market data connections would be required
        - User data is stored in a local SQLite database
        - The dashboard is fully responsive and mobile-friendly
        """
    )


if __name__ == "__main__":
    if not STREAMLIT_AVAILABLE:
        print("\n" + "=" * 80)
        print(" Dashboard Example - Missing Dependencies")
        print("=" * 80)
        print(f"\nError: {IMPORT_ERROR}")
        print("\nThis example requires Streamlit and related dependencies.")
        print("\nTo install:")
        print("  pip install streamlit plotly")
        print("\nTo run the dashboard:")
        print("  streamlit run examples/dashboard_example.py")
        print("\n" + "=" * 80)
    else:
        main()