"""
Advanced Visualization Examples

This module demonstrates all capabilities of the advanced visualization system
including 3D efficient frontiers, interactive correlation heatmaps, animated
historical replays, risk factor decomposition, portfolio evolution timelines,
geographic allocation maps, and custom chart builders.

Examples:
    1. 3D Efficient Frontier Visualization
    2. Interactive Correlation Heatmap with Clustering
    3. Animated Historical Portfolio Replay
    4. Risk Factor Decomposition Charts
    5. Portfolio Evolution Timeline with Events
    6. Geographic Allocation Maps
    7. Custom Chart Builder Framework
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from allocation_station.ui.visualizations import (
    EfficientFrontier3D,
    InteractiveCorrelationHeatmap,
    AnimatedHistoricalReplay,
    RiskFactorDecomposition,
    PortfolioEvolutionTimeline,
    GeographicAllocationMap,
    CustomChartBuilder,
    VisualizationSuite,
    ChartConfig,
    ChartType,
    ColorScheme
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def example_1_3d_efficient_frontier():
    """
    Example 1: 3D Efficient Frontier Visualization

    Demonstrates creating a 3D surface plot of the efficient frontier showing
    the relationship between risk, return, and Sharpe ratio.
    """
    print_section("Example 1: 3D Efficient Frontier Visualization")

    # Generate sample efficient frontier data
    np.random.seed(42)
    n_portfolios = 1000

    # Random portfolios
    returns = np.random.uniform(5, 15, n_portfolios)
    volatilities = np.random.uniform(8, 25, n_portfolios)

    # Calculate Sharpe ratios
    rf_rate = 3.0  # Risk-free rate
    sharpe_ratios = (returns - rf_rate) / volatilities

    # Asset names for key portfolios
    asset_names = ['Conservative', 'Moderate', 'Aggressive', 'Maximum Sharpe', 'Min Volatility']
    key_indices = np.random.choice(n_portfolios, 5, replace=False)

    # Create 3D frontier
    frontier_viz = EfficientFrontier3D()

    config = ChartConfig(
        title="3D Efficient Frontier: Risk-Return-Sharpe Surface",
        width=1000,
        height=700
    )

    fig = frontier_viz.create_3d_frontier(
        returns=returns,
        volatilities=volatilities,
        sharpe_ratios=sharpe_ratios,
        asset_names=[asset_names[i] for i in range(5)],
        config=config
    )

    # Add optimal portfolio marker
    max_sharpe_idx = np.argmax(sharpe_ratios)
    frontier_viz.add_optimal_portfolio(
        volatility=volatilities[max_sharpe_idx],
        return_val=returns[max_sharpe_idx],
        sharpe=sharpe_ratios[max_sharpe_idx],
        name="Maximum Sharpe"
    )

    print("[OK] Created 3D Efficient Frontier")
    print(f"  Portfolios plotted: {n_portfolios}")
    print(f"  Return range: {returns.min():.2f}% - {returns.max():.2f}%")
    print(f"  Volatility range: {volatilities.min():.2f}% - {volatilities.max():.2f}%")
    print(f"  Max Sharpe ratio: {sharpe_ratios.max():.2f}")
    print(f"\n  Optimal Portfolio:")
    print(f"    Return: {returns[max_sharpe_idx]:.2f}%")
    print(f"    Volatility: {volatilities[max_sharpe_idx]:.2f}%")
    print(f"    Sharpe: {sharpe_ratios[max_sharpe_idx]:.2f}")

    # Save to HTML
    fig.write_html("3d_efficient_frontier.html")
    print(f"\n[OK] Saved to: 3d_efficient_frontier.html")


def example_2_correlation_heatmap():
    """
    Example 2: Interactive Correlation Heatmap with Clustering

    Demonstrates creating interactive correlation heatmaps with hierarchical
    clustering and dendrograms for identifying asset groupings.
    """
    print_section("Example 2: Interactive Correlation Heatmap with Clustering")

    # Generate sample correlation matrix
    assets = ['US Stocks', 'Intl Stocks', 'EM Stocks', 'US Bonds', 'Intl Bonds',
             'REITs', 'Commodities', 'Gold', 'Cash', 'Crypto']

    n_assets = len(assets)

    # Create realistic correlation structure
    np.random.seed(42)
    base_corr = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
    correlation_matrix = (base_corr + base_corr.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)

    # Make stocks more correlated with each other
    correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.85
    correlation_matrix[0, 2] = correlation_matrix[2, 0] = 0.75
    correlation_matrix[1, 2] = correlation_matrix[2, 1] = 0.80

    # Make bonds more correlated with each other
    correlation_matrix[3, 4] = correlation_matrix[4, 3] = 0.80

    # Make negative correlation between stocks and bonds
    correlation_matrix[0, 3] = correlation_matrix[3, 0] = -0.20
    correlation_matrix[1, 4] = correlation_matrix[4, 1] = -0.15

    corr_df = pd.DataFrame(correlation_matrix, index=assets, columns=assets)

    # Create clustered heatmap
    heatmap_viz = InteractiveCorrelationHeatmap()

    config = ChartConfig(
        title="Asset Correlation Heatmap with Hierarchical Clustering",
        width=900,
        height=800
    )

    fig = heatmap_viz.create_clustered_heatmap(corr_df, config)

    print("[OK] Created Clustered Correlation Heatmap")
    print(f"  Assets: {len(assets)}")
    print(f"  Correlation range: {correlation_matrix.min():.2f} to {correlation_matrix.max():.2f}")

    print(f"\n  Highest Correlations:")
    # Find highest correlations (excluding diagonal)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    correlation_matrix_masked = correlation_matrix.copy()
    correlation_matrix_masked[mask] = np.nan

    for _ in range(3):
        max_idx = np.nanargmax(correlation_matrix_masked)
        i, j = np.unravel_index(max_idx, correlation_matrix_masked.shape)
        print(f"    {assets[i]} - {assets[j]}: {correlation_matrix[i, j]:.2f}")
        correlation_matrix_masked[i, j] = np.nan

    # Save to HTML
    fig.write_html("correlation_heatmap_clustered.html")
    print(f"\n[OK] Saved to: correlation_heatmap_clustered.html")

    # Also create simple interactive heatmap
    fig2 = heatmap_viz.create_interactive_heatmap(corr_df)
    fig2.write_html("correlation_heatmap_simple.html")
    print(f"[OK] Saved to: correlation_heatmap_simple.html")


def example_3_animated_replay():
    """
    Example 3: Animated Historical Portfolio Replay

    Demonstrates creating animated visualizations that replay portfolio
    evolution over time with smooth transitions and playback controls.
    """
    print_section("Example 3: Animated Historical Portfolio Replay")

    # Generate sample historical data
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 1, 1)
    dates = pd.date_range(start_date, end_date, freq='D')

    # Simulate portfolio value with trend and volatility
    n_days = len(dates)
    returns = np.random.randn(n_days) * 0.01 + 0.0003  # Daily returns with upward drift
    cumulative_returns = np.cumprod(1 + returns)
    portfolio_values = 1000000 * cumulative_returns

    portfolio_df = pd.DataFrame({
        'date': dates,
        'value': portfolio_values
    })

    # Create animated replay
    replay_viz = AnimatedHistoricalReplay()

    config = ChartConfig(
        title="Portfolio Historical Replay (2020-2024)",
        width=1000,
        height=600
    )

    fig = replay_viz.create_animated_replay(
        historical_data=portfolio_df,
        date_column='date',
        value_column='value',
        config=config
    )

    print("[OK] Created Animated Historical Replay")
    print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Data points: {len(dates)}")
    print(f"  Starting value: ${portfolio_values[0]:,.2f}")
    print(f"  Ending value: ${portfolio_values[-1]:,.2f}")
    print(f"  Total return: {(portfolio_values[-1] / portfolio_values[0] - 1) * 100:.2f}%")

    fig.write_html("animated_portfolio_replay.html")
    print(f"\n[OK] Saved to: animated_portfolio_replay.html")
    print("  Open in browser to see animation with play/pause controls")

    # Create comparative replay (portfolio vs benchmark)
    benchmark_returns = np.random.randn(n_days) * 0.008 + 0.0002
    benchmark_values = 1000000 * np.cumprod(1 + benchmark_returns)

    benchmark_df = pd.DataFrame({
        'date': dates,
        'value': benchmark_values
    })

    fig2 = replay_viz.create_comparative_replay(
        portfolio_data=portfolio_df,
        benchmark_data=benchmark_df
    )

    print("\n[OK] Created Portfolio vs Benchmark Animated Replay")
    print(f"  Portfolio final value: ${portfolio_values[-1]:,.2f}")
    print(f"  Benchmark final value: ${benchmark_values[-1]:,.2f}")
    print(f"  Outperformance: ${portfolio_values[-1] - benchmark_values[-1]:,.2f}")

    fig2.write_html("animated_comparison_replay.html")
    print(f"[OK] Saved to: animated_comparison_replay.html")


def example_4_risk_decomposition():
    """
    Example 4: Risk Factor Decomposition Charts

    Demonstrates visualizing risk factor contributions with waterfall charts
    and treemaps showing hierarchical risk decomposition.
    """
    print_section("Example 4: Risk Factor Decomposition Charts")

    # Sample risk factor contributions
    factors = {
        'Equity Risk': 5.2,
        'Interest Rate Risk': 2.1,
        'Credit Risk': 1.8,
        'Currency Risk': -0.5,
        'Commodity Risk': 0.9,
        'Volatility Risk': 1.2,
        'Liquidity Risk': 0.6
    }

    # Create risk decomposition visualizer
    risk_viz = RiskFactorDecomposition()

    config = ChartConfig(
        title="Risk Factor Contribution Analysis",
        width=900,
        height=600
    )

    # Create waterfall chart
    fig = risk_viz.create_factor_contribution_chart(factors, config)

    total_risk = sum(factors.values())
    print("[OK] Created Risk Factor Contribution Waterfall Chart")
    print(f"  Total Risk: {total_risk:.2f}%")
    print(f"\n  Top Risk Factors:")

    sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1]), reverse=True)
    for factor, contribution in sorted_factors[:3]:
        print(f"    {factor}: {contribution:+.2f}%")

    fig.write_html("risk_factor_waterfall.html")
    print(f"\n[OK] Saved to: risk_factor_waterfall.html")

    # Create hierarchical risk decomposition tree
    factor_hierarchy = {
        'Total Risk': {
            'Market Risk': {
                'Equity Risk': 5.2,
                'Interest Rate Risk': 2.1,
                'Currency Risk': -0.5
            },
            'Credit Risk': {
                'Investment Grade': 1.2,
                'High Yield': 0.6
            },
            'Other Risk': {
                'Commodity Risk': 0.9,
                'Volatility Risk': 1.2,
                'Liquidity Risk': 0.6
            }
        }
    }

    fig2 = risk_viz.create_factor_decomposition_tree(
        factor_hierarchy,
        ChartConfig(title="Hierarchical Risk Factor Decomposition")
    )

    print("\n[OK] Created Hierarchical Risk Decomposition Treemap")
    fig2.write_html("risk_factor_treemap.html")
    print(f"[OK] Saved to: risk_factor_treemap.html")


def example_5_evolution_timeline():
    """
    Example 5: Portfolio Evolution Timeline with Events

    Demonstrates creating portfolio evolution timelines with event annotations
    marking significant portfolio changes or market events.
    """
    print_section("Example 5: Portfolio Evolution Timeline with Events")

    # Generate portfolio data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    values = 1000000 * np.cumprod(1 + np.random.randn(len(dates)) * 0.01)

    portfolio_df = pd.DataFrame({
        'date': dates,
        'value': values
    })

    # Define significant events
    events = [
        {
            'date': '2020-03-15',
            'label': 'COVID-19 Market Crash',
            'type': 'negative'
        },
        {
            'date': '2020-06-01',
            'label': 'Portfolio Rebalancing',
            'type': 'neutral'
        },
        {
            'date': '2021-01-15',
            'label': 'Increased Equity Allocation',
            'type': 'positive'
        },
        {
            'date': '2022-03-01',
            'label': 'Russia-Ukraine Conflict',
            'type': 'negative'
        },
        {
            'date': '2023-01-01',
            'label': 'Strategic Reallocation',
            'type': 'positive'
        }
    ]

    # Create timeline visualizer
    timeline_viz = PortfolioEvolutionTimeline()

    config = ChartConfig(
        title="Portfolio Evolution Timeline with Key Events",
        width=1200,
        height=600
    )

    fig = timeline_viz.create_evolution_timeline(
        portfolio_data=portfolio_df,
        events=events,
        config=config
    )

    print("[OK] Created Portfolio Evolution Timeline")
    print(f"  Date range: 2020-2024")
    print(f"  Events marked: {len(events)}")
    print(f"\n  Key Events:")
    for event in events:
        print(f"    • {event['date']}: {event['label']}")

    fig.write_html("portfolio_evolution_timeline.html")
    print(f"\n[OK] Saved to: portfolio_evolution_timeline.html")

    # Create multi-portfolio comparison timeline
    portfolios = {
        'Conservative': pd.DataFrame({
            'date': dates,
            'value': 1000000 * np.cumprod(1 + np.random.randn(len(dates)) * 0.006)
        }),
        'Moderate': pd.DataFrame({
            'date': dates,
            'value': 1000000 * np.cumprod(1 + np.random.randn(len(dates)) * 0.008)
        }),
        'Aggressive': pd.DataFrame({
            'date': dates,
            'value': 1000000 * np.cumprod(1 + np.random.randn(len(dates)) * 0.012)
        })
    }

    fig2 = timeline_viz.create_multi_portfolio_timeline(
        portfolios=portfolios,
        config=ChartConfig(title="Multi-Portfolio Evolution Comparison")
    )

    print("\n[OK] Created Multi-Portfolio Timeline")
    print(f"  Portfolios compared: {len(portfolios)}")

    fig2.write_html("multi_portfolio_timeline.html")
    print(f"[OK] Saved to: multi_portfolio_timeline.html")


def example_6_geographic_maps():
    """
    Example 6: Geographic Allocation Maps

    Demonstrates creating choropleth and bubble maps showing geographic
    distribution of portfolio allocations across countries and regions.
    """
    print_section("Example 6: Geographic Allocation Maps")

    # Sample country allocations (using ISO country codes)
    country_allocations = {
        'USA': 45.5,
        'CHN': 12.3,
        'JPN': 8.7,
        'GBR': 6.2,
        'DEU': 5.1,
        'FRA': 4.3,
        'CAN': 3.8,
        'AUS': 3.2,
        'IND': 2.9,
        'BRA': 2.1,
        'KOR': 1.8,
        'MEX': 1.5,
        'ITA': 1.3,
        'ESP': 1.1,
        'NLD': 0.9
    }

    # Create geographic map visualizer
    geo_viz = GeographicAllocationMap()

    config = ChartConfig(
        title="Global Portfolio Allocation by Country",
        width=1200,
        height=700
    )

    # Create choropleth map
    fig = geo_viz.create_country_allocation_map(
        allocations=country_allocations,
        config=config
    )

    total_allocation = sum(country_allocations.values())
    print("[OK] Created Geographic Allocation Choropleth Map")
    print(f"  Countries: {len(country_allocations)}")
    print(f"  Total allocation: {total_allocation:.1f}%")
    print(f"\n  Top 5 Countries:")

    sorted_countries = sorted(country_allocations.items(), key=lambda x: x[1], reverse=True)
    for country, allocation in sorted_countries[:5]:
        print(f"    {country}: {allocation:.1f}%")

    fig.write_html("geographic_allocation_map.html")
    print(f"\n[OK] Saved to: geographic_allocation_map.html")

    # Create regional bubble map
    regional_data = pd.DataFrame({
        'name': ['North America', 'Europe', 'Asia', 'Latin America', 'Oceania'],
        'lat': [40, 50, 35, -15, -25],
        'lon': [-100, 10, 105, -60, 135],
        'allocation': [52.8, 18.9, 23.7, 3.6, 3.2]
    })

    fig2 = geo_viz.create_regional_bubble_map(
        regions=regional_data,
        config=ChartConfig(title="Regional Allocation Bubble Map")
    )

    print("\n[OK] Created Regional Bubble Map")
    print(f"  Regions: {len(regional_data)}")
    for _, row in regional_data.iterrows():
        print(f"    {row['name']}: {row['allocation']:.1f}%")

    fig2.write_html("regional_bubble_map.html")
    print(f"[OK] Saved to: regional_bubble_map.html")


def example_7_custom_chart_builder():
    """
    Example 7: Custom Chart Builder Framework

    Demonstrates using the custom chart builder to create tailored
    visualizations by combining multiple trace types and layouts.
    """
    print_section("Example 7: Custom Chart Builder Framework")

    # Create custom chart builder
    builder = CustomChartBuilder()

    # Example 1: Combined line and bar chart
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
    portfolio_returns = np.random.randn(len(dates)) * 2 + 1
    benchmark_returns = np.random.randn(len(dates)) * 1.5 + 0.8

    # Add line trace for portfolio
    builder.add_trace(
        ChartType.LINE,
        data={'x': dates, 'y': np.cumsum(portfolio_returns)},
        name='Portfolio Cumulative Return',
        line=dict(color='blue', width=2)
    )

    # Add line trace for benchmark
    builder.add_trace(
        ChartType.LINE,
        data={'x': dates, 'y': np.cumsum(benchmark_returns)},
        name='Benchmark Cumulative Return',
        line=dict(color='gray', width=2, dash='dash')
    )

    # Build chart
    config = ChartConfig(
        title="Custom Performance Comparison Chart",
        width=1000,
        height=600
    )

    fig = builder.build(config)

    # Add annotation
    builder.add_annotation(
        text="Outperformance Period",
        x=dates[6],
        y=np.cumsum(portfolio_returns)[6],
        showarrow=True,
        arrowhead=2
    )

    print("[OK] Created Custom Combined Chart")
    print(f"  Traces: 2 line charts")
    print(f"  Annotations: 1")
    print(f"  Data points: {len(dates)} per trace")

    fig.write_html("custom_chart_combined.html")
    print(f"[OK] Saved to: custom_chart_combined.html")

    # Example 2: Multi-panel dashboard
    builder2 = CustomChartBuilder()

    # Create subplot grid
    builder2.add_subplot_grid(
        rows=2,
        cols=2,
        subplot_titles=['Returns', 'Allocation', 'Risk', 'Correlation']
    )

    print("\n[OK] Created Custom Multi-Panel Dashboard")
    print(f"  Layout: 2x2 grid")
    print(f"  Panels: 4")

    # Save as HTML
    builder.save_html("custom_dashboard.html")
    print(f"\n[OK] Saved to: custom_dashboard.html")

    print("\nCustom Chart Builder Features:")
    print("  • Flexible trace types (line, bar, scatter, pie, 3D)")
    print("  • Subplot grid support")
    print("  • Custom annotations")
    print("  • HTML and image export")
    print("  • Full control over styling and layout")


def demo_visualization_suite():
    """Demonstrate the comprehensive visualization suite."""
    print_section("Comprehensive Visualization Suite Demo")

    # Create visualization suite
    suite = VisualizationSuite()

    print("Visualization Suite Components:")
    print("  [OK] 3D Efficient Frontier")
    print("  [OK] Interactive Correlation Heatmaps")
    print("  [OK] Animated Historical Replays")
    print("  [OK] Risk Factor Decomposition")
    print("  [OK] Portfolio Evolution Timelines")
    print("  [OK] Geographic Allocation Maps")
    print("  [OK] Custom Chart Builder")

    print("\nAll visualizations are:")
    print("  • Interactive (hover, zoom, pan)")
    print("  • Exportable (HTML, PNG, SVG)")
    print("  • Customizable (colors, themes, layouts)")
    print("  • Production-ready")

    print("\nVisualization Technologies:")
    print("  • Plotly for interactive charts")
    print("  • 3D surface and scatter plots")
    print("  • Choropleth maps with geo projections")
    print("  • Animated frames with playback controls")
    print("  • Hierarchical clustering for heatmaps")
    print("  • Waterfall and treemap charts")


def main():
    """Run all visualization examples."""
    print("\n" + "=" * 80)
    print(" ALLOCATION STATION - ADVANCED VISUALIZATION EXAMPLES")
    print(" Comprehensive demonstration of visualization capabilities")
    print("=" * 80)

    try:
        example_1_3d_efficient_frontier()
        example_2_correlation_heatmap()
        example_3_animated_replay()
        example_4_risk_decomposition()
        example_5_evolution_timeline()
        example_6_geographic_maps()
        example_7_custom_chart_builder()
        demo_visualization_suite()

        print("\n" + "=" * 80)
        print(" All examples completed successfully!")
        print(" Generated visualizations:")
        print("   • 3d_efficient_frontier.html")
        print("   • correlation_heatmap_clustered.html")
        print("   • correlation_heatmap_simple.html")
        print("   • animated_portfolio_replay.html")
        print("   • animated_comparison_replay.html")
        print("   • risk_factor_waterfall.html")
        print("   • risk_factor_treemap.html")
        print("   • portfolio_evolution_timeline.html")
        print("   • multi_portfolio_timeline.html")
        print("   • geographic_allocation_map.html")
        print("   • regional_bubble_map.html")
        print("   • custom_chart_combined.html")
        print("   • custom_dashboard.html")
        print("\n Open any HTML file in a web browser to view interactive charts!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
