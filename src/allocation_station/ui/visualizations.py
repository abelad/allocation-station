"""
Advanced Visualization System for Allocation Station

This module provides enhanced visualization capabilities including 3D efficient
frontier plots, interactive correlation heatmaps, animated historical replays,
risk factor decomposition charts, portfolio evolution timelines, geographic
allocation maps, and custom chart builders.

Features:
    - 3D efficient frontier with risk/return/Sharpe surface
    - Interactive correlation heatmaps with clustering
    - Animated historical portfolio replays with transitions
    - Risk factor decomposition with contribution analysis
    - Portfolio evolution timelines with event annotations
    - Geographic allocation maps with choropleth regions
    - Custom chart builder framework for tailored visualizations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


class ChartType(Enum):
    """Chart types for visualization."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    HEATMAP = "heatmap"
    SURFACE_3D = "surface_3d"
    SCATTER_3D = "scatter_3d"
    CHOROPLETH = "choropleth"
    SANKEY = "sankey"
    WATERFALL = "waterfall"
    TREEMAP = "treemap"


class ColorScheme(Enum):
    """Color schemes for visualizations."""
    BLUE_GRADIENT = "Blues"
    RED_GRADIENT = "Reds"
    GREEN_GRADIENT = "Greens"
    VIRIDIS = "Viridis"
    PLASMA = "Plasma"
    RAINBOW = "Rainbow"
    CUSTOM = "Custom"


@dataclass
class ChartConfig:
    """Configuration for chart creation."""
    title: str
    width: int = 900
    height: int = 600
    theme: str = "plotly_white"
    show_legend: bool = True
    interactive: bool = True
    color_scheme: ColorScheme = ColorScheme.VIRIDIS


class EfficientFrontier3D:
    """3D efficient frontier visualization with risk/return/Sharpe surface."""

    def __init__(self):
        """Initialize 3D efficient frontier visualizer."""
        self.fig = None

    def create_3d_frontier(self,
                          returns: np.ndarray,
                          volatilities: np.ndarray,
                          sharpe_ratios: np.ndarray,
                          asset_names: Optional[List[str]] = None,
                          config: Optional[ChartConfig] = None) -> go.Figure:
        """
        Create 3D efficient frontier visualization.

        Args:
            returns: Array of portfolio returns
            volatilities: Array of portfolio volatilities
            sharpe_ratios: Array of Sharpe ratios
            asset_names: Optional names for individual assets
            config: Chart configuration

        Returns:
            Plotly figure with 3D surface
        """
        if config is None:
            config = ChartConfig(title="3D Efficient Frontier")

        # Create meshgrid for surface
        vol_range = np.linspace(volatilities.min(), volatilities.max(), 50)
        ret_range = np.linspace(returns.min(), returns.max(), 50)
        vol_mesh, ret_mesh = np.meshgrid(vol_range, ret_range)

        # Calculate Sharpe ratio surface
        rf_rate = 0.03  # Risk-free rate
        sharpe_mesh = (ret_mesh - rf_rate) / vol_mesh

        # Create 3D surface plot
        fig = go.Figure()

        # Add surface
        fig.add_trace(go.Surface(
            x=vol_mesh,
            y=ret_mesh,
            z=sharpe_mesh,
            colorscale='Viridis',
            name='Efficient Frontier',
            opacity=0.8,
            colorbar=dict(title="Sharpe Ratio"),
            hovertemplate='Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{z:.2f}<extra></extra>'
        ))

        # Add scatter points for actual portfolios
        fig.add_trace(go.Scatter3d(
            x=volatilities,
            y=returns,
            z=sharpe_ratios,
            mode='markers',
            marker=dict(
                size=6,
                color=sharpe_ratios,
                colorscale='Plasma',
                showscale=False,
                line=dict(color='white', width=1)
            ),
            name='Portfolios',
            text=asset_names if asset_names else None,
            hovertemplate='%{text}<br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{z:.2f}<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            scene=dict(
                xaxis_title='Volatility (%)',
                yaxis_title='Expected Return (%)',
                zaxis_title='Sharpe Ratio',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            showlegend=config.show_legend,
            template=config.theme
        )

        self.fig = fig
        return fig

    def add_optimal_portfolio(self, volatility: float, return_val: float,
                            sharpe: float, name: str = "Optimal"):
        """Add optimal portfolio marker to 3D plot."""
        if self.fig is None:
            raise ValueError("Create 3D frontier first")

        self.fig.add_trace(go.Scatter3d(
            x=[volatility],
            y=[return_val],
            z=[sharpe],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            name=name,
            hovertemplate=f'{name}<br>Vol: {volatility:.2f}%<br>Return: {return_val:.2f}%<br>Sharpe: {sharpe:.2f}<extra></extra>'
        ))


class InteractiveCorrelationHeatmap:
    """Interactive correlation heatmap with clustering and dendrograms."""

    def __init__(self):
        """Initialize correlation heatmap visualizer."""
        self.fig = None

    def create_clustered_heatmap(self,
                                correlation_matrix: pd.DataFrame,
                                config: Optional[ChartConfig] = None) -> go.Figure:
        """
        Create clustered correlation heatmap with dendrograms.

        Args:
            correlation_matrix: Correlation matrix as DataFrame
            config: Chart configuration

        Returns:
            Plotly figure with interactive heatmap
        """
        if config is None:
            config = ChartConfig(title="Correlation Heatmap with Clustering")

        # Perform hierarchical clustering
        correlation_array = correlation_matrix.values
        distance_matrix = 1 - np.abs(correlation_array)

        # Handle any NaN or infinite values
        distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=0.0)

        # Convert to condensed distance matrix
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='ward')

        # Get dendrogram ordering
        dend = dendrogram(linkage_matrix, no_plot=True)
        order = dend['leaves']

        # Reorder correlation matrix
        reordered_corr = correlation_matrix.iloc[order, order]
        labels = reordered_corr.columns.tolist()

        # Create figure with subplots for dendrograms
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.15, 0.85],
            column_widths=[0.85, 0.15],
            vertical_spacing=0.01,
            horizontal_spacing=0.01,
            specs=[[{'type': 'scatter'}, None],
                   [{'type': 'heatmap'}, {'type': 'scatter'}]]
        )

        # Main heatmap
        heatmap = go.Heatmap(
            z=reordered_corr.values,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            text=reordered_corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(
                title="Correlation",
                x=1.15,
                len=0.8
            ),
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        )

        fig.add_trace(heatmap, row=2, col=1)

        # Top dendrogram
        icoord = np.array(dend['icoord'])
        dcoord = np.array(dend['dcoord'])

        for i in range(len(icoord)):
            fig.add_trace(go.Scatter(
                x=icoord[i],
                y=dcoord[i],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)

        # Update layout
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            showlegend=False,
            template=config.theme
        )

        # Update axes
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)
        fig.update_xaxes(side='bottom', row=2, col=1)
        fig.update_yaxes(side='left', row=2, col=1)

        self.fig = fig
        return fig

    def create_interactive_heatmap(self,
                                  correlation_matrix: pd.DataFrame,
                                  config: Optional[ChartConfig] = None) -> go.Figure:
        """Create simple interactive correlation heatmap."""
        if config is None:
            config = ChartConfig(title="Interactive Correlation Heatmap")

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=config.theme
        )

        return fig


class AnimatedHistoricalReplay:
    """Animated historical portfolio replay with smooth transitions."""

    def __init__(self):
        """Initialize animated replay visualizer."""
        self.fig = None

    def create_animated_replay(self,
                              historical_data: pd.DataFrame,
                              date_column: str = 'date',
                              value_column: str = 'value',
                              config: Optional[ChartConfig] = None) -> go.Figure:
        """
        Create animated replay of historical portfolio evolution.

        Args:
            historical_data: DataFrame with date and value columns
            date_column: Name of date column
            value_column: Name of value column
            config: Chart configuration

        Returns:
            Plotly figure with animation
        """
        if config is None:
            config = ChartConfig(title="Portfolio Historical Replay")

        # Sort by date
        data = historical_data.sort_values(date_column).copy()
        dates = data[date_column].values
        values = data[value_column].values

        # Create frames for animation
        frames = []
        for i in range(1, len(dates) + 1):
            frame_data = go.Scatter(
                x=dates[:i],
                y=values[:i],
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=6, color='blue'),
                name='Portfolio Value'
            )

            frames.append(go.Frame(
                data=[frame_data],
                name=str(i)
            ))

        # Initial plot
        fig = go.Figure(
            data=[go.Scatter(
                x=dates[:1],
                y=values[:1],
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=6, color='blue'),
                name='Portfolio Value'
            )],
            frames=frames
        )

        # Add play and pause buttons
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template=config.theme,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'x': 0.1,
                'y': 1.15
            }],
            sliders=[{
                'active': 0,
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': str(dates[int(f.name) - 1]) if int(f.name) > 0 else '',
                        'method': 'animate'
                    }
                    for f in frames
                ],
                'x': 0.1,
                'len': 0.85,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )

        self.fig = fig
        return fig

    def create_comparative_replay(self,
                                 portfolio_data: pd.DataFrame,
                                 benchmark_data: pd.DataFrame,
                                 config: Optional[ChartConfig] = None) -> go.Figure:
        """Create animated replay comparing portfolio to benchmark."""
        if config is None:
            config = ChartConfig(title="Portfolio vs Benchmark Replay")

        # Merge data
        merged = pd.merge(portfolio_data, benchmark_data, on='date', suffixes=('_port', '_bench'))
        dates = merged['date'].values

        # Create frames
        frames = []
        for i in range(1, len(dates) + 1):
            frame_data = [
                go.Scatter(
                    x=dates[:i],
                    y=merged['value_port'].values[:i],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Portfolio'
                ),
                go.Scatter(
                    x=dates[:i],
                    y=merged['value_bench'].values[:i],
                    mode='lines',
                    line=dict(color='gray', width=2, dash='dash'),
                    name='Benchmark'
                )
            ]

            frames.append(go.Frame(data=frame_data, name=str(i)))

        # Create figure
        fig = go.Figure(
            data=[
                go.Scatter(x=dates[:1], y=merged['value_port'].values[:1],
                          mode='lines', line=dict(color='blue', width=2), name='Portfolio'),
                go.Scatter(x=dates[:1], y=merged['value_bench'].values[:1],
                          mode='lines', line=dict(color='gray', width=2, dash='dash'), name='Benchmark')
            ],
            frames=frames
        )

        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            xaxis_title='Date',
            yaxis_title='Value',
            template=config.theme,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 50}, 'fromcurrent': True}]},
                    {'label': 'Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                ]
            }]
        )

        return fig


class RiskFactorDecomposition:
    """Risk factor decomposition and contribution analysis charts."""

    def __init__(self):
        """Initialize risk factor decomposition visualizer."""
        self.fig = None

    def create_factor_contribution_chart(self,
                                        factors: Dict[str, float],
                                        config: Optional[ChartConfig] = None) -> go.Figure:
        """
        Create risk factor contribution waterfall chart.

        Args:
            factors: Dictionary of factor names and contributions
            config: Chart configuration

        Returns:
            Waterfall chart showing factor contributions
        """
        if config is None:
            config = ChartConfig(title="Risk Factor Contribution Analysis")

        # Prepare data
        factor_names = list(factors.keys())
        contributions = list(factors.values())

        # Calculate cumulative
        cumulative = np.cumsum([0] + contributions)

        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Risk Contribution",
            orientation="v",
            measure=["relative"] * len(factors) + ["total"],
            x=factor_names + ["Total Risk"],
            y=contributions + [sum(contributions)],
            textposition="outside",
            text=[f"{c:+.2f}%" for c in contributions] + [f"{sum(contributions):.2f}%"],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "red"}},
            decreasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}}
        ))

        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            showlegend=False,
            template=config.theme,
            yaxis_title="Risk Contribution (%)"
        )

        return fig

    def create_factor_decomposition_tree(self,
                                        factor_hierarchy: Dict[str, Any],
                                        config: Optional[ChartConfig] = None) -> go.Figure:
        """Create treemap of risk factor decomposition."""
        if config is None:
            config = ChartConfig(title="Risk Factor Decomposition Tree")

        # Flatten hierarchy for treemap
        labels = []
        parents = []
        values = []

        def flatten_dict(d, parent=""):
            for key, value in d.items():
                labels.append(key)
                parents.append(parent)
                if isinstance(value, dict):
                    values.append(sum(flatten_dict(value, key)))
                else:
                    values.append(abs(value))
            return values if not parent else []

        flatten_dict(factor_hierarchy)

        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            textinfo="label+value+percent parent",
            marker=dict(
                colorscale='RdYlGn_r',
                cmid=np.mean(values)
            ),
            hovertemplate='<b>%{label}</b><br>Risk: %{value:.2f}%<br>% of Parent: %{percentParent}<extra></extra>'
        ))

        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=config.theme
        )

        return fig


class PortfolioEvolutionTimeline:
    """Portfolio evolution timeline with event annotations."""

    def __init__(self):
        """Initialize portfolio evolution timeline visualizer."""
        self.fig = None

    def create_evolution_timeline(self,
                                 portfolio_data: pd.DataFrame,
                                 events: Optional[List[Dict]] = None,
                                 config: Optional[ChartConfig] = None) -> go.Figure:
        """
        Create portfolio evolution timeline with annotations.

        Args:
            portfolio_data: DataFrame with date and value columns
            events: List of event dictionaries with 'date', 'label', 'type'
            config: Chart configuration

        Returns:
            Timeline chart with event annotations
        """
        if config is None:
            config = ChartConfig(title="Portfolio Evolution Timeline")

        # Create main line chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=portfolio_data['date'],
            y=portfolio_data['value'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Portfolio Value',
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))

        # Add event annotations
        if events:
            for event in events:
                event_date = pd.to_datetime(event['date'])
                event_value = portfolio_data[portfolio_data['date'] == event_date]['value'].values

                if len(event_value) > 0:
                    # Add vertical line
                    fig.add_vline(
                        x=event_date,
                        line=dict(color='red' if event.get('type') == 'negative' else 'green',
                                 width=2, dash='dash'),
                        opacity=0.7
                    )

                    # Add annotation
                    fig.add_annotation(
                        x=event_date,
                        y=event_value[0],
                        text=event['label'],
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='red' if event.get('type') == 'negative' else 'green',
                        ax=0,
                        ay=-40,
                        bgcolor='white',
                        bordercolor='black',
                        borderwidth=1
                    )

        # Update layout
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template=config.theme,
            hovermode='x unified'
        )

        self.fig = fig
        return fig

    def create_multi_portfolio_timeline(self,
                                       portfolios: Dict[str, pd.DataFrame],
                                       config: Optional[ChartConfig] = None) -> go.Figure:
        """Create timeline comparing multiple portfolios."""
        if config is None:
            config = ChartConfig(title="Multi-Portfolio Evolution")

        fig = go.Figure()

        colors = px.colors.qualitative.Plotly

        for i, (name, data) in enumerate(portfolios.items()):
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data['value'],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                name=name
            ))

        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            xaxis_title='Date',
            yaxis_title='Value',
            template=config.theme,
            hovermode='x unified'
        )

        return fig


class GeographicAllocationMap:
    """Geographic allocation visualization with choropleth maps."""

    def __init__(self):
        """Initialize geographic allocation map visualizer."""
        self.fig = None

    def create_country_allocation_map(self,
                                     allocations: Dict[str, float],
                                     config: Optional[ChartConfig] = None) -> go.Figure:
        """
        Create choropleth map of country allocations.

        Args:
            allocations: Dictionary of country codes to allocation percentages
            config: Chart configuration

        Returns:
            Choropleth map
        """
        if config is None:
            config = ChartConfig(title="Geographic Allocation Map")

        # Create DataFrame
        df = pd.DataFrame([
            {'country': code, 'allocation': value}
            for code, value in allocations.items()
        ])

        # Create choropleth
        fig = go.Figure(data=go.Choropleth(
            locations=df['country'],
            z=df['allocation'],
            text=df['country'],
            colorscale='Blues',
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_title='Allocation (%)',
            hovertemplate='<b>%{text}</b><br>Allocation: %{z:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            ),
            template=config.theme
        )

        self.fig = fig
        return fig

    def create_regional_bubble_map(self,
                                   regions: pd.DataFrame,
                                   config: Optional[ChartConfig] = None) -> go.Figure:
        """
        Create bubble map showing regional allocations.

        Args:
            regions: DataFrame with 'lat', 'lon', 'name', 'allocation' columns
            config: Chart configuration

        Returns:
            Scatter geo map with bubbles
        """
        if config is None:
            config = ChartConfig(title="Regional Allocation Bubble Map")

        fig = go.Figure(data=go.Scattergeo(
            lon=regions['lon'],
            lat=regions['lat'],
            text=regions['name'],
            mode='markers',
            marker=dict(
                size=regions['allocation'] * 5,  # Scale bubble size
                color=regions['allocation'],
                colorscale='Viridis',
                showscale=True,
                colorbar_title='Allocation (%)',
                line=dict(width=0.5, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>Allocation: %{marker.color:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(204, 204, 204)'
            ),
            template=config.theme
        )

        return fig


class CustomChartBuilder:
    """Custom chart builder framework for tailored visualizations."""

    def __init__(self):
        """Initialize custom chart builder."""
        self.fig = None
        self.traces = []

    def add_trace(self, trace_type: ChartType, data: Dict[str, Any],
                 name: Optional[str] = None, **kwargs):
        """
        Add a trace to the chart.

        Args:
            trace_type: Type of chart trace
            data: Data dictionary with x, y, z as needed
            name: Trace name
            **kwargs: Additional trace parameters
        """
        if trace_type == ChartType.LINE:
            trace = go.Scatter(
                x=data.get('x'),
                y=data.get('y'),
                mode='lines',
                name=name,
                **kwargs
            )
        elif trace_type == ChartType.BAR:
            trace = go.Bar(
                x=data.get('x'),
                y=data.get('y'),
                name=name,
                **kwargs
            )
        elif trace_type == ChartType.SCATTER:
            trace = go.Scatter(
                x=data.get('x'),
                y=data.get('y'),
                mode='markers',
                name=name,
                **kwargs
            )
        elif trace_type == ChartType.PIE:
            trace = go.Pie(
                labels=data.get('labels'),
                values=data.get('values'),
                name=name,
                **kwargs
            )
        elif trace_type == ChartType.SCATTER_3D:
            trace = go.Scatter3d(
                x=data.get('x'),
                y=data.get('y'),
                z=data.get('z'),
                mode='markers',
                name=name,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported trace type: {trace_type}")

        self.traces.append(trace)

    def build(self, config: Optional[ChartConfig] = None) -> go.Figure:
        """
        Build the final chart from added traces.

        Args:
            config: Chart configuration

        Returns:
            Plotly figure
        """
        if config is None:
            config = ChartConfig(title="Custom Chart")

        self.fig = go.Figure(data=self.traces)

        self.fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend,
            template=config.theme
        )

        return self.fig

    def add_subplot_grid(self, rows: int, cols: int,
                        subplot_titles: Optional[List[str]] = None) -> 'CustomChartBuilder':
        """Create subplot grid for multiple charts."""
        self.fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles
        )
        return self

    def add_annotation(self, text: str, x: float, y: float, **kwargs):
        """Add text annotation to chart."""
        if self.fig is None:
            self.fig = go.Figure(data=self.traces)

        self.fig.add_annotation(
            text=text,
            x=x,
            y=y,
            **kwargs
        )

    def save_html(self, filepath: str):
        """Save chart as interactive HTML file."""
        if self.fig is None:
            self.build()

        self.fig.write_html(filepath)

    def save_image(self, filepath: str, format: str = 'png'):
        """Save chart as static image."""
        if self.fig is None:
            self.build()

        self.fig.write_image(filepath, format=format)


class VisualizationSuite:
    """Main visualization suite coordinator."""

    def __init__(self):
        """Initialize visualization suite."""
        self.frontier_3d = EfficientFrontier3D()
        self.correlation_heatmap = InteractiveCorrelationHeatmap()
        self.animated_replay = AnimatedHistoricalReplay()
        self.risk_decomposition = RiskFactorDecomposition()
        self.timeline = PortfolioEvolutionTimeline()
        self.geo_map = GeographicAllocationMap()
        self.custom_builder = CustomChartBuilder()

    def create_dashboard(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """
        Create comprehensive dashboard with multiple visualizations.

        Args:
            portfolio_data: Dictionary containing all portfolio data

        Returns:
            Dashboard figure with subplots
        """
        # Create subplot grid
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Portfolio Value Over Time',
                'Asset Allocation',
                'Risk Metrics',
                'Geographic Distribution',
                'Performance vs Benchmark',
                'Correlation Matrix'
            ],
            specs=[
                [{'type': 'scatter'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'geo'}],
                [{'type': 'scatter'}, {'type': 'heatmap'}]
            ]
        )

        # Add traces for each subplot
        # (Implementation would add specific traces based on portfolio_data)

        fig.update_layout(
            height=1200,
            width=1600,
            showlegend=True,
            title_text="Portfolio Dashboard"
        )

        return fig
