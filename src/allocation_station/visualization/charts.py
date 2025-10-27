"""Charting functions for portfolio visualization."""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_portfolio_performance(
    portfolio_values: pd.Series,
    benchmark_values: Optional[pd.Series] = None,
    title: str = "Portfolio Performance",
    use_plotly: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot portfolio performance over time.

    Args:
        portfolio_values: Time series of portfolio values
        benchmark_values: Optional benchmark values for comparison
        title: Chart title
        use_plotly: Use plotly (True) or matplotlib (False)

    Returns:
        Figure object
    """
    if use_plotly:
        fig = go.Figure()

        # Portfolio line
        fig.add_trace(go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))

        # Benchmark line
        if benchmark_values is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_values.index,
                y=benchmark_values.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=2, dash='dash')
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            showlegend=True,
            template="plotly_white"
        )

        return fig

    else:
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(portfolio_values.index, portfolio_values.values,
                label='Portfolio', linewidth=2)

        if benchmark_values is not None:
            ax.plot(benchmark_values.index, benchmark_values.values,
                    label='Benchmark', linestyle='--', alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


def plot_efficient_frontier(
    frontier_points: pd.DataFrame,
    optimal_portfolio: Optional[Dict[str, float]] = None,
    current_portfolio: Optional[Dict[str, float]] = None,
    title: str = "Efficient Frontier",
    use_plotly: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot efficient frontier with portfolio positions.

    Args:
        frontier_points: DataFrame with return/volatility points
        optimal_portfolio: Optional optimal portfolio to highlight
        current_portfolio: Optional current portfolio to show
        title: Chart title
        use_plotly: Use plotly or matplotlib

    Returns:
        Figure object
    """
    if use_plotly:
        fig = go.Figure()

        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_points['volatility'],
            y=frontier_points['return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2)
        ))

        # Color by Sharpe ratio if available
        if 'sharpe_ratio' in frontier_points.columns:
            fig.add_trace(go.Scatter(
                x=frontier_points['volatility'],
                y=frontier_points['return'],
                mode='markers',
                marker=dict(
                    color=frontier_points['sharpe_ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    size=8,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Frontier Points',
                showlegend=False
            ))

        # Optimal portfolio
        if optimal_portfolio:
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio.get('volatility', 0)],
                y=[optimal_portfolio.get('return', 0)],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(color='red', size=15, symbol='star')
            ))

        # Current portfolio
        if current_portfolio:
            fig.add_trace(go.Scatter(
                x=[current_portfolio.get('volatility', 0)],
                y=[current_portfolio.get('return', 0)],
                mode='markers',
                name='Current Portfolio',
                marker=dict(color='green', size=12, symbol='diamond')
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            hovermode='closest',
            showlegend=True,
            template="plotly_white"
        )

        return fig

    else:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot frontier
        ax.plot(frontier_points['volatility'], frontier_points['return'],
                'b-', linewidth=2, label='Efficient Frontier')

        # Color points by Sharpe if available
        if 'sharpe_ratio' in frontier_points.columns:
            scatter = ax.scatter(frontier_points['volatility'],
                                  frontier_points['return'],
                                  c=frontier_points['sharpe_ratio'],
                                  cmap='viridis', s=30)
            plt.colorbar(scatter, label='Sharpe Ratio')

        # Plot portfolios
        if optimal_portfolio:
            ax.scatter(optimal_portfolio.get('volatility', 0),
                       optimal_portfolio.get('return', 0),
                       color='red', s=200, marker='*', label='Optimal Portfolio')

        if current_portfolio:
            ax.scatter(current_portfolio.get('volatility', 0),
                       current_portfolio.get('return', 0),
                       color='green', s=150, marker='D', label='Current Portfolio')

        ax.set_title(title)
        ax.set_xlabel('Volatility (Risk)')
        ax.set_ylabel('Expected Return')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


def plot_allocation_pie(
    allocation: Dict[str, float],
    title: str = "Portfolio Allocation",
    use_plotly: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot portfolio allocation as pie chart.

    Args:
        allocation: Dictionary of asset weights
        title: Chart title
        use_plotly: Use plotly or matplotlib

    Returns:
        Figure object
    """
    labels = list(allocation.keys())
    values = list(allocation.values())

    if use_plotly:
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,  # Donut chart
            textposition='auto',
            textinfo='label+percent',
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='white', width=2)
            )
        )])

        fig.update_layout(
            title=title,
            showlegend=True,
            template="plotly_white"
        )

        return fig

    else:
        fig, ax = plt.subplots(figsize=(10, 8))

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Set3.colors
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title(title)

        return fig


def plot_monte_carlo_paths(
    simulation_results: Any,  # SimulationResults object
    percentiles: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
    n_sample_paths: int = 100,
    title: str = "Monte Carlo Simulation Results",
    use_plotly: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot Monte Carlo simulation paths with percentiles.

    Args:
        simulation_results: Results from Monte Carlo simulation
        percentiles: Percentiles to highlight
        n_sample_paths: Number of sample paths to show
        title: Chart title
        use_plotly: Use plotly or matplotlib

    Returns:
        Figure object
    """
    paths = simulation_results.portfolio_values
    n_periods = paths.shape[1]
    time_axis = np.arange(n_periods)

    if use_plotly:
        fig = go.Figure()

        # Sample paths
        sample_indices = np.random.choice(paths.shape[0], n_sample_paths, replace=False)
        for i in sample_indices:
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=paths[i],
                mode='lines',
                line=dict(color='lightgray', width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Percentile paths
        colors = ['red', 'orange', 'green', 'orange', 'red']
        for percentile, color in zip(percentiles, colors):
            percentile_path = np.percentile(paths, percentile * 100, axis=0)
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=percentile_path,
                mode='lines',
                name=f'{percentile*100:.0f}th percentile',
                line=dict(color=color, width=2)
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            showlegend=True,
            template="plotly_white"
        )

        return fig

    else:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Sample paths
        sample_indices = np.random.choice(paths.shape[0], n_sample_paths, replace=False)
        for i in sample_indices:
            ax.plot(time_axis, paths[i], color='lightgray', alpha=0.3, linewidth=0.5)

        # Percentile paths
        colors = ['red', 'orange', 'green', 'orange', 'red']
        for percentile, color in zip(percentiles, colors):
            percentile_path = np.percentile(paths, percentile * 100, axis=0)
            ax.plot(time_axis, percentile_path, color=color, linewidth=2,
                    label=f'{percentile*100:.0f}th percentile')

        ax.set_title(title)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


def plot_drawdown(
    portfolio_values: pd.Series,
    title: str = "Portfolio Drawdown",
    use_plotly: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot portfolio drawdown over time.

    Args:
        portfolio_values: Time series of portfolio values
        title: Chart title
        use_plotly: Use plotly or matplotlib

    Returns:
        Figure object
    """
    # Calculate drawdown
    cumulative = portfolio_values / portfolio_values.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100

    if use_plotly:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Portfolio Value', 'Drawdown %'),
            row_heights=[0.6, 0.4]
        )

        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red', width=1),
                fillcolor='rgba(255, 0, 0, 0.2)'
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        fig.update_layout(
            title=title,
            hovermode='x unified',
            showlegend=False,
            template="plotly_white"
        )

        return fig

    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Portfolio value
        ax1.plot(portfolio_values.index, portfolio_values.values,
                 color='blue', linewidth=2)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2.fill_between(drawdown.index, 0, drawdown.values,
                          color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values,
                 color='red', linewidth=1)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def plot_returns_distribution(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Returns Distribution",
    use_plotly: bool = True
) -> Union[go.Figure, plt.Figure]:
    """
    Plot returns distribution histogram with statistics.

    Args:
        returns: Series of returns
        benchmark_returns: Optional benchmark returns
        title: Chart title
        use_plotly: Use plotly or matplotlib

    Returns:
        Figure object
    """
    # Calculate statistics
    mean_return = returns.mean()
    std_return = returns.std()
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    if use_plotly:
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=returns.values,
            name='Portfolio Returns',
            opacity=0.7,
            marker_color='blue',
            nbinsx=50,
            histnorm='probability'
        ))

        if benchmark_returns is not None:
            fig.add_trace(go.Histogram(
                x=benchmark_returns.values,
                name='Benchmark Returns',
                opacity=0.5,
                marker_color='gray',
                nbinsx=50,
                histnorm='probability'
            ))

        # Add normal distribution overlay
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = (1 / (std_return * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)
        normal_dist = normal_dist / normal_dist.max() * returns.value_counts().max() / len(returns)

        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2, dash='dash')
        ))

        # Add statistics annotation
        stats_text = f"Mean: {mean_return:.4f}<br>" \
                     f"Std: {std_return:.4f}<br>" \
                     f"Skew: {skewness:.2f}<br>" \
                     f"Kurt: {kurtosis:.2f}"

        fig.add_annotation(
            x=0.95, y=0.95,
            xref="paper", yref="paper",
            text=stats_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )

        fig.update_layout(
            title=title,
            xaxis_title="Returns",
            yaxis_title="Frequency",
            barmode='overlay',
            showlegend=True,
            template="plotly_white"
        )

        return fig

    else:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(returns.values, bins=50, alpha=0.7, color='blue',
                edgecolor='black', density=True, label='Portfolio Returns')

        if benchmark_returns is not None:
            ax.hist(benchmark_returns.values, bins=50, alpha=0.5, color='gray',
                    edgecolor='black', density=True, label='Benchmark Returns')

        # Normal distribution overlay
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = (1 / (std_return * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)

        ax.plot(x_range, normal_dist, 'r--', linewidth=2, label='Normal Distribution')

        # Add statistics
        stats_text = f'Mean: {mean_return:.4f}\\n' \
                     f'Std: {std_return:.4f}\\n' \
                     f'Skew: {skewness:.2f}\\n' \
                     f'Kurt: {kurtosis:.2f}'

        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(title)
        ax.set_xlabel('Returns')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig