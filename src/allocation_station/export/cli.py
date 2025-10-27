"""Command-Line Interface for Allocation Station."""

import click
import json
from pathlib import Path


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Allocation Station - Portfolio Analysis Framework CLI."""
    pass


@cli.command()
@click.argument('portfolio_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file path')
def analyze(portfolio_file, output):
    """Analyze a portfolio from file."""
    click.echo(f"Analyzing portfolio: {portfolio_file}")

    # Load portfolio
    with open(portfolio_file) as f:
        portfolio_data = json.load(f)

    # Perform analysis
    result = {
        'total_value': sum(portfolio_data.get('holdings', {}).values()),
        'num_holdings': len(portfolio_data.get('holdings', {})),
        'analysis_date': str(pd.Timestamp.now())
    }

    if output:
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        click.echo(f"Results saved to: {output}")
    else:
        click.echo(json.dumps(result, indent=2))


@cli.command()
@click.option('--host', default='127.0.0.1', help='Server host')
@click.option('--port', default=8000, help='Server port')
def serve(host, port):
    """Start the REST API server."""
    click.echo(f"Starting server on {host}:{port}")
    # Would start FastAPI server here


@cli.command()
@click.argument('portfolio_file')
@click.option('--format', type=click.Choice(['pdf', 'html', 'excel']), default='pdf')
@click.option('--output', '-o', required=True)
def report(portfolio_file, format, output):
    """Generate a portfolio report."""
    click.echo(f"Generating {format} report from {portfolio_file}")
    click.echo(f"Output: {output}")


@cli.command()
def version():
    """Show version information."""
    click.echo("Allocation Station v1.0.0")


class AllocationStationCLI:
    """CLI wrapper class."""

    @staticmethod
    def run():
        """Run the CLI."""
        cli()
