"""
Scenario Analysis Framework

This module provides comprehensive scenario analysis capabilities for
portfolio evaluation under various market conditions and hypothetical scenarios.

Key Features:
- Historical scenario replay (financial crises, market events)
- Custom scenario builder with flexible parameter specification
- Economic scenario generators (recession, inflation, growth cycles)
- What-if analysis tools for portfolio stress testing
- Sensitivity analysis for parameter perturbations
- Parametric scenario testing with multiple variables
- Scenario comparison framework with visualization

Classes:
    HistoricalScenarioReplay: Replay actual historical market scenarios
    CustomScenarioBuilder: Create user-defined scenarios
    EconomicScenarioGenerator: Generate macro-economic scenarios
    WhatIfAnalyzer: Perform what-if analysis on portfolios
    SensitivityAnalyzer: Analyze portfolio sensitivity to parameters
    ParametricScenarioTester: Test scenarios with parameter variations
    ScenarioComparisonFramework: Compare and analyze multiple scenarios
"""

from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from dataclasses import dataclass


class HistoricalEvent(str, Enum):
    """Pre-defined historical market events."""
    BLACK_MONDAY_1987 = "black_monday_1987"
    DOT_COM_CRASH = "dot_com_crash"
    FINANCIAL_CRISIS_2008 = "financial_crisis_2008"
    EUROPEAN_DEBT_CRISIS = "european_debt_crisis"
    COVID_PANDEMIC = "covid_pandemic"
    FLASH_CRASH_2010 = "flash_crash_2010"
    TAPER_TANTRUM = "taper_tantrum"
    BREXIT = "brexit"
    VOLMAGEDDON = "volmageddon"
    OIL_CRISIS_2020 = "oil_crisis_2020"


class EconomicScenarioType(str, Enum):
    """Types of economic scenarios."""
    RECESSION = "recession"
    RECOVERY = "recovery"
    INFLATION = "inflation"
    DEFLATION = "deflation"
    STAGFLATION = "stagflation"
    GROWTH = "growth"
    CRISIS = "crisis"
    BOOM = "boom"


class ScenarioParameter(BaseModel):
    """Parameter specification for scenario definition."""
    name: str = Field(description="Parameter name")
    base_value: float = Field(description="Base/current value")
    scenario_value: float = Field(description="Value under scenario")
    transition_type: str = Field("immediate", description="immediate, linear, exponential")
    transition_periods: int = Field(1, description="Number of periods for transition")

    class Config:
        use_enum_values = True


@dataclass
class ScenarioImpact:
    """Results of applying a scenario to a portfolio."""
    scenario_name: str
    initial_value: float
    final_value: float
    absolute_change: float
    percentage_change: float
    asset_impacts: Dict[str, float]
    time_series: Optional[pd.Series] = None
    risk_metrics: Optional[Dict[str, float]] = None


class HistoricalScenarioReplay:
    """
    Replay historical market scenarios for stress testing.

    Applies actual historical market movements to current portfolio
    to understand potential impact of similar events.
    """

    def __init__(self):
        """Initialize with historical scenario definitions."""
        self.scenarios = self._define_historical_scenarios()

    def _define_historical_scenarios(self) -> Dict[HistoricalEvent, Dict]:
        """Define parameters for historical scenarios."""
        return {
            HistoricalEvent.BLACK_MONDAY_1987: {
                'description': 'October 19, 1987 - Largest single-day market crash',
                'duration_days': 1,
                'asset_returns': {
                    'equity_us': -0.2079,  # -20.79% S&P 500
                    'equity_intl': -0.1500,
                    'bonds': 0.0050,
                    'gold': 0.0200,
                    'cash': 0.0000,
                },
                'volatility_multiplier': 4.0,
                'correlation_increase': 0.30,
            },
            HistoricalEvent.FINANCIAL_CRISIS_2008: {
                'description': 'September 2008 - March 2009 - Global Financial Crisis',
                'duration_days': 180,
                'asset_returns': {
                    'equity_us': -0.5678,  # -56.78% peak to trough
                    'equity_intl': -0.6100,
                    'equity_emerging': -0.6500,
                    'bonds': 0.0520,
                    'corporate_bonds': -0.2100,
                    'real_estate': -0.6800,
                    'commodities': -0.5400,
                    'gold': 0.2500,
                    'cash': 0.0010,
                },
                'volatility_multiplier': 3.5,
                'correlation_increase': 0.40,
                'credit_spread_widening': 0.0450,  # 450 bps
            },
            HistoricalEvent.COVID_PANDEMIC: {
                'description': 'February-March 2020 - COVID-19 Market Crash',
                'duration_days': 33,
                'asset_returns': {
                    'equity_us': -0.3395,  # -33.95% S&P 500
                    'equity_intl': -0.3500,
                    'equity_emerging': -0.3700,
                    'bonds': 0.0300,
                    'corporate_bonds': -0.1500,
                    'real_estate': -0.4200,
                    'commodities': -0.4000,
                    'oil': -0.6500,
                    'gold': 0.0100,
                    'bitcoin': -0.5000,
                    'cash': 0.0000,
                },
                'volatility_multiplier': 4.5,
                'correlation_increase': 0.50,
            },
            HistoricalEvent.DOT_COM_CRASH: {
                'description': 'March 2000 - October 2002 - Technology Bubble Burst',
                'duration_days': 950,
                'asset_returns': {
                    'equity_tech': -0.7800,  # -78% NASDAQ
                    'equity_us': -0.4900,
                    'equity_value': -0.2200,
                    'bonds': 0.2100,
                    'real_estate': 0.1500,
                    'gold': 0.1200,
                    'cash': 0.0500,
                },
                'volatility_multiplier': 2.0,
                'correlation_increase': 0.20,
            },
            HistoricalEvent.FLASH_CRASH_2010: {
                'description': 'May 6, 2010 - Intraday flash crash',
                'duration_days': 0.01,  # Intraday event
                'asset_returns': {
                    'equity_us': -0.0900,  # -9% intraday
                    'equity_intl': -0.0600,
                    'bonds': 0.0100,
                    'cash': 0.0000,
                },
                'volatility_multiplier': 10.0,  # Extreme short-term vol
                'correlation_increase': 0.60,
            },
        }

    def replay_scenario(
        self,
        portfolio_weights: pd.Series,
        scenario: HistoricalEvent,
        scale_factor: float = 1.0,
    ) -> ScenarioImpact:
        """
        Apply historical scenario to portfolio.

        Args:
            portfolio_weights: Current portfolio allocation
            scenario: Historical event to replay
            scale_factor: Scale the impact (1.0 = full historical impact)

        Returns:
            ScenarioImpact with results
        """
        scenario_data = self.scenarios[scenario]

        # Map portfolio assets to scenario impacts
        asset_impacts = {}
        total_impact = 0

        for asset, weight in portfolio_weights.items():
            # Find matching scenario return
            asset_return = self._map_asset_to_scenario_return(
                asset, scenario_data['asset_returns']
            )
            scaled_return = asset_return * scale_factor

            impact = weight * scaled_return
            asset_impacts[asset] = scaled_return
            total_impact += impact

        initial_value = 1.0  # Normalized
        final_value = initial_value * (1 + total_impact)

        return ScenarioImpact(
            scenario_name=f"{scenario.value} (scale={scale_factor})",
            initial_value=initial_value,
            final_value=final_value,
            absolute_change=final_value - initial_value,
            percentage_change=total_impact,
            asset_impacts=asset_impacts,
            risk_metrics={
                'volatility_multiplier': scenario_data['volatility_multiplier'],
                'correlation_increase': scenario_data['correlation_increase'],
                'duration_days': scenario_data['duration_days'],
            },
        )

    def _map_asset_to_scenario_return(
        self,
        asset: str,
        scenario_returns: Dict[str, float],
    ) -> float:
        """Map portfolio asset to scenario return category."""
        # Simple mapping logic - can be enhanced
        asset_lower = asset.lower()

        # Direct match
        if asset_lower in scenario_returns:
            return scenario_returns[asset_lower]

        # Category matching
        if 'equity' in asset_lower or 'stock' in asset_lower:
            if 'tech' in asset_lower:
                return scenario_returns.get('equity_tech',
                       scenario_returns.get('equity_us', -0.30))
            elif 'emerging' in asset_lower:
                return scenario_returns.get('equity_emerging',
                       scenario_returns.get('equity_intl', -0.30))
            else:
                return scenario_returns.get('equity_us', -0.30)
        elif 'bond' in asset_lower or 'fixed' in asset_lower:
            if 'corporate' in asset_lower or 'credit' in asset_lower:
                return scenario_returns.get('corporate_bonds',
                       scenario_returns.get('bonds', -0.05))
            else:
                return scenario_returns.get('bonds', 0.02)
        elif 'real_estate' in asset_lower or 'reit' in asset_lower:
            return scenario_returns.get('real_estate', -0.40)
        elif 'commodity' in asset_lower or 'commodities' in asset_lower:
            return scenario_returns.get('commodities', -0.30)
        elif 'gold' in asset_lower:
            return scenario_returns.get('gold', 0.10)
        elif 'cash' in asset_lower or 'money_market' in asset_lower:
            return scenario_returns.get('cash', 0.00)
        elif 'crypto' in asset_lower or 'bitcoin' in asset_lower:
            return scenario_returns.get('bitcoin', -0.50)
        else:
            # Default: assume equity-like behavior
            return scenario_returns.get('equity_us', -0.20)

    def get_scenario_details(self, scenario: HistoricalEvent) -> Dict:
        """Get detailed information about a historical scenario."""
        return self.scenarios[scenario]


class CustomScenarioBuilder:
    """
    Build custom scenarios with user-defined parameters.

    Allows creation of bespoke scenarios with specific asset movements,
    correlations, and market conditions.
    """

    def __init__(self):
        """Initialize custom scenario builder."""
        self.scenarios = {}

    def create_scenario(
        self,
        name: str,
        description: str,
        asset_shocks: Dict[str, float],
        duration_periods: int = 1,
        correlation_matrix: Optional[np.ndarray] = None,
        volatility_scaling: Optional[Dict[str, float]] = None,
        parameters: Optional[List[ScenarioParameter]] = None,
    ) -> Dict:
        """
        Create a custom scenario.

        Args:
            name: Scenario name
            description: Scenario description
            asset_shocks: Expected returns/shocks for each asset
            duration_periods: Number of periods for scenario
            correlation_matrix: Custom correlation structure
            volatility_scaling: Volatility adjustments by asset
            parameters: Additional scenario parameters

        Returns:
            Scenario specification dictionary
        """
        scenario = {
            'name': name,
            'description': description,
            'asset_shocks': asset_shocks,
            'duration_periods': duration_periods,
            'correlation_matrix': correlation_matrix,
            'volatility_scaling': volatility_scaling or {},
            'parameters': parameters or [],
            'created_at': datetime.now(),
        }

        self.scenarios[name] = scenario
        return scenario

    def apply_scenario(
        self,
        portfolio_weights: pd.Series,
        scenario_name: str,
        initial_value: float = 1000000,
    ) -> ScenarioImpact:
        """
        Apply custom scenario to portfolio.

        Args:
            portfolio_weights: Portfolio allocation
            scenario_name: Name of scenario to apply
            initial_value: Initial portfolio value

        Returns:
            ScenarioImpact with results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        scenario = self.scenarios[scenario_name]
        asset_shocks = scenario['asset_shocks']

        # Calculate portfolio impact
        total_impact = 0
        asset_impacts = {}

        for asset, weight in portfolio_weights.items():
            shock = asset_shocks.get(asset, 0.0)
            impact = weight * shock
            total_impact += impact
            asset_impacts[asset] = shock

        final_value = initial_value * (1 + total_impact)

        # Generate time series if multi-period
        time_series = None
        if scenario['duration_periods'] > 1:
            periods = scenario['duration_periods']
            values = np.linspace(initial_value, final_value, periods)
            time_series = pd.Series(values)

        return ScenarioImpact(
            scenario_name=scenario_name,
            initial_value=initial_value,
            final_value=final_value,
            absolute_change=final_value - initial_value,
            percentage_change=total_impact,
            asset_impacts=asset_impacts,
            time_series=time_series,
            risk_metrics={
                'duration_periods': scenario['duration_periods'],
            },
        )

    def combine_scenarios(
        self,
        scenario_names: List[str],
        weights: Optional[List[float]] = None,
        name: str = "combined_scenario",
    ) -> Dict:
        """
        Combine multiple scenarios with weights.

        Args:
            scenario_names: List of scenarios to combine
            weights: Weights for each scenario (equal if None)
            name: Name for combined scenario

        Returns:
            Combined scenario specification
        """
        if weights is None:
            weights = [1.0 / len(scenario_names)] * len(scenario_names)

        combined_shocks = {}
        all_assets = set()

        # Collect all assets
        for scenario_name in scenario_names:
            scenario = self.scenarios[scenario_name]
            all_assets.update(scenario['asset_shocks'].keys())

        # Weighted average of shocks
        for asset in all_assets:
            weighted_shock = 0
            for scenario_name, weight in zip(scenario_names, weights):
                scenario = self.scenarios[scenario_name]
                shock = scenario['asset_shocks'].get(asset, 0.0)
                weighted_shock += weight * shock
            combined_shocks[asset] = weighted_shock

        return self.create_scenario(
            name=name,
            description=f"Combination of: {', '.join(scenario_names)}",
            asset_shocks=combined_shocks,
        )


class EconomicScenarioGenerator:
    """
    Generate economic scenarios based on macro conditions.

    Creates consistent multi-asset scenarios based on economic
    environments like recession, inflation, etc.
    """

    def __init__(self):
        """Initialize economic scenario generator."""
        self.templates = self._define_economic_templates()

    def _define_economic_templates(self) -> Dict[EconomicScenarioType, Dict]:
        """Define templates for economic scenarios."""
        return {
            EconomicScenarioType.RECESSION: {
                'gdp_growth': -0.02,
                'inflation': 0.01,
                'unemployment_change': 0.03,
                'interest_rate_change': -0.02,
                'asset_impacts': {
                    'equity': -0.30,
                    'bonds': 0.10,
                    'real_estate': -0.20,
                    'commodities': -0.25,
                    'gold': 0.15,
                    'cash': 0.00,
                },
                'sector_impacts': {
                    'technology': -0.35,
                    'financials': -0.40,
                    'consumer_discretionary': -0.45,
                    'consumer_staples': -0.15,
                    'utilities': -0.10,
                    'healthcare': -0.20,
                },
            },
            EconomicScenarioType.INFLATION: {
                'gdp_growth': 0.02,
                'inflation': 0.06,
                'unemployment_change': -0.01,
                'interest_rate_change': 0.03,
                'asset_impacts': {
                    'equity': -0.15,
                    'bonds': -0.20,
                    'real_estate': 0.10,
                    'commodities': 0.30,
                    'gold': 0.25,
                    'tips': 0.15,  # Inflation-protected securities
                    'cash': -0.05,  # Real return negative
                },
                'sector_impacts': {
                    'energy': 0.20,
                    'materials': 0.15,
                    'financials': 0.05,
                    'technology': -0.25,
                    'utilities': -0.20,
                },
            },
            EconomicScenarioType.STAGFLATION: {
                'gdp_growth': -0.01,
                'inflation': 0.08,
                'unemployment_change': 0.02,
                'interest_rate_change': 0.01,
                'asset_impacts': {
                    'equity': -0.25,
                    'bonds': -0.15,
                    'real_estate': -0.05,
                    'commodities': 0.20,
                    'gold': 0.30,
                    'cash': -0.07,
                },
                'sector_impacts': {
                    'all_sectors': -0.20,  # Broad negative impact
                },
            },
            EconomicScenarioType.RECOVERY: {
                'gdp_growth': 0.04,
                'inflation': 0.02,
                'unemployment_change': -0.02,
                'interest_rate_change': 0.01,
                'asset_impacts': {
                    'equity': 0.25,
                    'bonds': -0.05,
                    'real_estate': 0.15,
                    'commodities': 0.10,
                    'gold': -0.10,
                    'cash': 0.00,
                },
                'sector_impacts': {
                    'technology': 0.35,
                    'financials': 0.30,
                    'consumer_discretionary': 0.40,
                    'industrials': 0.35,
                },
            },
            EconomicScenarioType.DEFLATION: {
                'gdp_growth': -0.03,
                'inflation': -0.02,
                'unemployment_change': 0.04,
                'interest_rate_change': -0.03,
                'asset_impacts': {
                    'equity': -0.35,
                    'bonds': 0.20,  # Bonds benefit from deflation
                    'real_estate': -0.30,
                    'commodities': -0.40,
                    'gold': -0.05,
                    'cash': 0.02,  # Real return positive
                },
                'sector_impacts': {
                    'all_sectors': -0.25,
                },
            },
        }

    def generate_scenario(
        self,
        scenario_type: EconomicScenarioType,
        severity: float = 1.0,
        duration_quarters: int = 4,
        custom_parameters: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate economic scenario.

        Args:
            scenario_type: Type of economic scenario
            severity: Severity multiplier (1.0 = baseline, 2.0 = severe)
            duration_quarters: Duration in quarters
            custom_parameters: Override default parameters

        Returns:
            Economic scenario specification
        """
        template = self.templates[scenario_type]

        # Apply severity scaling
        scenario = {
            'type': scenario_type,
            'severity': severity,
            'duration_quarters': duration_quarters,
            'macro_indicators': {
                'gdp_growth': template['gdp_growth'] * severity,
                'inflation': template['inflation'] * severity,
                'unemployment_change': template['unemployment_change'] * severity,
                'interest_rate_change': template['interest_rate_change'] * severity,
            },
            'asset_impacts': {
                asset: impact * severity
                for asset, impact in template['asset_impacts'].items()
            },
            'sector_impacts': {
                sector: impact * severity
                for sector, impact in template['sector_impacts'].items()
            },
        }

        # Apply custom parameters
        if custom_parameters:
            scenario.update(custom_parameters)

        return scenario

    def apply_to_portfolio(
        self,
        portfolio_weights: pd.Series,
        scenario_type: EconomicScenarioType,
        severity: float = 1.0,
    ) -> ScenarioImpact:
        """
        Apply economic scenario to portfolio.

        Args:
            portfolio_weights: Portfolio allocation
            scenario_type: Economic scenario type
            severity: Severity multiplier

        Returns:
            ScenarioImpact with results
        """
        scenario = self.generate_scenario(scenario_type, severity)

        # Map portfolio assets to economic impacts
        total_impact = 0
        asset_impacts = {}

        for asset, weight in portfolio_weights.items():
            # Determine asset class
            asset_class = self._classify_asset(asset)
            impact = scenario['asset_impacts'].get(asset_class, -0.10)

            # Check for sector-specific impact
            sector = self._identify_sector(asset)
            if sector in scenario['sector_impacts']:
                impact = scenario['sector_impacts'][sector]

            asset_impacts[asset] = impact
            total_impact += weight * impact

        initial_value = 1.0
        final_value = initial_value * (1 + total_impact)

        return ScenarioImpact(
            scenario_name=f"{scenario_type.value} (severity={severity})",
            initial_value=initial_value,
            final_value=final_value,
            absolute_change=final_value - initial_value,
            percentage_change=total_impact,
            asset_impacts=asset_impacts,
            risk_metrics=scenario['macro_indicators'],
        )

    def _classify_asset(self, asset: str) -> str:
        """Classify asset into broad category."""
        asset_lower = asset.lower()

        if 'equity' in asset_lower or 'stock' in asset_lower:
            return 'equity'
        elif 'bond' in asset_lower or 'fixed' in asset_lower:
            if 'tips' in asset_lower or 'inflation' in asset_lower:
                return 'tips'
            return 'bonds'
        elif 'real_estate' in asset_lower or 'reit' in asset_lower:
            return 'real_estate'
        elif 'commodity' in asset_lower or 'commodities' in asset_lower:
            return 'commodities'
        elif 'gold' in asset_lower:
            return 'gold'
        elif 'cash' in asset_lower:
            return 'cash'
        else:
            return 'equity'  # Default

    def _identify_sector(self, asset: str) -> str:
        """Identify sector for an asset."""
        asset_lower = asset.lower()

        sectors = [
            'technology', 'financials', 'healthcare',
            'consumer_discretionary', 'consumer_staples',
            'energy', 'materials', 'industrials',
            'utilities', 'real_estate', 'communication'
        ]

        for sector in sectors:
            if sector in asset_lower:
                return sector

        return 'general'


class WhatIfAnalyzer:
    """
    Perform what-if analysis on portfolios.

    Tests portfolio behavior under various hypothetical conditions
    and parameter changes.
    """

    def __init__(self):
        """Initialize what-if analyzer."""
        self.analysis_results = []

    def analyze_single_change(
        self,
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        parameter: str,
        change_value: float,
        change_type: str = 'absolute',  # 'absolute' or 'relative'
    ) -> Dict[str, float]:
        """
        Analyze impact of single parameter change.

        Args:
            portfolio_weights: Portfolio allocation
            portfolio_returns: Expected returns
            parameter: Parameter to change (e.g., 'equity_return')
            change_value: Amount of change
            change_type: Type of change (absolute or relative)

        Returns:
            Analysis results dictionary
        """
        base_return = (portfolio_weights * portfolio_returns).sum()

        # Apply change
        modified_returns = portfolio_returns.copy()

        if parameter in portfolio_returns.index:
            if change_type == 'absolute':
                modified_returns[parameter] += change_value
            else:  # relative
                modified_returns[parameter] *= (1 + change_value)

        new_return = (portfolio_weights * modified_returns).sum()

        return {
            'parameter': parameter,
            'change_value': change_value,
            'change_type': change_type,
            'base_return': base_return,
            'new_return': new_return,
            'impact': new_return - base_return,
            'impact_percentage': (new_return - base_return) / abs(base_return) if base_return != 0 else 0,
        }

    def analyze_multiple_changes(
        self,
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        changes: List[Tuple[str, float, str]],
    ) -> Dict[str, Any]:
        """
        Analyze multiple simultaneous parameter changes.

        Args:
            portfolio_weights: Portfolio allocation
            portfolio_returns: Expected returns
            changes: List of (parameter, change_value, change_type) tuples

        Returns:
            Combined analysis results
        """
        base_return = (portfolio_weights * portfolio_returns).sum()
        modified_returns = portfolio_returns.copy()

        # Apply all changes
        for parameter, change_value, change_type in changes:
            if parameter in modified_returns.index:
                if change_type == 'absolute':
                    modified_returns[parameter] += change_value
                else:
                    modified_returns[parameter] *= (1 + change_value)

        new_return = (portfolio_weights * modified_returns).sum()

        # Individual contributions
        individual_impacts = []
        for parameter, change_value, change_type in changes:
            individual_result = self.analyze_single_change(
                portfolio_weights,
                portfolio_returns,
                parameter,
                change_value,
                change_type,
            )
            individual_impacts.append(individual_result['impact'])

        return {
            'changes': changes,
            'base_return': base_return,
            'new_return': new_return,
            'total_impact': new_return - base_return,
            'individual_impacts': individual_impacts,
            'interaction_effect': (new_return - base_return) - sum(individual_impacts),
        }

    def analyze_rebalancing(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        expected_returns: pd.Series,
        transaction_cost: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Analyze impact of rebalancing to target weights.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            expected_returns: Expected asset returns
            transaction_cost: Transaction cost rate

        Returns:
            Rebalancing analysis results
        """
        # Current expected return
        current_return = (current_weights * expected_returns).sum()

        # Target expected return
        target_return = (target_weights * expected_returns).sum()

        # Rebalancing cost
        turnover = (target_weights - current_weights).abs().sum() / 2
        rebalancing_cost = turnover * transaction_cost

        # Net benefit
        net_benefit = target_return - current_return - rebalancing_cost

        return {
            'current_return': current_return,
            'target_return': target_return,
            'gross_benefit': target_return - current_return,
            'turnover': turnover,
            'transaction_cost': rebalancing_cost,
            'net_benefit': net_benefit,
            'breakeven_periods': abs(rebalancing_cost / (target_return - current_return))
                                if target_return != current_return else float('inf'),
        }


class SensitivityAnalyzer:
    """
    Perform sensitivity analysis on portfolio parameters.

    Tests how sensitive portfolio outcomes are to changes in
    input parameters and assumptions.
    """

    def __init__(self):
        """Initialize sensitivity analyzer."""
        self.results = {}

    def analyze_parameter_sensitivity(
        self,
        base_value: float,
        parameter_name: str,
        parameter_range: Tuple[float, float],
        n_steps: int,
        evaluation_function: Callable,
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to single parameter.

        Args:
            base_value: Base value of parameter
            parameter_name: Name of parameter
            parameter_range: (min, max) range to test
            n_steps: Number of steps in range
            evaluation_function: Function to evaluate portfolio

        Returns:
            DataFrame with sensitivity results
        """
        param_values = np.linspace(parameter_range[0], parameter_range[1], n_steps)
        results = []

        for param_value in param_values:
            outcome = evaluation_function(param_value)
            results.append({
                'parameter': parameter_name,
                'value': param_value,
                'outcome': outcome,
                'change_from_base': (param_value - base_value) / base_value if base_value != 0 else 0,
            })

        df = pd.DataFrame(results)

        # Calculate sensitivity metrics
        df['outcome_change'] = (df['outcome'] - df['outcome'].iloc[n_steps // 2]) / df['outcome'].iloc[n_steps // 2]

        # Linear sensitivity (slope)
        if len(df) > 1:
            sensitivity = np.polyfit(df['value'], df['outcome'], 1)[0]
        else:
            sensitivity = 0

        self.results[parameter_name] = {
            'data': df,
            'sensitivity': sensitivity,
            'base_value': base_value,
        }

        return df

    def analyze_multi_parameter_sensitivity(
        self,
        parameters: Dict[str, Tuple[float, float]],
        n_steps: int,
        evaluation_function: Callable,
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to multiple parameters.

        Args:
            parameters: Dict of parameter_name: (min, max) ranges
            n_steps: Steps per parameter
            evaluation_function: Function taking dict of parameters

        Returns:
            DataFrame with multi-parameter sensitivity results
        """
        from itertools import product

        # Create parameter grid
        param_grids = {}
        for param_name, (param_min, param_max) in parameters.items():
            param_grids[param_name] = np.linspace(param_min, param_max, n_steps)

        # Generate all combinations
        param_names = list(parameters.keys())
        param_values = [param_grids[name] for name in param_names]
        combinations = list(product(*param_values))

        results = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            outcome = evaluation_function(param_dict)

            result = {'outcome': outcome}
            result.update(param_dict)
            results.append(result)

        return pd.DataFrame(results)

    def calculate_tornado_chart_data(
        self,
        base_outcome: float,
        sensitivities: Dict[str, float],
        parameter_ranges: Dict[str, Tuple[float, float]],
    ) -> pd.DataFrame:
        """
        Prepare data for tornado chart visualization.

        Args:
            base_outcome: Base case outcome
            sensitivities: Sensitivity coefficients by parameter
            parameter_ranges: Parameter ranges tested

        Returns:
            DataFrame formatted for tornado chart
        """
        tornado_data = []

        for param, sensitivity in sensitivities.items():
            param_min, param_max = parameter_ranges[param]

            # Calculate outcome range
            outcome_min = base_outcome + sensitivity * (param_min - (param_min + param_max) / 2)
            outcome_max = base_outcome + sensitivity * (param_max - (param_min + param_max) / 2)

            tornado_data.append({
                'parameter': param,
                'outcome_min': outcome_min,
                'outcome_max': outcome_max,
                'range': abs(outcome_max - outcome_min),
                'sensitivity': sensitivity,
            })

        df = pd.DataFrame(tornado_data)
        df = df.sort_values('range', ascending=False)

        return df


class ParametricScenarioTester:
    """
    Test scenarios with systematic parameter variations.

    Enables comprehensive testing across parameter spaces
    to understand portfolio behavior under various conditions.
    """

    def __init__(self):
        """Initialize parametric scenario tester."""
        self.test_results = []

    def create_parameter_grid(
        self,
        parameters: Dict[str, List[float]],
    ) -> List[Dict[str, float]]:
        """
        Create grid of parameter combinations.

        Args:
            parameters: Dict of parameter_name: list of values

        Returns:
            List of parameter combination dictionaries
        """
        from itertools import product

        param_names = list(parameters.keys())
        param_values = [parameters[name] for name in param_names]
        combinations = list(product(*param_values))

        return [dict(zip(param_names, combo)) for combo in combinations]

    def run_parametric_test(
        self,
        portfolio_function: Callable,
        parameters: Dict[str, List[float]],
        metrics_to_calculate: List[str] = None,
    ) -> pd.DataFrame:
        """
        Run parametric scenario test.

        Args:
            portfolio_function: Function to evaluate portfolio
            parameters: Parameter grid specification
            metrics_to_calculate: List of metrics to calculate

        Returns:
            DataFrame with test results
        """
        if metrics_to_calculate is None:
            metrics_to_calculate = ['return', 'volatility', 'sharpe']

        param_grid = self.create_parameter_grid(parameters)
        results = []

        for params in param_grid:
            # Evaluate portfolio
            portfolio_result = portfolio_function(**params)

            # Extract metrics
            result_dict = params.copy()

            if isinstance(portfolio_result, dict):
                for metric in metrics_to_calculate:
                    result_dict[metric] = portfolio_result.get(metric, np.nan)
            else:
                result_dict['outcome'] = portfolio_result

            results.append(result_dict)

        return pd.DataFrame(results)

    def find_optimal_parameters(
        self,
        test_results: pd.DataFrame,
        objective_metric: str,
        constraints: Optional[List[Callable]] = None,
        maximize: bool = True,
    ) -> Dict[str, Any]:
        """
        Find optimal parameters from test results.

        Args:
            test_results: DataFrame from parametric test
            objective_metric: Metric to optimize
            constraints: List of constraint functions
            maximize: Whether to maximize or minimize objective

        Returns:
            Optimal parameters and outcome
        """
        # Apply constraints
        filtered_results = test_results.copy()

        if constraints:
            for constraint in constraints:
                mask = filtered_results.apply(constraint, axis=1)
                filtered_results = filtered_results[mask]

        if filtered_results.empty:
            return {'status': 'infeasible', 'message': 'No feasible solutions found'}

        # Find optimum
        if maximize:
            optimal_idx = filtered_results[objective_metric].idxmax()
        else:
            optimal_idx = filtered_results[objective_metric].idxmin()

        optimal_row = filtered_results.loc[optimal_idx]

        # Extract parameters
        param_columns = [col for col in optimal_row.index
                        if col not in ['return', 'volatility', 'sharpe', 'outcome']
                        and not col.startswith('_')]

        return {
            'status': 'optimal',
            'parameters': optimal_row[param_columns].to_dict(),
            'objective_value': optimal_row[objective_metric],
            'all_metrics': optimal_row.to_dict(),
        }


class ScenarioComparisonFramework:
    """
    Framework for comparing and analyzing multiple scenarios.

    Provides tools to compare scenario impacts, rank scenarios,
    and identify robust portfolio strategies.
    """

    def __init__(self):
        """Initialize scenario comparison framework."""
        self.scenarios = {}
        self.comparison_results = None

    def add_scenario(
        self,
        name: str,
        impact: ScenarioImpact,
        probability: float = None,
    ):
        """
        Add scenario to comparison framework.

        Args:
            name: Scenario name
            impact: ScenarioImpact object
            probability: Scenario probability (for weighted analysis)
        """
        self.scenarios[name] = {
            'impact': impact,
            'probability': probability or 1.0 / len(self.scenarios),
        }

    def compare_scenarios(self) -> pd.DataFrame:
        """
        Create comparison table of all scenarios.

        Returns:
            DataFrame with scenario comparisons
        """
        comparison_data = []

        for name, scenario_data in self.scenarios.items():
            impact = scenario_data['impact']
            comparison_data.append({
                'scenario': name,
                'initial_value': impact.initial_value,
                'final_value': impact.final_value,
                'percentage_change': impact.percentage_change,
                'probability': scenario_data['probability'],
                'expected_impact': impact.percentage_change * scenario_data['probability'],
            })

        self.comparison_results = pd.DataFrame(comparison_data)
        return self.comparison_results

    def calculate_portfolio_statistics(self) -> Dict[str, float]:
        """
        Calculate portfolio statistics across scenarios.

        Returns:
            Dictionary with portfolio statistics
        """
        if self.comparison_results is None:
            self.compare_scenarios()

        returns = self.comparison_results['percentage_change'].values
        probabilities = self.comparison_results['probability'].values

        # Expected return
        expected_return = np.sum(returns * probabilities)

        # Variance
        variance = np.sum(probabilities * (returns - expected_return) ** 2)
        std_dev = np.sqrt(variance)

        # Downside risk
        downside_returns = np.minimum(returns, 0)
        downside_variance = np.sum(probabilities * downside_returns ** 2)
        downside_deviation = np.sqrt(downside_variance)

        # Value at Risk (5%)
        sorted_returns = np.sort(returns)
        cumulative_prob = np.cumsum(np.sort(probabilities))
        var_idx = np.searchsorted(cumulative_prob, 0.05)
        var_5 = sorted_returns[var_idx] if var_idx < len(sorted_returns) else sorted_returns[0]

        # Conditional Value at Risk
        cvar_5 = np.mean(sorted_returns[:max(1, var_idx)])

        return {
            'expected_return': expected_return,
            'std_deviation': std_dev,
            'downside_deviation': downside_deviation,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'best_scenario': returns.max(),
            'worst_scenario': returns.min(),
            'scenario_range': returns.max() - returns.min(),
        }

    def rank_scenarios(
        self,
        ranking_metric: str = 'percentage_change',
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Rank scenarios by specified metric.

        Args:
            ranking_metric: Metric to rank by
            ascending: Sort order

        Returns:
            Ranked DataFrame
        """
        if self.comparison_results is None:
            self.compare_scenarios()

        ranked = self.comparison_results.sort_values(
            ranking_metric, ascending=ascending
        ).reset_index(drop=True)

        ranked['rank'] = range(1, len(ranked) + 1)

        return ranked

    def identify_robust_allocation(
        self,
        portfolio_options: Dict[str, pd.Series],
    ) -> str:
        """
        Identify most robust portfolio across scenarios.

        Args:
            portfolio_options: Dict of portfolio_name: weights

        Returns:
            Name of most robust portfolio
        """
        robustness_scores = {}

        for portfolio_name, weights in portfolio_options.items():
            # Calculate performance across all scenarios
            performances = []

            for scenario_name, scenario_data in self.scenarios.items():
                impact = scenario_data['impact']
                # Simple return calculation
                portfolio_return = sum(
                    weights.get(asset, 0) * asset_return
                    for asset, asset_return in impact.asset_impacts.items()
                )
                performances.append(portfolio_return)

            # Robustness metrics
            mean_return = np.mean(performances)
            worst_case = np.min(performances)
            std_dev = np.std(performances)

            # Robustness score (higher is better)
            # Combines average return, worst case, and stability
            if std_dev > 0:
                robustness_score = mean_return - 2 * std_dev + 0.5 * worst_case
            else:
                robustness_score = mean_return + 0.5 * worst_case

            robustness_scores[portfolio_name] = robustness_score

        # Find most robust
        most_robust = max(robustness_scores, key=robustness_scores.get)

        return most_robust