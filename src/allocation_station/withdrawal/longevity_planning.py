"""
Longevity Planning for Retirement

This module provides comprehensive longevity planning tools including mortality tables,
life expectancy calculations, longevity risk modeling, couple strategies, healthcare
cost projections, long-term care planning, legacy planning, and charitable giving.

Key Features:
- Mortality tables and life expectancy calculations (SSA, IRS tables)
- Longevity risk modeling with Monte Carlo simulation
- Couple and joint life strategies
- Healthcare cost projections with inflation
- Long-term care planning and insurance evaluation
- Legacy planning and wealth transfer optimization
- Charitable giving strategies (QCDs, DAFs, CLTs, CRTs)

Classes:
    MortalityTable: Actuarial life expectancy and survival probabilities
    LongevityRiskModeler: Monte Carlo simulation of longevity scenarios
    CoupleLifePlanner: Joint life planning for couples
    HealthcareCostProjector: Medical expense projections with inflation
    LongTermCarePlanner: LTC needs assessment and insurance evaluation
    LegacyPlanner: Estate planning and wealth transfer optimization
    CharitableGivingStrategy: Tax-efficient charitable giving strategies
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class Gender(str, Enum):
    """Gender for mortality calculations."""
    MALE = "male"
    FEMALE = "female"


class HealthStatus(str, Enum):
    """Health status categories."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"


class LTCTrigger(str, Enum):
    """Long-term care benefit triggers."""
    ADL_2_OF_6 = "adl_2_of_6"  # Cannot perform 2 of 6 Activities of Daily Living
    ADL_3_OF_6 = "adl_3_of_6"
    COGNITIVE_IMPAIRMENT = "cognitive_impairment"


class CharitableVehicle(str, Enum):
    """Charitable giving vehicles."""
    QCD = "qcd"  # Qualified Charitable Distribution
    DAF = "daf"  # Donor Advised Fund
    CLT = "clt"  # Charitable Lead Trust
    CRT = "crt"  # Charitable Remainder Trust
    DIRECT = "direct"  # Direct donation


@dataclass
class LifeExpectancy:
    """Life expectancy calculation results."""
    current_age: int
    life_expectancy: float
    age_at_death: float
    probability_age_90: float
    probability_age_95: float
    probability_age_100: float
    percentile_25: float  # 25th percentile (die earlier)
    percentile_50: float  # Median
    percentile_75: float  # 75th percentile (live longer)


class MortalityTable:
    """
    Actuarial mortality tables and life expectancy calculations.

    Implements Social Security Administration and IRS mortality tables
    for retirement planning and longevity analysis.
    """

    # SSA 2020 Period Life Table (simplified, per 1000)
    # Format: {age: {'male': death_rate, 'female': death_rate}}
    SSA_MORTALITY = {
        60: {'male': 7.79, 'female': 4.72},
        65: {'male': 12.30, 'female': 7.45},
        70: {'male': 19.98, 'female': 12.51},
        75: {'male': 32.87, 'female': 21.12},
        80: {'male': 54.95, 'female': 36.77},
        85: {'male': 90.24, 'female': 64.26},
        90: {'male': 142.60, 'female': 110.70},
        95: {'male': 214.68, 'female': 183.70},
        100: {'male': 303.06, 'female': 283.48},
    }

    def __init__(self):
        """Initialize mortality table."""
        pass

    def get_mortality_rate(self, age: int, gender: Gender) -> float:
        """
        Get annual mortality rate (probability of death) for given age/gender.

        Args:
            age: Current age
            gender: Gender

        Returns:
            Annual probability of death (0 to 1)
        """
        # Find closest age in table
        ages = sorted(self.SSA_MORTALITY.keys())
        closest_age = min(ages, key=lambda x: abs(x - age))

        # Get rate per 1000
        rate_per_1000 = self.SSA_MORTALITY[closest_age][gender.value]

        # Convert to probability
        return rate_per_1000 / 1000.0

    def get_survival_probability(
        self,
        current_age: int,
        target_age: int,
        gender: Gender,
    ) -> float:
        """
        Calculate probability of surviving from current age to target age.

        Args:
            current_age: Starting age
            target_age: Age to survive to
            gender: Gender

        Returns:
            Probability of survival (0 to 1)
        """
        survival_prob = 1.0

        for age in range(current_age, target_age):
            mortality_rate = self.get_mortality_rate(age, gender)
            survival_prob *= (1 - mortality_rate)

        return survival_prob

    def calculate_life_expectancy(
        self,
        age: int,
        gender: Gender,
        health_status: HealthStatus = HealthStatus.AVERAGE,
    ) -> LifeExpectancy:
        """
        Calculate comprehensive life expectancy metrics.

        Args:
            age: Current age
            gender: Gender
            health_status: Health status adjustment

        Returns:
            LifeExpectancy with detailed metrics
        """
        # Calculate expected remaining years
        expected_years = 0
        max_age = 120

        for future_age in range(age, max_age):
            survival_prob = self.get_survival_probability(age, future_age, gender)
            expected_years += survival_prob

        # Health adjustment
        health_adjustments = {
            HealthStatus.EXCELLENT: 1.10,  # +10% life expectancy
            HealthStatus.GOOD: 1.03,
            HealthStatus.AVERAGE: 1.00,
            HealthStatus.POOR: 0.90,  # -10% life expectancy
        }

        adjustment = health_adjustments[health_status]
        adjusted_years = expected_years * adjustment

        # Probabilities of reaching certain ages
        prob_90 = self.get_survival_probability(age, 90, gender) * adjustment
        prob_95 = self.get_survival_probability(age, 95, gender) * adjustment
        prob_100 = self.get_survival_probability(age, 100, gender) * adjustment

        # Percentiles (simplified)
        median_age = age + expected_years
        percentile_25 = median_age - 5
        percentile_75 = median_age + 5

        return LifeExpectancy(
            current_age=age,
            life_expectancy=adjusted_years,
            age_at_death=age + adjusted_years,
            probability_age_90=min(prob_90, 1.0),
            probability_age_95=min(prob_95, 1.0),
            probability_age_100=min(prob_100, 1.0),
            percentile_25=percentile_25,
            percentile_50=median_age,
            percentile_75=percentile_75,
        )

    def create_survival_curve(
        self,
        current_age: int,
        gender: Gender,
        years: int = 40,
    ) -> pd.DataFrame:
        """
        Create survival curve showing probability of survival by age.

        Args:
            current_age: Starting age
            gender: Gender
            years: Years to project

        Returns:
            DataFrame with survival probabilities
        """
        ages = list(range(current_age, current_age + years + 1))
        survival_probs = []

        for age in ages:
            prob = self.get_survival_probability(current_age, age, gender)
            survival_probs.append(prob)

        return pd.DataFrame({
            'age': ages,
            'survival_probability': survival_probs,
            'mortality_probability': [1 - p for p in survival_probs],
        })


class LongevityRiskModeler:
    """
    Model longevity risk using Monte Carlo simulation.

    Simulates lifespan scenarios to assess portfolio sustainability
    and longevity-related risks.
    """

    def __init__(
        self,
        mortality_table: MortalityTable,
    ):
        """
        Initialize longevity risk modeler.

        Args:
            mortality_table: Mortality table for calculations
        """
        self.mortality_table = mortality_table

    def simulate_lifespan(
        self,
        age: int,
        gender: Gender,
        n_simulations: int = 10000,
    ) -> np.ndarray:
        """
        Simulate lifespan using Monte Carlo.

        Args:
            age: Current age
            gender: Gender
            n_simulations: Number of simulations

        Returns:
            Array of simulated ages at death
        """
        ages_at_death = np.zeros(n_simulations)

        for sim in range(n_simulations):
            current_sim_age = age

            # Simulate year by year until death
            while current_sim_age < 120:
                mortality_rate = self.mortality_table.get_mortality_rate(current_sim_age, gender)

                # Random draw for survival
                if np.random.random() < mortality_rate:
                    ages_at_death[sim] = current_sim_age
                    break

                current_sim_age += 1

            # If survived to 120
            if ages_at_death[sim] == 0:
                ages_at_death[sim] = 120

        return ages_at_death

    def assess_longevity_risk(
        self,
        age: int,
        gender: Gender,
        portfolio_value: float,
        annual_spending: float,
        return_rate: float = 0.05,
        n_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Assess risk of outliving portfolio.

        Args:
            age: Current age
            gender: Gender
            portfolio_value: Current portfolio value
            annual_spending: Annual spending amount
            return_rate: Expected portfolio return
            n_simulations: Number of simulations

        Returns:
            Risk assessment results
        """
        ages_at_death = self.simulate_lifespan(age, gender, n_simulations)

        # Simulate portfolio for each lifespan scenario
        success_count = 0
        shortfall_amounts = []

        for age_at_death in ages_at_death:
            years = int(age_at_death - age)

            # Simple portfolio projection
            portfolio = portfolio_value

            for year in range(years):
                portfolio -= annual_spending
                if portfolio < 0:
                    shortfall_amounts.append(-portfolio)
                    break
                portfolio *= (1 + return_rate)
            else:
                success_count += 1

        success_rate = success_count / n_simulations

        return {
            'success_rate': success_rate,
            'failure_rate': 1 - success_rate,
            'median_lifespan': np.median(ages_at_death),
            'percentile_90_lifespan': np.percentile(ages_at_death, 90),
            'average_shortfall': np.mean(shortfall_amounts) if shortfall_amounts else 0,
            'max_shortfall': max(shortfall_amounts) if shortfall_amounts else 0,
            'simulated_lifespans': ages_at_death,
        }


class CoupleLifePlanner:
    """
    Joint life planning for couples.

    Analyzes joint survival probabilities and coordinates
    retirement planning for married couples.
    """

    def __init__(
        self,
        mortality_table: MortalityTable,
    ):
        """
        Initialize couple life planner.

        Args:
            mortality_table: Mortality table for calculations
        """
        self.mortality_table = mortality_table

    def calculate_joint_life_expectancy(
        self,
        age1: int,
        gender1: Gender,
        age2: int,
        gender2: Gender,
    ) -> Dict[str, Any]:
        """
        Calculate joint and survivor life expectancies.

        Args:
            age1: First person's age
            gender1: First person's gender
            age2: Second person's age
            gender2: Second person's gender

        Returns:
            Joint life expectancy metrics
        """
        # Individual life expectancies
        le1 = self.mortality_table.calculate_life_expectancy(age1, gender1)
        le2 = self.mortality_table.calculate_life_expectancy(age2, gender2)

        # Probability both survive to various ages
        target_ages = [70, 75, 80, 85, 90, 95]
        joint_survival_probs = {}

        for target in target_ages:
            if target > max(age1, age2):
                prob1 = self.mortality_table.get_survival_probability(age1, target, gender1)
                prob2 = self.mortality_table.get_survival_probability(age2, target, gender2)
                # Probability both survive (assuming independence)
                joint_survival_probs[target] = prob1 * prob2

        # Probability at least one survives
        survivor_probs = {}
        for target in target_ages:
            if target > max(age1, age2):
                prob1 = self.mortality_table.get_survival_probability(age1, target, gender1)
                prob2 = self.mortality_table.get_survival_probability(age2, target, gender2)
                # At least one survives = 1 - (both die)
                survivor_probs[target] = 1 - ((1 - prob1) * (1 - prob2))

        # Expected first death and second death ages
        first_death_age = min(le1.age_at_death, le2.age_at_death)
        second_death_age = max(le1.age_at_death, le2.age_at_death)

        return {
            'person1_life_expectancy': le1,
            'person2_life_expectancy': le2,
            'expected_first_death_age': first_death_age,
            'expected_second_death_age': second_death_age,
            'survivor_years': second_death_age - first_death_age,
            'joint_survival_probabilities': joint_survival_probs,
            'survivor_probabilities': survivor_probs,
        }

    def optimize_couple_strategy(
        self,
        ages: Tuple[int, int],
        genders: Tuple[Gender, Gender],
        portfolio_value: float,
        desired_spending_couple: float,
        desired_spending_survivor: float,
    ) -> Dict[str, Any]:
        """
        Optimize retirement strategy for couple.

        Args:
            ages: (age1, age2)
            genders: (gender1, gender2)
            portfolio_value: Current portfolio value
            desired_spending_couple: Spending when both alive
            desired_spending_survivor: Spending for survivor

        Returns:
            Optimized strategy recommendations
        """
        joint_le = self.calculate_joint_life_expectancy(
            ages[0], genders[0], ages[1], genders[1]
        )

        first_death = joint_le['expected_first_death_age']
        second_death = joint_le['expected_second_death_age']

        # Calculate needed portfolio value
        years_both_alive = first_death - max(ages)
        years_survivor = second_death - first_death

        pv_couple = desired_spending_couple * years_both_alive
        pv_survivor = desired_spending_survivor * years_survivor

        total_need = pv_couple + pv_survivor
        surplus_or_shortfall = portfolio_value - total_need

        return {
            'portfolio_value': portfolio_value,
            'total_spending_need': total_need,
            'surplus_or_shortfall': surplus_or_shortfall,
            'years_both_alive': years_both_alive,
            'years_survivor_alone': years_survivor,
            'spending_couple_phase': pv_couple,
            'spending_survivor_phase': pv_survivor,
            'adequately_funded': surplus_or_shortfall >= 0,
            'recommendations': self._generate_couple_recommendations(
                surplus_or_shortfall,
                joint_le,
            ),
        }

    def _generate_couple_recommendations(
        self,
        surplus_or_shortfall: float,
        joint_le: Dict[str, Any],
    ) -> List[str]:
        """Generate strategy recommendations for couples."""
        recommendations = []

        if surplus_or_shortfall < 0:
            recommendations.append(f"Shortfall of ${abs(surplus_or_shortfall):,.0f} - consider:")
            recommendations.append("- Reduce spending in early retirement years")
            recommendations.append("- Delay Social Security for higher earner to age 70")
            recommendations.append("- Consider life insurance for survivor protection")
        else:
            recommendations.append(f"Surplus of ${surplus_or_shortfall:,.0f}")
            recommendations.append("- Consider legacy planning for heirs")
            recommendations.append("- Evaluate long-term care insurance needs")
            recommendations.append("- Explore charitable giving strategies")

        # Survivor benefit recommendations
        survivor_years = joint_le['survivor_years']
        if survivor_years > 15:
            recommendations.append(f"Long survivor period ({survivor_years:.0f} years) - ensure:")
            recommendations.append("- Adequate survivor Social Security benefit")
            recommendations.append("- Pension survivor option if available")

        return recommendations


class HealthcareCostProjector:
    """
    Project healthcare costs in retirement.

    Estimates Medicare premiums, out-of-pocket costs, and
    long-term care expenses with medical inflation.
    """

    def __init__(
        self,
        medical_inflation: float = 0.05,  # 5% medical inflation
    ):
        """
        Initialize healthcare cost projector.

        Args:
            medical_inflation: Annual medical cost inflation rate
        """
        self.medical_inflation = medical_inflation

    def project_medicare_costs(
        self,
        age: int,
        income: float,
        years: int = 30,
    ) -> pd.DataFrame:
        """
        Project Medicare Part B and D premiums.

        Args:
            age: Current age
            income: Modified Adjusted Gross Income (MAGI)
            years: Years to project

        Returns:
            DataFrame with Medicare cost projections
        """
        # 2024 Medicare Part B standard premium
        part_b_standard = 174.70  # Monthly

        # IRMAA surcharges based on income (2024)
        if income < 103000:
            part_b_irmaa = 0
        elif income < 129000:
            part_b_irmaa = 69.90
        elif income < 161000:
            part_b_irmaa = 174.70
        elif income < 193000:
            part_b_irmaa = 279.50
        elif income < 500000:
            part_b_irmaa = 384.30
        else:
            part_b_irmaa = 419.30

        part_b_monthly = part_b_standard + part_b_irmaa

        # Part D standard (estimated)
        part_d_monthly = 35.00  # Varies by plan

        projections = []

        for year in range(years):
            current_age = age + year

            # Medicare starts at 65
            if current_age >= 65:
                inflation_factor = (1 + self.medical_inflation) ** year

                annual_part_b = part_b_monthly * 12 * inflation_factor
                annual_part_d = part_d_monthly * 12 * inflation_factor

                # Medigap/supplement (estimated)
                medigap = 2000 * inflation_factor

                total_premiums = annual_part_b + annual_part_d + medigap
            else:
                # Pre-Medicare (employer or ACA)
                inflation_factor = (1 + self.medical_inflation) ** year
                total_premiums = 12000 * inflation_factor  # Estimated pre-65 premium

            projections.append({
                'year': year,
                'age': current_age,
                'part_b_premium': annual_part_b if current_age >= 65 else 0,
                'part_d_premium': annual_part_d if current_age >= 65 else 0,
                'supplemental': medigap if current_age >= 65 else 0,
                'pre_medicare': total_premiums if current_age < 65 else 0,
                'total_premiums': total_premiums,
            })

        return pd.DataFrame(projections)

    def estimate_out_of_pocket_costs(
        self,
        age: int,
        health_status: HealthStatus = HealthStatus.AVERAGE,
    ) -> float:
        """
        Estimate annual out-of-pocket medical costs.

        Args:
            age: Current age
            health_status: Health status

        Returns:
            Estimated annual OOP costs
        """
        # Base OOP costs for average 65-year-old
        base_oop = 4500

        # Age adjustment (costs increase with age)
        if age >= 65:
            age_factor = 1 + ((age - 65) * 0.02)  # 2% per year over 65
        else:
            age_factor = 0.8

        # Health status adjustment
        health_factors = {
            HealthStatus.EXCELLENT: 0.7,
            HealthStatus.GOOD: 0.9,
            HealthStatus.AVERAGE: 1.0,
            HealthStatus.POOR: 1.5,
        }

        total_oop = base_oop * age_factor * health_factors[health_status]

        return total_oop

    def project_lifetime_healthcare_costs(
        self,
        current_age: int,
        life_expectancy_age: int,
        income: float,
        health_status: HealthStatus = HealthStatus.AVERAGE,
    ) -> Dict[str, Any]:
        """
        Project total lifetime healthcare costs.

        Args:
            current_age: Current age
            life_expectancy_age: Expected age at death
            income: Annual income (for IRMAA)
            health_status: Health status

        Returns:
            Lifetime healthcare cost projection
        """
        years = int(life_expectancy_age - current_age)

        medicare_df = self.project_medicare_costs(current_age, income, years)

        total_premiums = medicare_df['total_premiums'].sum()

        # Out-of-pocket costs
        total_oop = 0
        for year in range(years):
            age = current_age + year
            annual_oop = self.estimate_out_of_pocket_costs(age, health_status)
            inflation_factor = (1 + self.medical_inflation) ** year
            total_oop += annual_oop * inflation_factor

        # One-time costs (end of life)
        end_of_life_costs = 25000  # Estimated

        total_lifetime = total_premiums + total_oop + end_of_life_costs

        return {
            'total_lifetime_costs': total_lifetime,
            'total_premiums': total_premiums,
            'total_out_of_pocket': total_oop,
            'end_of_life_costs': end_of_life_costs,
            'average_annual': total_lifetime / years if years > 0 else 0,
            'medicare_projections': medicare_df,
        }


class LongTermCarePlanner:
    """
    Long-term care planning and insurance evaluation.

    Assesses LTC needs, evaluates insurance options, and
    plans for potential care costs.
    """

    def __init__(self):
        """Initialize long-term care planner."""
        # Average costs (2024 estimates, annual)
        self.nursing_home_cost = 108000  # Semi-private room
        self.assisted_living_cost = 54000
        self.home_care_cost = 61000  # 44 hours/week
        self.ltc_inflation = 0.04  # 4% LTC cost inflation

    def assess_ltc_probability(
        self,
        age: int,
        gender: Gender,
        health_status: HealthStatus = HealthStatus.AVERAGE,
    ) -> Dict[str, float]:
        """
        Assess probability of needing long-term care.

        Args:
            age: Current age
            gender: Gender
            health_status: Health status

        Returns:
            LTC probability estimates
        """
        # Base probabilities (per SSA and industry data)
        # Women have higher probability than men
        if gender == Gender.FEMALE:
            base_nursing_home = 0.58  # 58% of women need nursing home
            base_assisted_living = 0.35
            base_home_care = 0.48
        else:  # MALE
            base_nursing_home = 0.40  # 40% of men need nursing home
            base_assisted_living = 0.25
            base_home_care = 0.35

        # Health adjustment
        health_adjustments = {
            HealthStatus.EXCELLENT: 0.7,
            HealthStatus.GOOD: 0.85,
            HealthStatus.AVERAGE: 1.0,
            HealthStatus.POOR: 1.3,
        }

        adj = health_adjustments[health_status]

        return {
            'probability_nursing_home': min(base_nursing_home * adj, 1.0),
            'probability_assisted_living': min(base_assisted_living * adj, 1.0),
            'probability_home_care': min(base_home_care * adj, 1.0),
            'average_duration_years': 3.0 if gender == Gender.MALE else 3.7,
        }

    def calculate_expected_ltc_cost(
        self,
        age: int,
        gender: Gender,
        years_until_need: int = 15,
    ) -> Dict[str, Any]:
        """
        Calculate expected LTC costs.

        Args:
            age: Current age
            gender: Gender
            years_until_need: Years until care needed

        Returns:
            Expected LTC cost analysis
        """
        probs = self.assess_ltc_probability(age, gender)

        # Project costs to future
        inflation_factor = (1 + self.ltc_inflation) ** years_until_need

        future_nh_cost = self.nursing_home_cost * inflation_factor
        future_al_cost = self.assisted_living_cost * inflation_factor
        future_hc_cost = self.home_care_cost * inflation_factor

        avg_duration = probs['average_duration_years']

        # Expected cost = probability × cost × duration
        expected_nh = probs['probability_nursing_home'] * future_nh_cost * avg_duration * 0.5  # Mix of private/home
        expected_al = probs['probability_assisted_living'] * future_al_cost * avg_duration * 0.3
        expected_hc = probs['probability_home_care'] * future_hc_cost * avg_duration * 0.4

        total_expected = expected_nh + expected_al + expected_hc

        return {
            'total_expected_cost': total_expected,
            'nursing_home_expected': expected_nh,
            'assisted_living_expected': expected_al,
            'home_care_expected': expected_hc,
            'future_annual_costs': {
                'nursing_home': future_nh_cost,
                'assisted_living': future_al_cost,
                'home_care': future_hc_cost,
            },
        }

    def evaluate_ltc_insurance(
        self,
        age: int,
        gender: Gender,
        annual_premium: float,
        daily_benefit: float = 200,
        benefit_period_years: int = 3,
        elimination_period_days: int = 90,
    ) -> Dict[str, Any]:
        """
        Evaluate LTC insurance purchase decision.

        Args:
            age: Current age
            gender: Gender
            annual_premium: Annual insurance premium
            daily_benefit: Daily benefit amount
            benefit_period_years: Benefit period in years
            elimination_period_days: Waiting period before benefits

        Returns:
            Insurance evaluation
        """
        years_to_claim = 15  # Assume claim at age + 15

        # Calculate total premiums paid
        total_premiums = annual_premium * years_to_claim

        # Calculate potential benefits
        annual_benefit = daily_benefit * 365
        total_max_benefits = annual_benefit * benefit_period_years

        # Expected benefit (probability-adjusted)
        probs = self.assess_ltc_probability(age, gender)
        prob_claim = max(probs['probability_nursing_home'], probs['probability_assisted_living'])

        expected_benefit = prob_claim * total_max_benefits * 0.8  # 80% utilization

        # Cost of self-insuring
        ltc_costs = self.calculate_expected_ltc_cost(age, gender, years_to_claim)
        self_insure_cost = ltc_costs['total_expected_cost']

        # ROI analysis
        net_benefit = expected_benefit - total_premiums

        return {
            'total_premiums_paid': total_premiums,
            'max_potential_benefit': total_max_benefits,
            'expected_benefit': expected_benefit,
            'probability_of_claim': prob_claim,
            'net_expected_value': net_benefit,
            'self_insure_cost': self_insure_cost,
            'recommendation': 'purchase' if expected_benefit > total_premiums * 1.2 else 'self_insure',
            'breakeven_years': total_premiums / annual_benefit if annual_benefit > 0 else float('inf'),
        }


class LegacyPlanner:
    """
    Legacy and estate planning optimization.

    Coordinates wealth transfer strategies, estate tax planning,
    and inheritance optimization.
    """

    def __init__(
        self,
        estate_tax_exemption: float = 13610000,  # 2024 federal exemption
        state_estate_tax: float = 0.0,
    ):
        """
        Initialize legacy planner.

        Args:
            estate_tax_exemption: Federal estate tax exemption
            state_estate_tax: State estate tax rate
        """
        self.estate_tax_exemption = estate_tax_exemption
        self.state_estate_tax = state_estate_tax

    def calculate_estate_taxes(
        self,
        estate_value: float,
        marital_deduction: float = 0,
    ) -> Dict[str, float]:
        """
        Calculate federal estate taxes.

        Args:
            estate_value: Total estate value
            marital_deduction: Amount passing to spouse (unlimited deduction)

        Returns:
            Estate tax calculation
        """
        taxable_estate = max(0, estate_value - marital_deduction - self.estate_tax_exemption)

        # Federal estate tax is 40% on amount over exemption
        federal_tax = taxable_estate * 0.40

        # State tax (if applicable)
        state_tax = estate_value * self.state_estate_tax

        total_tax = federal_tax + state_tax

        return {
            'gross_estate': estate_value,
            'marital_deduction': marital_deduction,
            'taxable_estate': taxable_estate,
            'federal_estate_tax': federal_tax,
            'state_estate_tax': state_tax,
            'total_estate_tax': total_tax,
            'net_to_heirs': estate_value - total_tax,
        }

    def optimize_legacy_distribution(
        self,
        total_estate: float,
        desired_to_heirs: float,
        desired_to_charity: float,
        life_expectancy_years: int,
    ) -> Dict[str, Any]:
        """
        Optimize estate distribution strategy.

        Args:
            total_estate: Total estate value
            desired_to_heirs: Amount desired for heirs
            desired_to_charity: Amount desired for charity
            life_expectancy_years: Years until death

        Returns:
            Optimized distribution strategy
        """
        # Calculate estate taxes without planning
        baseline = self.calculate_estate_taxes(total_estate)

        # With charitable giving (reduces taxable estate)
        with_charity = self.calculate_estate_taxes(
            total_estate - desired_to_charity
        )

        # Charitable giving reduces estate taxes
        tax_savings = baseline['total_estate_tax'] - with_charity['total_estate_tax']

        # Net cost of charitable gift
        net_gift_cost = desired_to_charity - tax_savings

        # Check if goals can be met
        after_tax_to_heirs = with_charity['net_to_heirs']
        goals_met = after_tax_to_heirs >= desired_to_heirs

        return {
            'total_estate': total_estate,
            'baseline_taxes': baseline['total_estate_tax'],
            'charitable_giving': desired_to_charity,
            'tax_savings_from_charity': tax_savings,
            'net_cost_of_gift': net_gift_cost,
            'amount_to_heirs': after_tax_to_heirs,
            'goals_met': goals_met,
            'recommendations': self._generate_legacy_recommendations(
                total_estate, desired_to_heirs, desired_to_charity, goals_met
            ),
        }

    def _generate_legacy_recommendations(
        self,
        estate: float,
        to_heirs: float,
        to_charity: float,
        goals_met: bool,
    ) -> List[str]:
        """Generate legacy planning recommendations."""
        recommendations = []

        if estate > self.estate_tax_exemption:
            recommendations.append("Estate may be subject to federal estate tax")
            recommendations.append("Consider: Annual gift tax exclusions ($18,000/person in 2024)")
            recommendations.append("Consider: Irrevocable life insurance trust (ILIT)")

        if to_charity > 0:
            recommendations.append("Charitable giving strategies:")
            recommendations.append("- Qualified Charitable Distributions (QCDs) from IRA")
            recommendations.append("- Donor Advised Fund for immediate deduction")
            recommendations.append("- Charitable Remainder Trust for income + charity")

        if not goals_met:
            recommendations.append("Estate insufficient for all goals - consider:")
            recommendations.append("- Life insurance to increase estate value")
            recommendations.append("- Reduce charitable giving target")

        return recommendations


class CharitableGivingStrategy:
    """
    Tax-efficient charitable giving strategies.

    Implements QCDs, DAFs, Charitable Trusts, and other
    philanthropic vehicles for tax optimization.
    """

    def __init__(self):
        """Initialize charitable giving strategy."""
        self.qcd_max = 105000  # 2024 QCD limit (was $100k, adjusted for inflation)
        self.standard_deduction_single = 14600  # 2024
        self.standard_deduction_married = 29200

    def evaluate_qcd_strategy(
        self,
        age: int,
        ira_balance: float,
        rmd_amount: float,
        desired_charity: float,
        other_deductions: float = 0,
    ) -> Dict[str, Any]:
        """
        Evaluate Qualified Charitable Distribution strategy.

        Args:
            age: Current age (must be 70.5+)
            ira_balance: Traditional IRA balance
            rmd_amount: Required RMD amount
            desired_charity: Desired charitable giving
            other_deductions: Other itemized deductions

        Returns:
            QCD strategy evaluation
        """
        if age < 70.5:
            return {
                'eligible': False,
                'message': 'Must be age 70.5 or older for QCD',
            }

        # QCD can satisfy RMD and reduce taxable income
        qcd_amount = min(desired_charity, rmd_amount, self.qcd_max)

        # Tax savings from QCD
        # QCD excludes amount from taxable income (better than deduction)
        marginal_tax_rate = 0.24  # Assumed
        qcd_tax_savings = qcd_amount * marginal_tax_rate

        # Compare to regular charitable deduction
        regular_deduction_benefit = 0
        if other_deductions + desired_charity > self.standard_deduction_single:
            # Itemizing is beneficial
            regular_deduction_benefit = desired_charity * marginal_tax_rate

        qcd_advantage = qcd_tax_savings - regular_deduction_benefit

        return {
            'eligible': True,
            'recommended_qcd_amount': qcd_amount,
            'rmd_remaining': max(0, rmd_amount - qcd_amount),
            'qcd_tax_savings': qcd_tax_savings,
            'regular_deduction_value': regular_deduction_benefit,
            'qcd_advantage': qcd_advantage,
            'recommendation': 'Use QCD' if qcd_advantage > 0 else 'Regular donation may be better',
        }

    def evaluate_daf_strategy(
        self,
        contribution_amount: float,
        marginal_tax_rate: float,
        annual_grants: float,
        years_of_giving: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate Donor Advised Fund strategy.

        Args:
            contribution_amount: Initial DAF contribution
            marginal_tax_rate: Current marginal tax rate
            annual_grants: Annual grants from DAF
            years_of_giving: Years of charitable giving

        Returns:
            DAF evaluation
        """
        # Immediate tax deduction
        immediate_deduction = contribution_amount * marginal_tax_rate

        # DAF administrative fees (typically 0.6% - 1%)
        admin_fee_rate = 0.008  # 0.8%
        total_fees = 0

        daf_balance = contribution_amount
        for year in range(years_of_giving):
            total_fees += daf_balance * admin_fee_rate
            daf_balance -= annual_grants
            daf_balance *= 1.05  # Assumed 5% growth

        # Compare to direct annual giving
        direct_giving_deductions = annual_grants * marginal_tax_rate * years_of_giving

        # DAF benefit: immediate deduction timing
        time_value_benefit = immediate_deduction - direct_giving_deductions

        return {
            'contribution': contribution_amount,
            'immediate_tax_benefit': immediate_deduction,
            'estimated_admin_fees': total_fees,
            'direct_giving_tax_benefit': direct_giving_deductions,
            'daf_advantage': time_value_benefit,
            'recommendation': 'DAF beneficial' if time_value_benefit > total_fees else 'Direct giving may be better',
            'advantages': [
                'Immediate tax deduction',
                'Investment growth potential',
                'Flexibility in grant timing',
                'Simplified record-keeping',
            ],
        }

    def evaluate_crt_strategy(
        self,
        asset_value: float,
        cost_basis: float,
        annual_payout_rate: float = 0.05,
        trust_term_years: int = 20,
    ) -> Dict[str, Any]:
        """
        Evaluate Charitable Remainder Trust strategy.

        Args:
            asset_value: Value of asset to contribute
            cost_basis: Cost basis of asset
            annual_payout_rate: Annual payout percentage (5-50%)
            trust_term_years: Trust term in years

        Returns:
            CRT evaluation
        """
        # Capital gains if sold directly
        capital_gain = asset_value - cost_basis
        capital_gains_tax = capital_gain * 0.20  # 20% LTCG rate + 3.8% NIIT

        # CRT benefits
        # 1. Avoid immediate capital gains tax
        # 2. Receive income stream
        # 3. Immediate charitable deduction

        annual_income = asset_value * annual_payout_rate
        total_income = annual_income * trust_term_years

        # Charitable deduction (present value of remainder)
        # Simplified: assume 40% of asset value
        charitable_deduction = asset_value * 0.40
        tax_benefit_deduction = charitable_deduction * 0.37  # Assumed tax rate

        # Net benefit
        net_benefit = (
            capital_gains_tax +  # Tax avoided
            tax_benefit_deduction  # Deduction benefit
        )

        # To charity at end
        to_charity = asset_value - total_income  # Simplified

        return {
            'asset_value': asset_value,
            'capital_gains_tax_avoided': capital_gains_tax,
            'annual_income': annual_income,
            'total_income_stream': total_income,
            'charitable_deduction': charitable_deduction,
            'deduction_tax_benefit': tax_benefit_deduction,
            'total_benefit': net_benefit,
            'remainder_to_charity': to_charity,
            'recommendation': 'CRT beneficial' if net_benefit > 10000 else 'May not be worth complexity',
            'requirements': [
                'Minimum $100,000 asset value typically required',
                'Irrevocable trust',
                'Annual payout rate between 5% and 50%',
                'Professional trustee recommended',
            ],
        }