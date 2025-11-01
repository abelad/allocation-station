"""
Longevity Planning Examples

This module demonstrates all capabilities of the longevity planning framework,
including mortality analysis, longevity risk modeling, couple planning,
healthcare cost projections, long-term care planning, legacy planning, and
charitable giving strategies.

Examples:
    1. Basic Mortality Analysis
    2. Longevity Risk Modeling
    3. Couple/Joint Life Planning
    4. Healthcare Cost Projections
    5. Long-Term Care Planning
    6. Legacy Planning
    7. Charitable Giving Strategy Analysis
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from allocation_station.withdrawal.longevity_planning import (
    MortalityTable,
    LongevityRiskModeler,
    CoupleLifePlanner,
    HealthcareCostProjector,
    LongTermCarePlanner,
    LegacyPlanner,
    CharitableGivingStrategy,
    Gender,
    HealthStatus,
    LTCTrigger,
    CharitableVehicle,
)
import pandas as pd
import numpy as np
from datetime import datetime


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def example_1_basic_mortality_analysis():
    """
    Example 1: Basic Mortality Analysis

    Demonstrates how to use the MortalityTable class to analyze life expectancy,
    survival probabilities, and mortality rates for different ages, genders,
    and health statuses.
    """
    print_section("Example 1: Basic Mortality Analysis")

    # Initialize mortality table
    mortality_table = MortalityTable()

    # Analyze life expectancy for different scenarios
    print("Life Expectancy Analysis:")
    print("-" * 80)

    scenarios = [
        (65, Gender.MALE, HealthStatus.AVERAGE),
        (65, Gender.FEMALE, HealthStatus.AVERAGE),
        (65, Gender.MALE, HealthStatus.EXCELLENT),
        (65, Gender.FEMALE, HealthStatus.EXCELLENT),
        (70, Gender.MALE, HealthStatus.AVERAGE),
        (70, Gender.FEMALE, HealthStatus.POOR),
    ]

    for age, gender, health in scenarios:
        life_exp = mortality_table.calculate_life_expectancy(age, gender, health)
        print(f"\nAge: {age}, Gender: {gender.value}, Health: {health.value}")
        print(f"  Expected Additional Years: {life_exp.life_expectancy:.1f}")
        print(f"  Expected Age at Death: {life_exp.age_at_death:.1f}")
        print(f"  25th Percentile Age: {life_exp.percentile_25:.1f}")
        print(f"  50th Percentile Age: {life_exp.percentile_50:.1f}")
        print(f"  75th Percentile Age: {life_exp.percentile_75:.1f}")
        print(f"  Probability Age 90: {life_exp.probability_age_90:.1%}")
        print(f"  Probability Age 95: {life_exp.probability_age_95:.1%}")
        print(f"  Probability Age 100: {life_exp.probability_age_100:.1%}")

    # Survival probability analysis
    print("\n\nSurvival Probability Analysis:")
    print("-" * 80)

    current_age = 65
    gender = Gender.MALE
    target_ages = [70, 75, 80, 85, 90, 95, 100]

    print(f"\nProbability of a {current_age}-year-old {gender.value} surviving to:")
    for target_age in target_ages:
        prob = mortality_table.get_survival_probability(current_age, target_age, gender)
        print(f"  Age {target_age}: {prob:.1%}")

    # Mortality rate analysis
    print("\n\nMortality Rate Analysis:")
    print("-" * 80)

    ages_to_check = [60, 65, 70, 75, 80, 85, 90, 95, 100]

    print("\nAnnual Mortality Rates (deaths per 1,000):")
    print(f"{'Age':<8}{'Male':<15}{'Female':<15}")
    for age in ages_to_check:
        male_rate = mortality_table.get_mortality_rate(age, Gender.MALE) * 1000
        female_rate = mortality_table.get_mortality_rate(age, Gender.FEMALE) * 1000
        print(f"{age:<8}{male_rate:<15.2f}{female_rate:<15.2f}")

    # Survival curve  (using survival probability method)
    print("\n\nSurvival Curve:")
    print("-" * 80)

    print("\nSurvival probabilities for 65-year-old female:")
    print(f"{'Age':<8}{'Survival %':<15}")
    current_age = 65
    for age in range(70, 105, 5):
        prob = mortality_table.get_survival_probability(current_age, age, Gender.FEMALE)
        print(f"{age:<8}{prob:.1%}")


def example_2_longevity_risk_modeling():
    """
    Example 2: Longevity Risk Modeling

    Demonstrates Monte Carlo simulation of lifespan and portfolio sustainability,
    analyzing the risk of outliving one's assets under different scenarios.
    """
    print_section("Example 2: Longevity Risk Modeling")

    # Initialize modeler
    mortality_table = MortalityTable()
    modeler = LongevityRiskModeler(mortality_table=mortality_table)

    # Basic longevity risk simulation
    print("Longevity Risk Simulation:")
    print("-" * 80)

    current_age = 65
    gender = Gender.MALE
    health = HealthStatus.AVERAGE

    lifespans = modeler.simulate_lifespan(current_age, gender, health)

    print(f"\nSimulation for {current_age}-year-old {gender.value} ({health.value} health):")
    print(f"  Simulations: {len(lifespans):,}")
    print(f"  Mean Lifespan: {lifespans['lifespan'].mean():.1f} years")
    print(f"  Median Lifespan: {lifespans['lifespan'].median():.1f} years")
    print(f"  Standard Deviation: {lifespans['lifespan'].std():.1f} years")
    print(f"  Min Lifespan: {lifespans['lifespan'].min():.1f} years")
    print(f"  Max Lifespan: {lifespans['lifespan'].max():.1f} years")

    print("\nLifespan Percentiles:")
    percentiles = [10, 25, 50, 75, 90, 95]
    for p in percentiles:
        value = lifespans['lifespan'].quantile(p / 100)
        print(f"  {p}th percentile: {value:.1f} years")

    # Portfolio sustainability analysis
    print("\n\nPortfolio Sustainability Analysis:")
    print("-" * 80)

    # Scenario 1: Conservative withdrawal
    portfolio_value = 1_000_000
    annual_withdrawal = 40_000  # 4% rule
    annual_return = 0.06
    annual_volatility = 0.12

    result_conservative = modeler.analyze_portfolio_sustainability(
        current_age=current_age,
        gender=gender,
        health_status=health,
        portfolio_value=portfolio_value,
        annual_withdrawal=annual_withdrawal,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
    )

    print("\nScenario 1: Conservative Withdrawal (4% rule)")
    print(f"  Portfolio Value: ${portfolio_value:,.0f}")
    print(f"  Annual Withdrawal: ${annual_withdrawal:,.0f}")
    print(f"  Expected Return: {annual_return:.1%}")
    print(f"  Volatility: {annual_volatility:.1%}")
    print(f"  Success Rate: {result_conservative.success_rate:.1%}")
    print(f"  Median Years Until Depletion: {result_conservative.median_years_until_depletion:.1f}")
    print(f"  Median Final Portfolio Value: ${result_conservative.median_final_value:,.0f}")
    print(f"  Risk of Outliving Portfolio: {result_conservative.outlive_risk:.1%}")

    # Scenario 2: Aggressive withdrawal
    annual_withdrawal_aggressive = 60_000  # 6% rule

    result_aggressive = modeler.analyze_portfolio_sustainability(
        current_age=current_age,
        gender=gender,
        health_status=health,
        portfolio_value=portfolio_value,
        annual_withdrawal=annual_withdrawal_aggressive,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
    )

    print("\nScenario 2: Aggressive Withdrawal (6% rule)")
    print(f"  Portfolio Value: ${portfolio_value:,.0f}")
    print(f"  Annual Withdrawal: ${annual_withdrawal_aggressive:,.0f}")
    print(f"  Expected Return: {annual_return:.1%}")
    print(f"  Volatility: {annual_volatility:.1%}")
    print(f"  Success Rate: {result_aggressive.success_rate:.1%}")
    print(f"  Median Years Until Depletion: {result_aggressive.median_years_until_depletion:.1f}")
    print(f"  Median Final Portfolio Value: ${result_aggressive.median_final_value:,.0f}")
    print(f"  Risk of Outliving Portfolio: {result_aggressive.outlive_risk:.1%}")

    # Scenario 3: High return, low withdrawal
    annual_withdrawal_conservative = 30_000  # 3% rule
    annual_return_high = 0.08

    result_optimistic = modeler.analyze_portfolio_sustainability(
        current_age=current_age,
        gender=gender,
        health_status=health,
        portfolio_value=portfolio_value,
        annual_withdrawal=annual_withdrawal_conservative,
        annual_return=annual_return_high,
        annual_volatility=annual_volatility,
    )

    print("\nScenario 3: Optimistic (3% withdrawal, 8% return)")
    print(f"  Portfolio Value: ${portfolio_value:,.0f}")
    print(f"  Annual Withdrawal: ${annual_withdrawal_conservative:,.0f}")
    print(f"  Expected Return: {annual_return_high:.1%}")
    print(f"  Volatility: {annual_volatility:.1%}")
    print(f"  Success Rate: {result_optimistic.success_rate:.1%}")
    print(f"  Median Years Until Depletion: {result_optimistic.median_years_until_depletion:.1f}")
    print(f"  Median Final Portfolio Value: ${result_optimistic.median_final_value:,.0f}")
    print(f"  Risk of Outliving Portfolio: {result_optimistic.outlive_risk:.1%}")


def example_3_couple_planning():
    """
    Example 3: Couple/Joint Life Planning

    Demonstrates planning for couples including joint life expectancy,
    first and second death probabilities, and optimization strategies.
    """
    print_section("Example 3: Couple/Joint Life Planning")

    # Initialize planner with mortality table
    mortality_table = MortalityTable()
    planner = CoupleLifePlanner(mortality_table=mortality_table)

    # Basic couple analysis
    print("Couple Life Expectancy Analysis:")
    print("-" * 80)

    person1_age = 65
    person1_gender = Gender.MALE

    person2_age = 63
    person2_gender = Gender.FEMALE

    result = planner.calculate_joint_life_expectancy(
        age1=person1_age,
        gender1=person1_gender,
        age2=person2_age,
        gender2=person2_gender,
    )

    print(f"\nPerson 1: {person1_age}-year-old {person1_gender.value}")
    le1 = result['person1_life_expectancy']
    print(f"  Life Expectancy: {le1.life_expectancy:.1f} additional years")
    print(f"  Expected Age at Death: {le1.age_at_death:.1f}")

    print(f"\nPerson 2: {person2_age}-year-old {person2_gender.value}")
    le2 = result['person2_life_expectancy']
    print(f"  Life Expectancy: {le2.life_expectancy:.1f} additional years")
    print(f"  Expected Age at Death: {le2.age_at_death:.1f}")

    print(f"\nJoint Analysis:")
    print(f"  Expected First Death Age: {result['expected_first_death_age']:.1f}")
    print(f"  Expected Second Death Age: {result['expected_second_death_age']:.1f}")
    print(f"  Survivor Years: {result['survivor_years']:.1f}")

    print(f"\nJoint Survival Probabilities:")
    for age, prob in sorted(result['joint_survival_probabilities'].items()):
        print(f"  Both alive at age {age}: {prob:.1%}")

    print(f"\nSurvivor Probabilities (at least one alive):")
    for age, prob in sorted(result['survivor_probabilities'].items()):
        print(f"  At least one alive at age {age}: {prob:.1%}")

    # Couple strategy optimization
    print("\n\nCouple Strategy Optimization:")
    print("-" * 80)

    portfolio_value = 2_000_000
    joint_annual_expenses = 80_000
    survivor_annual_expenses = 55_000

    strategy = planner.optimize_couple_strategy(
        ages=(person1_age, person2_age),
        genders=(person1_gender, person2_gender),
        portfolio_value=portfolio_value,
        desired_spending_couple=joint_annual_expenses,
        desired_spending_survivor=survivor_annual_expenses,
    )

    print(f"\nPortfolio: ${portfolio_value:,.0f}")
    print(f"Joint Annual Expenses: ${joint_annual_expenses:,.0f}")
    print(f"Survivor Annual Expenses: ${survivor_annual_expenses:,.0f}")
    print(f"\nStrategy Analysis:")
    print(f"  Total Spending Need: ${strategy['total_spending_need']:,.0f}")
    print(f"  Surplus/Shortfall: ${strategy['surplus_or_shortfall']:,.0f}")
    print(f"  Years Both Alive: {strategy['years_both_alive']:.1f}")
    print(f"  Years Survivor Alone: {strategy['years_survivor_alone']:.1f}")
    print(f"  Adequately Funded: {'Yes' if strategy['adequately_funded'] else 'No'}")
    print(f"\nRecommendations:")
    for rec in strategy['recommendations']:
        print(f"  - {rec}")


def example_4_healthcare_cost_projections():
    """
    Example 4: Healthcare Cost Projections

    Demonstrates projection of Medicare costs, IRMAA surcharges, out-of-pocket
    expenses, and lifetime healthcare costs.
    """
    print_section("Example 4: Healthcare Cost Projections")

    # Initialize projector
    projector = HealthcareCostProjector()

    # Medicare cost projections
    print("Medicare Cost Projections:")
    print("-" * 80)

    # Scenario 1: Lower income (no IRMAA)
    age = 65
    income_low = 95_000
    years = 30

    medicare_low = projector.project_medicare_costs(age, income_low, years)

    print(f"\nScenario 1: Lower Income (No IRMAA)")
    print(f"  Age: {age}")
    print(f"  Income: ${income_low:,.0f}")
    print(f"  Projection Period: {years} years")
    print(f"\n  Annual Costs (Year 1):")
    print(f"    Part B Premium: ${medicare_low['annual_costs'][0]['part_b']:.2f}")
    print(f"    Part D Premium: ${medicare_low['annual_costs'][0]['part_d']:.2f}")
    print(f"    IRMAA Surcharge: ${medicare_low['annual_costs'][0]['irmaa']:.2f}")
    print(f"    Total: ${medicare_low['annual_costs'][0]['total']:.2f}")

    print(f"\n  Total Projected Costs:")
    print(f"    Part B: ${medicare_low['total_part_b']:,.0f}")
    print(f"    Part D: ${medicare_low['total_part_d']:,.0f}")
    print(f"    IRMAA: ${medicare_low['total_irmaa']:,.0f}")
    print(f"    Grand Total: ${medicare_low['total_cost']:,.0f}")

    # Scenario 2: Higher income (with IRMAA)
    income_high = 250_000

    medicare_high = projector.project_medicare_costs(age, income_high, years)

    print(f"\nScenario 2: Higher Income (With IRMAA)")
    print(f"  Age: {age}")
    print(f"  Income: ${income_high:,.0f}")
    print(f"  Projection Period: {years} years")
    print(f"\n  Annual Costs (Year 1):")
    print(f"    Part B Premium: ${medicare_high['annual_costs'][0]['part_b']:.2f}")
    print(f"    Part D Premium: ${medicare_high['annual_costs'][0]['part_d']:.2f}")
    print(f"    IRMAA Surcharge: ${medicare_high['annual_costs'][0]['irmaa']:.2f}")
    print(f"    Total: ${medicare_high['annual_costs'][0]['total']:.2f}")

    print(f"\n  Total Projected Costs:")
    print(f"    Part B: ${medicare_high['total_part_b']:,.0f}")
    print(f"    Part D: ${medicare_high['total_part_d']:,.0f}")
    print(f"    IRMAA: ${medicare_high['total_irmaa']:,.0f}")
    print(f"    Grand Total: ${medicare_high['total_cost']:,.0f}")

    print(f"\n  IRMAA Impact: ${medicare_high['total_cost'] - medicare_low['total_cost']:,.0f} additional cost")

    # Out-of-pocket cost projections
    print("\n\nOut-of-Pocket Cost Projections:")
    print("-" * 80)

    # Scenario 1: Average health
    oop_average = projector.project_out_of_pocket_costs(
        age=age,
        years=years,
        health_status=HealthStatus.AVERAGE,
    )

    print(f"\nScenario 1: Average Health Status")
    print(f"  Annual Cost (Year 1): ${oop_average['annual_costs'][0]:,.0f}")
    print(f"  Annual Cost (Year 30): ${oop_average['annual_costs'][-1]:,.0f}")
    print(f"  Total {years}-Year Cost: ${oop_average['total_cost']:,.0f}")
    print(f"  Average Annual Cost: ${oop_average['average_annual_cost']:,.0f}")

    # Scenario 2: Poor health
    oop_poor = projector.project_out_of_pocket_costs(
        age=age,
        years=years,
        health_status=HealthStatus.POOR,
    )

    print(f"\nScenario 2: Poor Health Status")
    print(f"  Annual Cost (Year 1): ${oop_poor['annual_costs'][0]:,.0f}")
    print(f"  Annual Cost (Year 30): ${oop_poor['annual_costs'][-1]:,.0f}")
    print(f"  Total {years}-Year Cost: ${oop_poor['total_cost']:,.0f}")
    print(f"  Average Annual Cost: ${oop_poor['average_annual_cost']:,.0f}")

    # Lifetime healthcare cost estimate
    print("\n\nLifetime Healthcare Cost Estimates:")
    print("-" * 80)

    scenarios = [
        (Gender.MALE, HealthStatus.AVERAGE, 95_000),
        (Gender.FEMALE, HealthStatus.AVERAGE, 95_000),
        (Gender.MALE, HealthStatus.POOR, 250_000),
        (Gender.FEMALE, HealthStatus.GOOD, 150_000),
    ]

    for gender, health, income in scenarios:
        lifetime = projector.estimate_lifetime_healthcare_costs(
            age=age,
            gender=gender,
            health_status=health,
            income=income,
        )

        print(f"\n{gender.value.capitalize()}, {health.value} health, ${income:,.0f} income:")
        print(f"  Medicare Costs: ${lifetime['medicare_costs']:,.0f}")
        print(f"  Out-of-Pocket Costs: ${lifetime['out_of_pocket_costs']:,.0f}")
        print(f"  Total Lifetime Healthcare: ${lifetime['total_healthcare_costs']:,.0f}")
        print(f"  Planning Period: {lifetime['years_planned']} years")


def example_5_long_term_care_planning():
    """
    Example 5: Long-Term Care Planning

    Demonstrates analysis of long-term care probability, expected costs,
    insurance evaluation, and strategy recommendations.
    """
    print_section("Example 5: Long-Term Care Planning")

    # Initialize planner
    planner = LongTermCarePlanner()

    # LTC probability assessment
    print("Long-Term Care Probability Assessment:")
    print("-" * 80)

    scenarios = [
        (65, Gender.MALE, HealthStatus.AVERAGE),
        (65, Gender.FEMALE, HealthStatus.AVERAGE),
        (70, Gender.MALE, HealthStatus.POOR),
        (70, Gender.FEMALE, HealthStatus.EXCELLENT),
    ]

    for age, gender, health in scenarios:
        prob = planner.calculate_ltc_probability(age, gender, health)

        print(f"\n{age}-year-old {gender.value} ({health.value} health):")
        print(f"  Any LTC Need: {prob['any_ltc_probability']:.1%}")
        print(f"  Home Care: {prob['home_care_probability']:.1%}")
        print(f"  Assisted Living: {prob['assisted_living_probability']:.1%}")
        print(f"  Nursing Home: {prob['nursing_home_probability']:.1%}")
        print(f"  Average Duration (if needed): {prob['average_duration_years']:.1f} years")

    # Expected LTC costs
    print("\n\nExpected Long-Term Care Costs:")
    print("-" * 80)

    for age, gender, health in scenarios:
        costs = planner.calculate_expected_ltc_costs(age, gender, health)

        print(f"\n{age}-year-old {gender.value} ({health.value} health):")
        print(f"  Expected Total Cost: ${costs['expected_total_cost']:,.0f}")
        print(f"  Home Care Expected: ${costs['home_care_expected_cost']:,.0f}")
        print(f"  Assisted Living Expected: ${costs['assisted_living_expected_cost']:,.0f}")
        print(f"  Nursing Home Expected: ${costs['nursing_home_expected_cost']:,.0f}")
        print(f"  Risk-Adjusted Annual Cost: ${costs['risk_adjusted_annual_cost']:,.0f}")

    # LTC insurance evaluation
    print("\n\nLong-Term Care Insurance Evaluation:")
    print("-" * 80)

    # Scenario 1: 65-year-old male, comprehensive coverage
    eval1 = planner.evaluate_ltc_insurance(
        age=65,
        gender=Gender.MALE,
        health_status=HealthStatus.AVERAGE,
        annual_premium=3_500,
        daily_benefit=200,
        benefit_period_years=5,
        elimination_period_days=90,
        inflation_protection=True,
    )

    print(f"\nScenario 1: 65-year-old male, comprehensive coverage")
    print(f"  Annual Premium: ${eval1['annual_premium']:,.0f}")
    print(f"  Daily Benefit: ${eval1['daily_benefit']:.0f}")
    print(f"  Benefit Period: {eval1['benefit_period_years']} years")
    print(f"  Total Premiums (to age 85): ${eval1['total_premiums_paid']:,.0f}")
    print(f"  Expected Benefits: ${eval1['expected_benefits_received']:,.0f}")
    print(f"  Net Expected Value: ${eval1['net_expected_value']:,.0f}")
    print(f"  Return on Investment: {eval1['roi']:.1%}")
    print(f"  Break-Even Probability: {eval1['break_even_probability']:.1%}")
    print(f"  Recommendation: {eval1['recommendation']}")

    # Scenario 2: 70-year-old female, basic coverage
    eval2 = planner.evaluate_ltc_insurance(
        age=70,
        gender=Gender.FEMALE,
        health_status=HealthStatus.GOOD,
        annual_premium=5_000,
        daily_benefit=150,
        benefit_period_years=3,
        elimination_period_days=180,
        inflation_protection=False,
    )

    print(f"\nScenario 2: 70-year-old female, basic coverage")
    print(f"  Annual Premium: ${eval2['annual_premium']:,.0f}")
    print(f"  Daily Benefit: ${eval2['daily_benefit']:.0f}")
    print(f"  Benefit Period: {eval2['benefit_period_years']} years")
    print(f"  Total Premiums (to age 85): ${eval2['total_premiums_paid']:,.0f}")
    print(f"  Expected Benefits: ${eval2['expected_benefits_received']:,.0f}")
    print(f"  Net Expected Value: ${eval2['net_expected_value']:,.0f}")
    print(f"  Return on Investment: {eval2['roi']:.1%}")
    print(f"  Break-Even Probability: {eval2['break_even_probability']:.1%}")
    print(f"  Recommendation: {eval2['recommendation']}")

    # LTC planning strategy
    print("\n\nLong-Term Care Planning Strategy:")
    print("-" * 80)

    strategy = planner.create_ltc_strategy(
        age=65,
        gender=Gender.FEMALE,
        health_status=HealthStatus.AVERAGE,
        net_worth=2_500_000,
        annual_income=120_000,
    )

    print(f"\n65-year-old female, average health")
    print(f"Net Worth: ${2_500_000:,.0f}")
    print(f"Annual Income: ${120_000:,.0f}")
    print(f"\nRecommended Strategy: {strategy['recommended_strategy']}")
    print(f"\nLTC Risk Assessment:")
    print(f"  Probability of Need: {strategy['ltc_probability']:.1%}")
    print(f"  Expected Costs: ${strategy['expected_ltc_costs']:,.0f}")
    print(f"  % of Net Worth: {strategy['expected_ltc_costs'] / 2_500_000:.1%}")

    if strategy['insurance_recommendation']['should_purchase']:
        print(f"\nInsurance Recommendation: Purchase LTC Insurance")
        print(f"  Suggested Daily Benefit: ${strategy['insurance_recommendation']['suggested_daily_benefit']:.0f}")
        print(f"  Suggested Benefit Period: {strategy['insurance_recommendation']['suggested_benefit_period']} years")
        print(f"  Inflation Protection: {'Yes' if strategy['insurance_recommendation']['inflation_protection'] else 'No'}")
        print(f"  Estimated Annual Premium: ${strategy['insurance_recommendation']['estimated_annual_premium']:,.0f}")
    else:
        print(f"\nInsurance Recommendation: Self-Insure")

    print(f"\nAlternative Strategies:")
    for alt in strategy['alternative_strategies']:
        print(f"  - {alt}")


def example_6_legacy_planning():
    """
    Example 6: Legacy Planning

    Demonstrates estate tax calculations, legacy distribution optimization,
    charitable giving strategies, and wealth transfer planning.
    """
    print_section("Example 6: Legacy Planning")

    # Initialize planner
    planner = LegacyPlanner()

    # Estate tax calculation
    print("Estate Tax Calculations:")
    print("-" * 80)

    scenarios = [
        (10_000_000, False),
        (20_000_000, False),
        (30_000_000, False),
        (30_000_000, True),  # With marital deduction
    ]

    for estate_value, use_marital in scenarios:
        tax_result = planner.calculate_estate_tax(estate_value, use_marital_deduction=use_marital)

        marital_str = " (with marital deduction)" if use_marital else ""
        print(f"\nEstate Value: ${estate_value:,.0f}{marital_str}")
        print(f"  Exemption Amount: ${tax_result['exemption_amount']:,.0f}")
        print(f"  Taxable Estate: ${tax_result['taxable_estate']:,.0f}")
        print(f"  Estate Tax: ${tax_result['estate_tax']:,.0f}")
        print(f"  Effective Tax Rate: {tax_result['effective_tax_rate']:.2%}")
        print(f"  Net Estate After Tax: ${tax_result['net_estate']:,.0f}")

    # Legacy distribution optimization
    print("\n\nLegacy Distribution Optimization:")
    print("-" * 80)

    estate_value = 25_000_000
    heirs = [
        {'name': 'Child 1', 'age': 45, 'tax_bracket': 0.35},
        {'name': 'Child 2', 'age': 42, 'tax_bracket': 0.32},
        {'name': 'Child 3', 'age': 38, 'tax_bracket': 0.24},
    ]
    charitable_intent = 0.15  # 15% to charity

    distribution = planner.optimize_legacy_distribution(
        estate_value=estate_value,
        heirs=heirs,
        charitable_intent=charitable_intent,
    )

    print(f"\nEstate Value: ${estate_value:,.0f}")
    print(f"Charitable Intent: {charitable_intent:.0%}")
    print(f"\nOptimized Distribution:")
    print(f"  Estate Tax: ${distribution['estate_tax']:,.0f}")
    print(f"  Charitable Deduction: ${distribution['charitable_deduction']:,.0f}")
    print(f"  Total to Heirs: ${distribution['total_to_heirs']:,.0f}")
    print(f"  Total to Charity: ${distribution['total_to_charity']:,.0f}")

    print(f"\n  Heir Distributions:")
    for heir_dist in distribution['heir_distributions']:
        print(f"    {heir_dist['name']}:")
        print(f"      Direct Bequest: ${heir_dist['direct_bequest']:,.0f}")
        print(f"      Trust Amount: ${heir_dist['trust_amount']:,.0f}")
        print(f"      Total: ${heir_dist['total_amount']:,.0f}")

    print(f"\n  Tax Efficiency Metrics:")
    print(f"    Total Taxes: ${distribution['total_taxes']:,.0f}")
    print(f"    Effective Tax Rate: {distribution['effective_tax_rate']:.2%}")
    print(f"    Charitable Tax Savings: ${distribution['charitable_tax_savings']:,.0f}")

    # Legacy goal analysis
    print("\n\nLegacy Goal Analysis:")
    print("-" * 80)

    current_age = 65
    gender = Gender.MALE
    health = HealthStatus.AVERAGE
    current_portfolio = 15_000_000
    annual_expenses = 200_000
    legacy_goal = 5_000_000

    goal_analysis = planner.analyze_legacy_goal(
        current_age=current_age,
        gender=gender,
        health_status=health,
        current_portfolio_value=current_portfolio,
        annual_expenses=annual_expenses,
        legacy_goal=legacy_goal,
    )

    print(f"\nCurrent Situation:")
    print(f"  Age: {current_age}, {gender.value}, {health.value} health")
    print(f"  Portfolio: ${current_portfolio:,.0f}")
    print(f"  Annual Expenses: ${annual_expenses:,.0f}")
    print(f"  Legacy Goal: ${legacy_goal:,.0f}")

    print(f"\nAnalysis Results:")
    print(f"  Probability of Meeting Goal: {goal_analysis['probability_of_meeting_goal']:.1%}")
    print(f"  Expected Legacy Amount: ${goal_analysis['expected_legacy_amount']:,.0f}")
    print(f"  Goal Feasibility: {goal_analysis['feasibility']}")
    print(f"  Required Return: {goal_analysis['required_return']:.2%}")
    print(f"  Shortfall Risk: {goal_analysis['shortfall_risk']:.1%}")

    print(f"\nRecommendations:")
    for rec in goal_analysis['recommendations']:
        print(f"  - {rec}")


def example_7_charitable_giving_strategy():
    """
    Example 7: Charitable Giving Strategy Analysis

    Demonstrates evaluation of charitable giving vehicles including QCDs,
    Donor Advised Funds, and Charitable Remainder Trusts.
    """
    print_section("Example 7: Charitable Giving Strategy Analysis")

    # Initialize strategy analyzer
    strategy = CharitableGivingStrategy()

    # QCD (Qualified Charitable Distribution) analysis
    print("Qualified Charitable Distribution (QCD) Analysis:")
    print("-" * 80)

    qcd_scenarios = [
        (72, 150_000, 10_000, 0.24),
        (75, 200_000, 25_000, 0.32),
        (80, 120_000, 15_000, 0.22),
    ]

    for age, income, qcd_amount, tax_rate in qcd_scenarios:
        qcd_eval = strategy.evaluate_qcd(
            age=age,
            ira_balance=500_000,
            annual_rmd=20_000,
            charitable_intent=qcd_amount,
            marginal_tax_rate=tax_rate,
        )

        print(f"\nAge {age}, Income ${income:,.0f}, QCD ${qcd_amount:,.0f}, Tax Rate {tax_rate:.0%}")
        print(f"  RMD Amount: ${qcd_eval['rmd_amount']:,.0f}")
        print(f"  QCD Amount: ${qcd_eval['qcd_amount']:,.0f}")
        print(f"  Taxable RMD: ${qcd_eval['taxable_rmd']:,.0f}")
        print(f"  Tax Savings: ${qcd_eval['tax_savings']:,.0f}")
        print(f"  Effective Benefit Rate: {qcd_eval['effective_benefit_rate']:.1%}")
        print(f"  Recommendation: {qcd_eval['recommendation']}")

    # Donor Advised Fund analysis
    print("\n\nDonor Advised Fund (DAF) Analysis:")
    print("-" * 80)

    daf_scenarios = [
        (100_000, 0.0, 0.32, 5),
        (250_000, 0.50, 0.35, 10),
        (50_000, 0.30, 0.24, 3),
    ]

    for contribution, ltcg_pct, tax_rate, years in daf_scenarios:
        daf_eval = strategy.evaluate_daf(
            contribution_amount=contribution,
            asset_appreciation=ltcg_pct,
            marginal_tax_rate=tax_rate,
            time_horizon_years=years,
        )

        print(f"\nContribution: ${contribution:,.0f}, Appreciation: {ltcg_pct:.0%}, Tax Rate: {tax_rate:.0%}")
        print(f"  Contribution Amount: ${daf_eval['contribution_amount']:,.0f}")
        print(f"  Immediate Tax Deduction: ${daf_eval['immediate_tax_deduction']:,.0f}")
        print(f"  Capital Gains Tax Avoided: ${daf_eval['capital_gains_avoided']:,.0f}")
        print(f"  Total Tax Benefit: ${daf_eval['total_tax_benefit']:,.0f}")
        print(f"  Effective Benefit Rate: {daf_eval['effective_benefit_rate']:.1%}")
        print(f"  Projected Future Value: ${daf_eval['projected_future_value']:,.0f}")
        print(f"  Annual Granting Capacity ({years} years): ${daf_eval['annual_granting_capacity']:,.0f}")
        print(f"  Recommendation: {daf_eval['recommendation']}")

    # Charitable Remainder Trust analysis
    print("\n\nCharitable Remainder Trust (CRT) Analysis:")
    print("-" * 80)

    crt_scenarios = [
        (1_000_000, 0.05, 20, 65),
        (2_000_000, 0.06, 15, 70),
        (500_000, 0.055, 25, 60),
    ]

    for asset_value, payout_rate, term, age in crt_scenarios:
        crt_eval = strategy.evaluate_crt(
            asset_value=asset_value,
            payout_rate=payout_rate,
            term_years=term,
            beneficiary_age=age,
            marginal_tax_rate=0.35,
            capital_gains_tax_rate=0.20,
            asset_appreciation=0.80,
        )

        print(f"\nAsset: ${asset_value:,.0f}, Payout: {payout_rate:.1%}, Term: {term} years, Age: {age}")
        print(f"  Asset Value: ${crt_eval['asset_value']:,.0f}")
        print(f"  Annual Payout: ${crt_eval['annual_payout']:,.0f}")
        print(f"  Total Payouts ({term} years): ${crt_eval['total_payouts']:,.0f}")
        print(f"  Immediate Tax Deduction: ${crt_eval['immediate_tax_deduction']:,.0f}")
        print(f"  Capital Gains Tax Avoided: ${crt_eval['capital_gains_tax_avoided']:,.0f}")
        print(f"  Total Tax Benefits: ${crt_eval['total_tax_benefits']:,.0f}")
        print(f"  Charitable Remainder: ${crt_eval['charitable_remainder']:,.0f}")
        print(f"  Net Benefit to Donor: ${crt_eval['net_benefit_to_donor']:,.0f}")
        print(f"  Recommendation: {crt_eval['recommendation']}")

    # Comprehensive charitable strategy comparison
    print("\n\nComprehensive Charitable Strategy Comparison:")
    print("-" * 80)

    comparison = strategy.compare_strategies(
        annual_charitable_intent=25_000,
        age=70,
        ira_balance=800_000,
        taxable_assets=1_500_000,
        marginal_tax_rate=0.32,
        planning_horizon_years=15,
    )

    print(f"\nScenario:")
    print(f"  Annual Charitable Intent: ${25_000:,.0f}")
    print(f"  Age: 70")
    print(f"  IRA Balance: ${800_000:,.0f}")
    print(f"  Taxable Assets: ${1_500_000:,.0f}")
    print(f"  Planning Horizon: 15 years")

    print(f"\nStrategy Comparison:")
    strategies_list = ['direct_giving', 'qcd', 'daf', 'crt']
    strategy_names = {
        'direct_giving': 'Direct Giving',
        'qcd': 'QCD Strategy',
        'daf': 'DAF Strategy',
        'crt': 'CRT Strategy',
    }

    for strat in strategies_list:
        print(f"\n  {strategy_names[strat]}:")
        print(f"    Total Tax Benefits: ${comparison[strat]['total_tax_benefits']:,.0f}")
        print(f"    Total Charitable Impact: ${comparison[strat]['total_charitable_impact']:,.0f}")
        print(f"    Annual Cost to Donor: ${comparison[strat]['annual_cost_to_donor']:,.0f}")
        print(f"    Efficiency Score: {comparison[strat]['efficiency_score']:.1%}")

    print(f"\nRecommended Strategy: {comparison['recommended_strategy']}")
    print(f"\nKey Insights:")
    for insight in comparison['insights']:
        print(f"  - {insight}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print(" LONGEVITY PLANNING EXAMPLES")
    print(" Comprehensive demonstration of longevity planning capabilities")
    print("=" * 80)

    try:
        example_1_basic_mortality_analysis()

        # Note: Examples 2-7 require additional implementation or API adjustments
        # Uncomment when all longevity planning features are fully implemented
        # example_2_longevity_risk_modeling()
        # example_3_couple_planning()
        # example_4_healthcare_cost_projections()
        # example_5_long_term_care_planning()
        # example_6_legacy_planning()
        # example_7_charitable_giving_strategy()

        print("\n" + "=" * 80)
        print(" Example 1 (Basic Mortality Analysis) completed successfully!")
        print(" (Examples 2-7 are commented out pending full implementation)")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
