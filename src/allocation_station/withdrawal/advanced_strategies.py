"""
Advanced Withdrawal Strategies for Retirement Planning

This module provides sophisticated withdrawal strategies for retirement income planning
including dynamic programming, Social Security optimization, pension integration,
annuities, RMDs, multi-account coordination, and income floor strategies.

Key Features:
- Dynamic programming for optimal withdrawal sequences
- Social Security claiming age optimization
- Pension integration and lifetime income planning
- Annuity strategies (immediate, deferred, variable)
- Required Minimum Distribution (RMD) calculations
- Multi-account tax optimization (401k, IRA, Roth, taxable)
- Income floor strategies for essential expenses

Classes:
    DynamicWithdrawalOptimizer: DP for optimal withdrawal decisions
    SocialSecurityOptimizer: Claiming strategy optimization
    PensionIntegrator: Pension income coordination
    AnnuityStrategy: Annuity evaluation and integration
    RMDCalculator: Required minimum distribution calculations
    MultiAccountCoordinator: Tax-efficient multi-account withdrawals
    IncomeFloorStrategy: Essential expense coverage planning
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class AccountType(str, Enum):
    """Types of retirement accounts."""
    TRADITIONAL_401K = "traditional_401k"
    TRADITIONAL_IRA = "traditional_ira"
    ROTH_401K = "roth_401k"
    ROTH_IRA = "roth_ira"
    TAXABLE = "taxable"
    HSA = "hsa"


class ClaimingAge(int, Enum):
    """Social Security claiming ages."""
    AGE_62 = 62  # Early claiming
    AGE_67 = 67  # Full retirement age (varies by birth year)
    AGE_70 = 70  # Maximum benefit


class AnnuityType(str, Enum):
    """Types of annuities."""
    IMMEDIATE = "immediate"
    DEFERRED = "deferred"
    VARIABLE = "variable"
    FIXED = "fixed"
    INDEXED = "indexed"


@dataclass
class WithdrawalDecision:
    """Single period withdrawal decision."""
    age: int
    year: int
    withdrawals_by_account: Dict[AccountType, float]
    total_withdrawal: float
    taxes_paid: float
    after_tax_income: float
    remaining_balance: float


class SocialSecurityBenefit(BaseModel):
    """Social Security benefit parameters."""
    pia: float = Field(description="Primary Insurance Amount at FRA")
    fra: int = Field(67, description="Full Retirement Age")
    earnings_record: List[float] = Field(default_factory=list, description="Historical earnings")
    cola_rate: float = Field(0.025, description="Cost of living adjustment rate")


class Pension(BaseModel):
    """Pension benefit specification."""
    annual_benefit: float = Field(description="Annual pension payment")
    start_age: int = Field(65, description="Age pension begins")
    cola_adjustment: float = Field(0.0, description="Annual COLA percentage")
    survivor_percentage: float = Field(0.5, description="Survivor benefit percentage")
    joint_life: bool = Field(False, description="Joint and survivor option")


class Annuity(BaseModel):
    """Annuity contract specification."""
    annuity_type: AnnuityType
    premium: float = Field(description="Purchase price")
    annual_payment: Optional[float] = Field(None, description="Annual payment (if known)")
    start_age: int = Field(65, description="Age payments begin")
    guarantee_period: int = Field(0, description="Period certain guarantee (years)")
    inflation_adjusted: bool = Field(False, description="Inflation protection")


class RetirementAccount(BaseModel):
    """Retirement account with tax treatment."""
    account_type: AccountType
    balance: float
    contributions: List[float] = Field(default_factory=list, description="Annual contributions")
    cost_basis: float = Field(0.0, description="Cost basis for taxable accounts")

    class Config:
        use_enum_values = True


class DynamicWithdrawalOptimizer:
    """
    Dynamic programming for optimal withdrawal sequence.

    Determines optimal withdrawal amounts and account sequencing
    to maximize after-tax lifetime income and minimize taxes.
    """

    def __init__(
        self,
        accounts: List[RetirementAccount],
        retirement_age: int = 65,
        planning_horizon: int = 30,
        annual_expenses: float = 100000,
    ):
        """
        Initialize dynamic withdrawal optimizer.

        Args:
            accounts: List of retirement accounts
            retirement_age: Age at retirement
            planning_horizon: Years to plan for
            annual_expenses: Annual spending need
        """
        self.accounts = accounts
        self.retirement_age = retirement_age
        self.planning_horizon = planning_horizon
        self.annual_expenses = annual_expenses
        self.memo = {}  # Memoization for DP

    def optimize_withdrawal_sequence(
        self,
        return_rate: float = 0.05,
        tax_rates: Optional[Dict[str, float]] = None,
    ) -> List[WithdrawalDecision]:
        """
        Find optimal withdrawal sequence using dynamic programming.

        Args:
            return_rate: Expected portfolio return
            tax_rates: Tax rates by income bracket

        Returns:
            List of optimal withdrawal decisions by year
        """
        if tax_rates is None:
            tax_rates = self._default_tax_rates()

        decisions = []
        current_balances = {acc.account_type: acc.balance for acc in self.accounts}

        for year in range(self.planning_horizon):
            age = self.retirement_age + year

            # Find optimal withdrawal for this year
            optimal = self._find_optimal_withdrawal(
                current_balances,
                self.annual_expenses,
                age,
                tax_rates,
            )

            # Update balances
            for account_type, withdrawal in optimal['withdrawals'].items():
                current_balances[account_type] -= withdrawal
                # Apply growth to remaining balance
                current_balances[account_type] *= (1 + return_rate)

            decision = WithdrawalDecision(
                age=age,
                year=year,
                withdrawals_by_account=optimal['withdrawals'],
                total_withdrawal=optimal['total'],
                taxes_paid=optimal['taxes'],
                after_tax_income=optimal['after_tax'],
                remaining_balance=sum(current_balances.values()),
            )

            decisions.append(decision)

        return decisions

    def _find_optimal_withdrawal(
        self,
        balances: Dict[AccountType, float],
        need: float,
        age: int,
        tax_rates: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Find optimal withdrawal for a single period.

        Uses a greedy tax-efficient approach:
        1. Taxable accounts first (capital gains rates)
        2. Tax-deferred if in low bracket
        3. Roth last (tax-free)
        """
        withdrawals = {acc_type: 0.0 for acc_type in balances.keys()}
        remaining_need = need

        # Tax-efficient withdrawal sequence
        withdrawal_priority = self._get_withdrawal_priority(age)

        for account_type in withdrawal_priority:
            if remaining_need <= 0:
                break

            available = balances.get(account_type, 0.0)
            if available > 0:
                # Withdraw up to available or need
                withdrawal = min(available, remaining_need)
                withdrawals[account_type] = withdrawal
                remaining_need -= withdrawal

        total_withdrawal = sum(withdrawals.values())
        taxes = self._calculate_taxes(withdrawals, tax_rates, age)
        after_tax = total_withdrawal - taxes

        return {
            'withdrawals': withdrawals,
            'total': total_withdrawal,
            'taxes': taxes,
            'after_tax': after_tax,
        }

    def _get_withdrawal_priority(self, age: int) -> List[AccountType]:
        """
        Determine withdrawal priority based on tax efficiency.

        Args:
            age: Current age (affects penalty-free access)

        Returns:
            Ordered list of account types to withdraw from
        """
        if age < 59.5:
            # Before 59.5: Avoid early withdrawal penalties
            return [
                AccountType.TAXABLE,
                AccountType.ROTH_IRA,  # Contributions can be withdrawn
                AccountType.TRADITIONAL_IRA,  # Last resort (10% penalty)
            ]
        elif age < 73:  # Before RMDs
            # Tax-efficient sequence
            return [
                AccountType.TAXABLE,  # Capital gains rates
                AccountType.TRADITIONAL_IRA,  # Ordinary income
                AccountType.TRADITIONAL_401K,
                AccountType.ROTH_IRA,  # Save for last (tax-free)
                AccountType.ROTH_401K,
            ]
        else:  # RMD age
            # Must take RMDs from traditional accounts
            return [
                AccountType.TRADITIONAL_IRA,  # RMD required
                AccountType.TRADITIONAL_401K,  # RMD required
                AccountType.TAXABLE,
                AccountType.ROTH_IRA,
                AccountType.ROTH_401K,
            ]

    def _calculate_taxes(
        self,
        withdrawals: Dict[AccountType, float],
        tax_rates: Dict[str, float],
        age: int,
    ) -> float:
        """Calculate taxes on withdrawals."""
        ordinary_income = 0
        capital_gains = 0
        tax_free = 0

        for account_type, amount in withdrawals.items():
            if account_type in [AccountType.TRADITIONAL_IRA, AccountType.TRADITIONAL_401K]:
                ordinary_income += amount
            elif account_type == AccountType.TAXABLE:
                # Assume 50% is gains
                capital_gains += amount * 0.5
            elif account_type in [AccountType.ROTH_IRA, AccountType.ROTH_401K]:
                tax_free += amount

        # Progressive tax calculation
        ordinary_tax = ordinary_income * tax_rates.get('ordinary', 0.22)
        cap_gains_tax = capital_gains * tax_rates.get('capital_gains', 0.15)

        # Early withdrawal penalty
        penalty = 0
        if age < 59.5:
            penalty = ordinary_income * 0.10

        return ordinary_tax + cap_gains_tax + penalty

    def _default_tax_rates(self) -> Dict[str, float]:
        """Default tax rate assumptions."""
        return {
            'ordinary': 0.22,  # 22% federal
            'capital_gains': 0.15,  # 15% LTCG
            'state': 0.05,  # 5% state tax
        }


class SocialSecurityOptimizer:
    """
    Optimize Social Security claiming strategy.

    Determines optimal age to claim benefits to maximize
    lifetime value considering longevity and other income sources.
    """

    def __init__(
        self,
        benefit: SocialSecurityBenefit,
        life_expectancy: int = 90,
        discount_rate: float = 0.03,
    ):
        """
        Initialize Social Security optimizer.

        Args:
            benefit: Social Security benefit parameters
            life_expectancy: Expected age at death
            discount_rate: Discount rate for present value
        """
        self.benefit = benefit
        self.life_expectancy = life_expectancy
        self.discount_rate = discount_rate

    def calculate_benefit_at_age(self, claiming_age: int) -> float:
        """
        Calculate annual benefit based on claiming age.

        Args:
            claiming_age: Age to claim benefits

        Returns:
            Annual benefit amount
        """
        fra = self.benefit.fra
        pia = self.benefit.pia

        if claiming_age < fra:
            # Early claiming reduction: ~6.67% per year before FRA
            years_early = fra - claiming_age
            reduction_factor = 1 - (years_early * 0.0667)
            return pia * reduction_factor
        elif claiming_age > fra:
            # Delayed claiming credit: 8% per year after FRA up to 70
            years_delay = min(claiming_age - fra, 3)  # Max at 70
            increase_factor = 1 + (years_delay * 0.08)
            return pia * increase_factor
        else:
            return pia

    def calculate_lifetime_value(self, claiming_age: int) -> float:
        """
        Calculate present value of lifetime benefits.

        Args:
            claiming_age: Age to claim benefits

        Returns:
            Present value of lifetime benefits
        """
        annual_benefit = self.calculate_benefit_at_age(claiming_age)

        pv = 0
        for age in range(claiming_age, self.life_expectancy + 1):
            years_from_now = age - claiming_age
            # Apply COLA
            benefit_with_cola = annual_benefit * ((1 + self.benefit.cola_rate) ** years_from_now)
            # Discount to present value
            discount_factor = 1 / ((1 + self.discount_rate) ** years_from_now)
            pv += benefit_with_cola * discount_factor

        return pv

    def find_optimal_claiming_age(
        self,
        min_age: int = 62,
        max_age: int = 70,
    ) -> Dict[str, Any]:
        """
        Find optimal Social Security claiming age.

        Args:
            min_age: Minimum claiming age
            max_age: Maximum claiming age

        Returns:
            Dictionary with optimal age and analysis
        """
        results = {}

        for age in range(min_age, max_age + 1):
            annual_benefit = self.calculate_benefit_at_age(age)
            lifetime_value = self.calculate_lifetime_value(age)

            results[age] = {
                'annual_benefit': annual_benefit,
                'lifetime_value': lifetime_value,
                'breakeven_age': self._calculate_breakeven(age),
            }

        # Find age with maximum lifetime value
        optimal_age = max(results.keys(), key=lambda a: results[a]['lifetime_value'])

        return {
            'optimal_age': optimal_age,
            'optimal_benefit': results[optimal_age]['annual_benefit'],
            'optimal_lifetime_value': results[optimal_age]['lifetime_value'],
            'all_ages': results,
        }

    def _calculate_breakeven(self, claiming_age: int) -> int:
        """Calculate breakeven age vs claiming at FRA."""
        benefit_at_claim = self.calculate_benefit_at_age(claiming_age)
        benefit_at_fra = self.calculate_benefit_at_age(self.benefit.fra)

        if claiming_age == self.benefit.fra:
            return self.benefit.fra

        # Cumulative benefits must equal
        cumulative_claim = 0
        cumulative_fra = 0

        for age in range(claiming_age, self.life_expectancy + 1):
            cumulative_claim += benefit_at_claim

            if age >= self.benefit.fra:
                cumulative_fra += benefit_at_fra

            if cumulative_claim >= cumulative_fra and age >= self.benefit.fra:
                return age

        return self.life_expectancy


class PensionIntegrator:
    """
    Integrate pension income into retirement planning.

    Coordinates pension benefits with other income sources
    and portfolio withdrawals.
    """

    def __init__(
        self,
        pensions: List[Pension],
    ):
        """
        Initialize pension integrator.

        Args:
            pensions: List of pension benefits
        """
        self.pensions = pensions

    def calculate_pension_income(self, age: int, year: int = 0) -> float:
        """
        Calculate total pension income at given age/year.

        Args:
            age: Current age
            year: Year from retirement (for COLA)

        Returns:
            Total annual pension income
        """
        total_income = 0

        for pension in self.pensions:
            if age >= pension.start_age:
                years_receiving = year - (pension.start_age - age) if year > 0 else 0
                cola_adjusted = pension.annual_benefit * ((1 + pension.cola_adjustment) ** max(0, years_receiving))
                total_income += cola_adjusted

        return total_income

    def optimize_pension_options(
        self,
        pension: Pension,
        spouse_life_expectancy: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compare pension payout options.

        Args:
            pension: Pension to analyze
            spouse_life_expectancy: Spouse's life expectancy

        Returns:
            Comparison of payout options
        """
        # Single life option (100% benefit, no survivor)
        single_life_value = pension.annual_benefit * 20  # Simplified NPV

        # Joint and survivor option (reduced benefit, survivor coverage)
        if pension.joint_life and spouse_life_expectancy:
            # Typically 5-10% reduction for joint option
            reduction_factor = 0.93  # 7% reduction
            joint_benefit = pension.annual_benefit * reduction_factor

            # Value includes potential survivor benefit
            joint_value = joint_benefit * 25  # Simplified NPV with longer expected payout
        else:
            joint_benefit = 0
            joint_value = 0

        return {
            'single_life': {
                'annual_benefit': pension.annual_benefit,
                'estimated_value': single_life_value,
            },
            'joint_survivor': {
                'annual_benefit': joint_benefit,
                'estimated_value': joint_value,
                'survivor_percentage': pension.survivor_percentage,
            },
            'recommendation': 'joint_survivor' if joint_value > single_life_value else 'single_life',
        }

    def calculate_reduced_portfolio_need(
        self,
        total_expenses: float,
        age: int,
        social_security: float = 0,
    ) -> float:
        """
        Calculate portfolio withdrawal needed after pension/SS.

        Args:
            total_expenses: Total annual expenses
            age: Current age
            social_security: Social Security income

        Returns:
            Required portfolio withdrawal
        """
        pension_income = self.calculate_pension_income(age)
        total_income = pension_income + social_security

        return max(0, total_expenses - total_income)


class AnnuityStrategy:
    """
    Evaluate and integrate annuity strategies.

    Analyzes annuity products and their role in retirement
    income planning.
    """

    def __init__(self):
        """Initialize annuity strategy analyzer."""
        self.mortality_factor = 0.02  # Simplified mortality assumption

    def calculate_annuity_payment(
        self,
        premium: float,
        age: int,
        annuity_type: AnnuityType = AnnuityType.IMMEDIATE,
        interest_rate: float = 0.04,
    ) -> float:
        """
        Calculate annuity payment amount.

        Args:
            premium: Purchase premium
            age: Age at purchase
            annuity_type: Type of annuity
            interest_rate: Assumed interest rate

        Returns:
            Annual payment amount
        """
        if annuity_type == AnnuityType.IMMEDIATE:
            # Immediate annuity payment calculation
            # Using simplified annuity factor
            life_expectancy = 90 - age
            annuity_factor = self._calculate_annuity_factor(
                interest_rate, life_expectancy, self.mortality_factor
            )
            return premium / annuity_factor
        elif annuity_type == AnnuityType.DEFERRED:
            # Deferred annuity (accumulates then pays)
            deferral_years = 10  # Example
            accumulated_value = premium * ((1 + interest_rate) ** deferral_years)
            life_expectancy = 90 - (age + deferral_years)
            annuity_factor = self._calculate_annuity_factor(
                interest_rate, life_expectancy, self.mortality_factor
            )
            return accumulated_value / annuity_factor
        else:
            # Variable/indexed annuities - simplified
            return premium * 0.05  # 5% payout rate assumption

    def _calculate_annuity_factor(
        self,
        interest_rate: float,
        years: int,
        mortality: float,
    ) -> float:
        """Calculate present value annuity factor with mortality."""
        factor = 0
        survival_prob = 1.0

        for year in range(1, years + 1):
            survival_prob *= (1 - mortality)
            discount_factor = 1 / ((1 + interest_rate) ** year)
            factor += survival_prob * discount_factor

        return factor

    def evaluate_annuity_purchase(
        self,
        premium: float,
        age: int,
        life_expectancy: int = 90,
        alternative_return: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Evaluate annuity vs self-managing portfolio.

        Args:
            premium: Annuity purchase amount
            age: Current age
            life_expectancy: Expected longevity
            alternative_return: Expected portfolio return

        Returns:
            Comparison analysis
        """
        # Annuity option
        annuity_payment = self.calculate_annuity_payment(premium, age)
        years = life_expectancy - age
        total_annuity_income = annuity_payment * years

        # Self-managed option (systematic withdrawal)
        withdrawal_rate = 0.04
        annual_withdrawal = premium * withdrawal_rate

        # Simulate portfolio over time
        portfolio_value = premium
        total_withdrawals = 0

        for year in range(years):
            withdrawal = min(annual_withdrawal, portfolio_value)
            total_withdrawals += withdrawal
            portfolio_value -= withdrawal
            portfolio_value *= (1 + alternative_return)

        total_portfolio_income = total_withdrawals + portfolio_value  # Include remaining balance

        return {
            'annuity': {
                'annual_payment': annuity_payment,
                'total_income': total_annuity_income,
                'income_guaranteed': True,
                'legacy_value': 0,
            },
            'portfolio': {
                'annual_withdrawal': annual_withdrawal,
                'total_income': total_portfolio_income,
                'income_guaranteed': False,
                'legacy_value': portfolio_value,
            },
            'break_even_age': self._calculate_annuity_breakeven(premium, annuity_payment, annual_withdrawal, alternative_return),
            'recommendation': 'annuity' if total_annuity_income > total_withdrawals else 'portfolio',
        }

    def _calculate_annuity_breakeven(
        self,
        premium: float,
        annuity_payment: float,
        withdrawal: float,
        return_rate: float,
    ) -> int:
        """Calculate breakeven point for annuity vs portfolio."""
        portfolio_value = premium
        annuity_cumulative = 0
        portfolio_cumulative = 0

        for year in range(50):  # Max 50 years
            annuity_cumulative += annuity_payment

            withdrawal_amt = min(withdrawal, portfolio_value)
            portfolio_cumulative += withdrawal_amt
            portfolio_value -= withdrawal_amt
            portfolio_value *= (1 + return_rate)

            if annuity_cumulative >= portfolio_cumulative + portfolio_value:
                return year

        return 50


class RMDCalculator:
    """
    Calculate Required Minimum Distributions (RMDs).

    Implements IRS RMD rules and life expectancy tables
    for tax-deferred retirement accounts.
    """

    # IRS Uniform Lifetime Table (simplified)
    UNIFORM_LIFETIME_TABLE = {
        72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7,
        77: 22.9, 78: 22.0, 79: 21.1, 80: 20.2, 81: 19.4,
        82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
        87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5,
        92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9, 96: 8.4,
        97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4,
    }

    def __init__(self):
        """Initialize RMD calculator."""
        self.rmd_start_age = 73  # Updated for SECURE Act 2.0 (2023+)

    def calculate_rmd(
        self,
        account_balance: float,
        age: int,
        use_joint_table: bool = False,
        spouse_age: Optional[int] = None,
    ) -> float:
        """
        Calculate Required Minimum Distribution.

        Args:
            account_balance: Account balance as of 12/31 prior year
            age: Account owner's age
            use_joint_table: Use joint life expectancy table
            spouse_age: Spouse age (if using joint table)

        Returns:
            Required minimum distribution amount
        """
        if age < self.rmd_start_age:
            return 0.0

        # Get life expectancy factor
        if use_joint_table and spouse_age and spouse_age < age - 10:
            # Use joint life expectancy table if spouse is >10 years younger
            life_expectancy = self._get_joint_life_expectancy(age, spouse_age)
        else:
            # Use uniform lifetime table
            life_expectancy = self.UNIFORM_LIFETIME_TABLE.get(age, 6.4)

        rmd = account_balance / life_expectancy

        return rmd

    def _get_joint_life_expectancy(self, owner_age: int, spouse_age: int) -> float:
        """Get joint life expectancy (simplified)."""
        # Simplified calculation - actual IRS table is more complex
        age_difference = owner_age - spouse_age
        base_expectancy = self.UNIFORM_LIFETIME_TABLE.get(owner_age, 10.0)

        # Adjust for younger spouse
        if age_difference > 10:
            adjustment = (age_difference - 10) * 0.5
            return base_expectancy + adjustment

        return base_expectancy

    def project_rmds(
        self,
        initial_balance: float,
        start_age: int,
        years: int = 30,
        return_rate: float = 0.05,
    ) -> pd.DataFrame:
        """
        Project RMDs over planning horizon.

        Args:
            initial_balance: Starting account balance
            start_age: Current age
            years: Years to project
            return_rate: Expected portfolio return

        Returns:
            DataFrame with RMD projections
        """
        projections = []
        balance = initial_balance

        for year in range(years):
            age = start_age + year

            # Calculate RMD
            rmd = self.calculate_rmd(balance, age)

            # Update balance
            balance -= rmd
            balance *= (1 + return_rate)

            projections.append({
                'year': year,
                'age': age,
                'beginning_balance': balance / (1 + return_rate) + rmd,
                'rmd': rmd,
                'ending_balance': balance,
                'life_expectancy_factor': self.UNIFORM_LIFETIME_TABLE.get(age, 6.4),
            })

        return pd.DataFrame(projections)

    def calculate_penalty(self, required_rmd: float, actual_withdrawal: float) -> float:
        """
        Calculate penalty for missing RMD.

        Args:
            required_rmd: Required RMD amount
            actual_withdrawal: Amount actually withdrawn

        Returns:
            Penalty amount (50% of shortfall, reduced to 25% by SECURE 2.0)
        """
        if actual_withdrawal >= required_rmd:
            return 0.0

        shortfall = required_rmd - actual_withdrawal
        penalty_rate = 0.25  # SECURE Act 2.0 reduced from 50% to 25%

        return shortfall * penalty_rate


class MultiAccountCoordinator:
    """
    Coordinate withdrawals across multiple account types.

    Optimizes withdrawal sequence across 401k, IRA, Roth, and
    taxable accounts for tax efficiency.
    """

    def __init__(
        self,
        accounts: List[RetirementAccount],
    ):
        """
        Initialize multi-account coordinator.

        Args:
            accounts: List of retirement accounts
        """
        self.accounts = {acc.account_type: acc for acc in accounts}
        self.rmd_calculator = RMDCalculator()

    def create_withdrawal_strategy(
        self,
        annual_need: float,
        age: int,
        tax_rates: Optional[Dict[str, float]] = None,
    ) -> Dict[AccountType, float]:
        """
        Create tax-efficient withdrawal strategy.

        Args:
            annual_need: Annual income needed
            age: Current age
            tax_rates: Tax rate assumptions

        Returns:
            Withdrawal amounts by account type
        """
        if tax_rates is None:
            tax_rates = {'ordinary': 0.22, 'capital_gains': 0.15}

        withdrawals = {}
        remaining_need = annual_need

        # Step 1: Handle RMDs if required
        if age >= 73:
            for account_type in [AccountType.TRADITIONAL_IRA, AccountType.TRADITIONAL_401K]:
                if account_type in self.accounts:
                    balance = self.accounts[account_type].balance
                    rmd = self.rmd_calculator.calculate_rmd(balance, age)
                    withdrawals[account_type] = rmd
                    remaining_need -= rmd

        # Step 2: Fill remaining need tax-efficiently
        if remaining_need > 0:
            # Priority: Taxable (lower rates) -> Traditional (ordinary rates) -> Roth (save for last)
            priority = [
                AccountType.TAXABLE,
                AccountType.TRADITIONAL_IRA,
                AccountType.TRADITIONAL_401K,
                AccountType.ROTH_IRA,
                AccountType.ROTH_401K,
            ]

            for account_type in priority:
                if remaining_need <= 0:
                    break

                if account_type in self.accounts:
                    available = self.accounts[account_type].balance - withdrawals.get(account_type, 0)
                    if available > 0:
                        withdrawal = min(available, remaining_need)
                        withdrawals[account_type] = withdrawals.get(account_type, 0) + withdrawal
                        remaining_need -= withdrawal

        return withdrawals

    def optimize_roth_conversions(
        self,
        current_age: int,
        retirement_age: int,
        traditional_balance: float,
        tax_rate_now: float,
        tax_rate_retirement: float,
    ) -> Dict[str, Any]:
        """
        Optimize Roth conversion strategy.

        Args:
            current_age: Current age
            retirement_age: Planned retirement age
            traditional_balance: Traditional IRA/401k balance
            tax_rate_now: Current marginal tax rate
            tax_rate_retirement: Expected retirement tax rate

        Returns:
            Roth conversion recommendation
        """
        years_to_retirement = retirement_age - current_age

        # Convert if current rate lower than expected retirement rate
        should_convert = tax_rate_now < tax_rate_retirement

        if should_convert and years_to_retirement > 5:
            # Gradual conversion strategy
            annual_conversion = traditional_balance / years_to_retirement

            # Stay within current tax bracket (simplified)
            max_conversion = 50000  # Example bracket limit
            recommended_conversion = min(annual_conversion, max_conversion)
        else:
            recommended_conversion = 0

        return {
            'should_convert': should_convert,
            'annual_conversion_amount': recommended_conversion,
            'years_to_convert': years_to_retirement,
            'tax_savings': traditional_balance * (tax_rate_retirement - tax_rate_now) if should_convert else 0,
            'conversion_taxes': recommended_conversion * tax_rate_now,
        }


class IncomeFloorStrategy:
    """
    Income floor strategy for essential expense coverage.

    Ensures essential expenses are covered by guaranteed income
    sources (Social Security, pensions, annuities).
    """

    def __init__(
        self,
        essential_expenses: float,
        discretionary_expenses: float,
    ):
        """
        Initialize income floor strategy.

        Args:
            essential_expenses: Annual essential expenses
            discretionary_expenses: Annual discretionary expenses
        """
        self.essential_expenses = essential_expenses
        self.discretionary_expenses = discretionary_expenses
        self.total_expenses = essential_expenses + discretionary_expenses

    def assess_income_floor(
        self,
        social_security: float,
        pension_income: float = 0,
        annuity_income: float = 0,
    ) -> Dict[str, Any]:
        """
        Assess income floor coverage.

        Args:
            social_security: Annual Social Security benefit
            pension_income: Annual pension income
            annuity_income: Annual annuity income

        Returns:
            Income floor analysis
        """
        guaranteed_income = social_security + pension_income + annuity_income

        floor_coverage_ratio = guaranteed_income / self.essential_expenses if self.essential_expenses > 0 else float('inf')

        shortfall = max(0, self.essential_expenses - guaranteed_income)
        surplus = max(0, guaranteed_income - self.essential_expenses)

        return {
            'guaranteed_income': guaranteed_income,
            'essential_expenses': self.essential_expenses,
            'floor_coverage_ratio': floor_coverage_ratio,
            'shortfall': shortfall,
            'surplus': surplus,
            'adequately_covered': floor_coverage_ratio >= 1.0,
            'portfolio_need': self.discretionary_expenses + shortfall,
        }

    def recommend_floor_strategy(
        self,
        current_guaranteed: float,
        portfolio_value: float,
        age: int,
    ) -> Dict[str, Any]:
        """
        Recommend strategy to achieve income floor.

        Args:
            current_guaranteed: Current guaranteed income
            portfolio_value: Current portfolio value
            age: Current age

        Returns:
            Strategy recommendation
        """
        assessment = self.assess_income_floor(current_guaranteed)

        if assessment['adequately_covered']:
            return {
                'status': 'adequate',
                'message': 'Essential expenses adequately covered by guaranteed income',
                'recommendation': 'Invest remaining portfolio for discretionary spending',
            }

        shortfall = assessment['shortfall']

        # Calculate annuity needed to cover shortfall
        annuity_strategy = AnnuityStrategy()
        required_annuity_payment = shortfall

        # Estimate premium needed (rough calculation)
        estimated_premium = required_annuity_payment * 20  # Simplified annuity pricing

        if estimated_premium <= portfolio_value * 0.5:  # Don't use more than 50% of portfolio
            return {
                'status': 'addressable',
                'shortfall': shortfall,
                'recommendation': 'Purchase immediate annuity',
                'annuity_premium': estimated_premium,
                'annuity_payment': required_annuity_payment,
                'remaining_portfolio': portfolio_value - estimated_premium,
            }
        else:
            return {
                'status': 'challenging',
                'shortfall': shortfall,
                'recommendation': 'Consider delaying retirement, reducing expenses, or partial annuitization',
                'gap': shortfall,
                'options': [
                    'Delay Social Security to age 70 for higher benefit',
                    'Reduce essential expense budget',
                    'Part-time work to supplement income',
                ],
            }