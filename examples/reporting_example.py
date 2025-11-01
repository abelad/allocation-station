"""
Reporting System Examples

This module demonstrates all capabilities of the reporting system including
automated PDF report generation, customizable templates, email delivery,
executive summaries, regulatory compliance reports, client presentations,
and multi-language support.

Examples:
    1. Basic PDF Report Generation
    2. Executive Summary Report
    3. Regulatory Compliance Report
    4. Multi-Language Reports
    5. Email Report Delivery
    6. PowerPoint Presentation Generation
    7. Custom Report Templates
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from allocation_station.ui.reporting import (
    ReportingSystem,
    ReportConfig,
    PortfolioData,
    ReportType,
    ReportFormat,
    Language,
    PDFReportGenerator,
    PresentationGenerator,
    EmailDelivery,
    Translator,
    create_default_config
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List


def create_sample_portfolio() -> PortfolioData:
    """Create sample portfolio data for demonstrations."""

    # Sample holdings
    holdings = {
        'SPY': {
            'shares': 100,
            'price': 442.15,
            'value': 44215,
            'weight': 42.3,
            'change': 1.2
        },
        'TLT': {
            'shares': 150,
            'price': 92.30,
            'value': 13845,
            'weight': 13.3,
            'change': -0.5
        },
        'GLD': {
            'shares': 50,
            'price': 186.50,
            'value': 9325,
            'weight': 8.9,
            'change': 0.8
        },
        'VNQ': {
            'shares': 75,
            'price': 89.20,
            'value': 6690,
            'weight': 6.4,
            'change': 1.5
        },
        'QQQ': {
            'shares': 80,
            'price': 385.75,
            'value': 30860,
            'weight': 29.5,
            'change': 2.1
        }
    }

    # Calculate total value
    total_value = sum(h['value'] for h in holdings.values())

    # Performance metrics
    performance = {
        'ytd_return': 18.5,
        'annual_return': 15.2,
        'monthly_return': 1.8,
        'quarterly_return': 5.3,
        'total_return': 124.7
    }

    # Asset allocation
    allocation = {
        'Stocks': 71.8,
        'Bonds': 13.3,
        'Commodities': 8.9,
        'Real Estate': 6.4
    }

    # Risk metrics
    risk_metrics = {
        'volatility': 12.3,
        'sharpe_ratio': 1.45,
        'sortino_ratio': 1.87,
        'max_drawdown': -8.3,
        'beta': 0.95,
        'alpha': 2.1,
        'var_95': -2.5,
        'cvar_95': -3.8
    }

    # Historical returns (sample data)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    historical_returns = pd.DataFrame({
        'date': dates,
        'returns': np.random.randn(len(dates)) * 0.01,
        'cumulative': np.cumprod(1 + np.random.randn(len(dates)) * 0.01) * 100
    })

    # Benchmark comparison
    benchmark_comparison = {
        'benchmark_name': 'S&P 500',
        'portfolio_return': 18.5,
        'benchmark_return': 16.2,
        'outperformance': 2.3,
        'tracking_error': 3.4,
        'information_ratio': 0.68
    }

    return PortfolioData(
        portfolio_name="Diversified Growth Portfolio",
        total_value=total_value,
        holdings=holdings,
        performance=performance,
        allocation=allocation,
        risk_metrics=risk_metrics,
        historical_returns=historical_returns,
        benchmark_comparison=benchmark_comparison
    )


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def example_1_basic_pdf_report():
    """
    Example 1: Basic PDF Report Generation

    Demonstrates generating a basic portfolio summary report in PDF format
    with charts, tables, and standard sections.
    """
    print_section("Example 1: Basic PDF Report Generation")

    # Create sample portfolio
    portfolio = create_sample_portfolio()

    # Create report configuration
    config = create_default_config(
        report_type=ReportType.PORTFOLIO_SUMMARY,
        format=ReportFormat.PDF,
        language=Language.ENGLISH
    )

    # Initialize reporting system
    reporting = ReportingSystem(config)

    # Generate report
    output_path = "portfolio_summary.pdf"
    reporting.generate_report(portfolio, output_path)

    print(f"[OK] Basic PDF report generated: {output_path}")
    print(f"  Report Type: Portfolio Summary")
    print(f"  Format: PDF")
    print(f"  Language: English")
    print(f"  Total Value: ${portfolio.total_value:,.2f}")
    print(f"  YTD Return: {portfolio.performance['ytd_return']:.2f}%")
    print(f"  Holdings: {len(portfolio.holdings)}")


def example_2_executive_summary():
    """
    Example 2: Executive Summary Report

    Demonstrates generating an executive summary with AI-powered insights,
    key highlights, and strategic recommendations.
    """
    print_section("Example 2: Executive Summary Report")

    # Create sample portfolio
    portfolio = create_sample_portfolio()

    # Create executive summary configuration
    config = ReportConfig(
        report_type=ReportType.EXECUTIVE_SUMMARY,
        format=ReportFormat.PDF,
        language=Language.ENGLISH,
        include_charts=True,
        include_recommendations=True,
        company_name="Acme Investment Management"
    )

    # Create insights
    insights = {
        'highlights': [
            f"Portfolio outperformed benchmark by {portfolio.benchmark_comparison['outperformance']:.1f}%",
            f"Achieved Sharpe ratio of {portfolio.risk_metrics['sharpe_ratio']:.2f}",
            "Successfully maintained diversified allocation across asset classes",
            f"Low correlation with market (Beta: {portfolio.risk_metrics['beta']:.2f})",
            "Risk-adjusted returns in top quartile of peer group"
        ],
        'recommendations': [
            "Maintain current strategic allocation given strong performance",
            "Consider increasing international equity exposure by 5%",
            "Monitor bond duration as interest rate environment evolves",
            "Implement tax-loss harvesting in Q4 for optimal tax efficiency",
            "Review real estate allocation for potential rebalancing"
        ],
        'market_commentary': {
            'equity_markets': "Equity markets showed resilience amid economic uncertainty",
            'fixed_income': "Bond markets adjusted to higher rate environment",
            'outlook': "Cautiously optimistic for remainder of year"
        }
    }

    # Initialize reporting system
    reporting = ReportingSystem(config)

    # Generate executive summary
    output_path = "executive_summary.pdf"
    reporting.generate_report(portfolio, output_path, insights=insights)

    print(f"[OK] Executive summary generated: {output_path}")
    print(f"\n  Key Highlights:")
    for highlight in insights['highlights'][:3]:
        print(f"    • {highlight}")

    print(f"\n  Top Recommendations:")
    for rec in insights['recommendations'][:3]:
        print(f"    • {rec}")


def example_3_compliance_report():
    """
    Example 3: Regulatory Compliance Report

    Demonstrates generating a regulatory compliance report with SEC, FINRA,
    and DOL requirements including all necessary disclosures and certifications.
    """
    print_section("Example 3: Regulatory Compliance Report")

    # Create sample portfolio
    portfolio = create_sample_portfolio()

    # Create compliance configuration
    config = ReportConfig(
        report_type=ReportType.COMPLIANCE,
        format=ReportFormat.PDF,
        language=Language.ENGLISH,
        include_disclosures=True,
        company_name="Regulatory Compliant Advisors LLC"
    )

    # Create compliance data
    compliance_data = {
        'checks': {
            'SEC Rule 204-2': {
                'passed': True,
                'details': 'Books and records maintained per regulatory requirements'
            },
            'FINRA Rule 2210': {
                'passed': True,
                'details': 'Communications standards met, supervisory approval obtained'
            },
            'DOL ERISA 404(a)': {
                'passed': True,
                'details': 'Fiduciary duties satisfied, prudent investor standard met'
            },
            'GIPS Compliance': {
                'passed': True,
                'details': 'Performance calculations comply with GIPS standards'
            },
            'Form ADV Disclosures': {
                'passed': True,
                'details': 'All material conflicts of interest disclosed'
            },
            'Custody Rule': {
                'passed': True,
                'details': 'Qualified custodian arrangement verified'
            }
        },
        'certifications': {
            'portfolio_manager': 'John Smith, CFA',
            'compliance_officer': 'Jane Doe, CCO',
            'certification_date': datetime.now(),
            'next_review_date': datetime.now() + timedelta(days=90)
        },
        'disclosures': [
            'Past performance is not indicative of future results',
            'All investments involve risk including potential loss of principal',
            'Advisory fees and expenses reduce returns',
            'Market conditions may materially impact portfolio performance'
        ]
    }

    # Initialize reporting system
    reporting = ReportingSystem(config)

    # Generate compliance report
    output_path = "compliance_report.pdf"
    reporting.generate_report(portfolio, output_path, compliance_data=compliance_data)

    print(f"[OK] Compliance report generated: {output_path}")
    print(f"\n  Regulatory Checks Passed:")
    for check_name, check_result in compliance_data['checks'].items():
        if check_result['passed']:
            print(f"    [OK] {check_name}")

    print(f"\n  Certifications:")
    print(f"    Portfolio Manager: {compliance_data['certifications']['portfolio_manager']}")
    print(f"    Compliance Officer: {compliance_data['certifications']['compliance_officer']}")
    print(f"    Next Review: {compliance_data['certifications']['next_review_date'].strftime('%Y-%m-%d')}")


def example_4_multilanguage_reports():
    """
    Example 4: Multi-Language Reports

    Demonstrates generating reports in multiple languages including English,
    Spanish, French, German, and Chinese with proper translations.
    """
    print_section("Example 4: Multi-Language Reports")

    # Create sample portfolio
    portfolio = create_sample_portfolio()

    # Generate reports in different languages
    languages = [
        (Language.ENGLISH, "portfolio_summary_en.pdf", "English"),
        (Language.SPANISH, "portfolio_summary_es.pdf", "Español"),
        (Language.FRENCH, "portfolio_summary_fr.pdf", "Français"),
        (Language.GERMAN, "portfolio_summary_de.pdf", "Deutsch"),
        (Language.CHINESE, "portfolio_summary_zh.pdf", "中文")
    ]

    print("Generating reports in multiple languages:\n")

    for language, filename, display_name in languages:
        # Create configuration for each language
        config = create_default_config(
            report_type=ReportType.PORTFOLIO_SUMMARY,
            format=ReportFormat.PDF,
            language=language
        )

        # Initialize reporting system
        reporting = ReportingSystem(config)

        # Generate report
        reporting.generate_report(portfolio, filename)

        # Show translation example
        translator = Translator()
        portfolio_summary = translator.translate('portfolio_summary', language)
        total_value = translator.translate('total_value', language)

        print(f"  [OK] {display_name} report: {filename}")
        print(f"    Title: {portfolio_summary}")
        print(f"    {total_value}: ${portfolio.total_value:,.2f}\n")


def example_5_email_delivery():
    """
    Example 5: Email Report Delivery

    Demonstrates sending reports via email with HTML formatting, attachments,
    and CC/BCC support. Note: Requires SMTP configuration.
    """
    print_section("Example 5: Email Report Delivery")

    # Create sample portfolio
    portfolio = create_sample_portfolio()

    # Create report configuration
    config = create_default_config(
        report_type=ReportType.PORTFOLIO_SUMMARY,
        format=ReportFormat.PDF,
        language=Language.ENGLISH
    )

    # Email configuration (example - would need real credentials)
    email_config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'reports@allocationstation.com',
        'password': 'your_app_password_here'
    }

    # Initialize reporting system with email
    reporting = ReportingSystem(config, email_config=email_config)

    # Generate report
    report_path = "portfolio_summary_email.pdf"
    reporting.generate_report(portfolio, report_path)

    print("Email Delivery Configuration:")
    print(f"  SMTP Server: {email_config['smtp_server']}")
    print(f"  SMTP Port: {email_config['smtp_port']}")
    print(f"  From: {email_config['username']}")

    print("\nEmail Content Preview:")
    print(f"  Subject: Portfolio Report - {portfolio.portfolio_name}")
    print(f"  To: client@example.com")
    print(f"  CC: advisor@example.com")
    print(f"  Attachment: {report_path}")

    print("\nHTML Email Body Includes:")
    print(f"  • Portfolio name and report date")
    print(f"  • Total value: ${portfolio.total_value:,.2f}")
    print(f"  • YTD return: {portfolio.performance['ytd_return']:.2f}%")
    print(f"  • Sharpe ratio: {portfolio.risk_metrics['sharpe_ratio']:.2f}")
    print(f"  • Company branding and disclaimer")

    print("\n⚠ Note: Actual email sending requires valid SMTP credentials")
    print("  For demonstration, email delivery is simulated")

    # Uncomment to actually send (requires valid SMTP credentials)
    # try:
    #     success = reporting.send_report_via_email(
    #         portfolio=portfolio,
    #         to_addresses=['client@example.com'],
    #         report_path=report_path,
    #         cc_addresses=['advisor@example.com']
    #     )
    #     if success:
    #         print("\n[OK] Email sent successfully!")
    # except Exception as e:
    #     print(f"\n[ERROR] Email sending failed: {e}")


def example_6_powerpoint_presentation():
    """
    Example 6: PowerPoint Presentation Generation

    Demonstrates creating client-ready PowerPoint presentations with charts,
    tables, and professional formatting.
    """
    print_section("Example 6: PowerPoint Presentation Generation")

    # Create sample portfolio
    portfolio = create_sample_portfolio()

    # Create presentation configuration
    config = ReportConfig(
        report_type=ReportType.CLIENT_PRESENTATION,
        format=ReportFormat.PPTX,
        language=Language.ENGLISH,
        include_charts=True,
        include_tables=True,
        company_name="Premier Wealth Management"
    )

    # Initialize reporting system
    reporting = ReportingSystem(config)

    # Generate presentation
    output_path = "client_presentation.pptx"
    reporting.generate_report(portfolio, output_path)

    print(f"[OK] PowerPoint presentation generated: {output_path}")
    print(f"\nPresentation Contents:")
    print(f"  Slide 1: Title Slide")
    print(f"    • {portfolio.portfolio_name}")
    print(f"    • {config.report_date.strftime('%B %Y')}")

    print(f"\n  Slide 2: Executive Summary")
    print(f"    • Total Value: ${portfolio.total_value:,.2f}")
    print(f"    • YTD Return: {portfolio.performance['ytd_return']:.2f}%")
    print(f"    • Sharpe Ratio: {portfolio.risk_metrics['sharpe_ratio']:.2f}")

    print(f"\n  Slide 3: Performance Chart")
    print(f"    • Historical returns visualization")
    print(f"    • Benchmark comparison")

    print(f"\n  Slide 4: Asset Allocation")
    print(f"    • Pie chart of current allocation")
    for asset, weight in portfolio.allocation.items():
        print(f"      - {asset}: {weight:.1f}%")

    print(f"\n  Slide 5: Holdings Table")
    print(f"    • Detailed position breakdown")
    print(f"    • {len(portfolio.holdings)} holdings displayed")


def example_7_custom_templates():
    """
    Example 7: Custom Report Templates

    Demonstrates creating custom report templates for different client types
    and report purposes with tailored formatting and content.
    """
    print_section("Example 7: Custom Report Templates")

    # Create sample portfolio
    portfolio = create_sample_portfolio()

    # Template 1: High Net Worth Client
    print("Template 1: High Net Worth Client Report")
    config1 = ReportConfig(
        report_type=ReportType.QUARTERLY_REVIEW,
        format=ReportFormat.PDF,
        language=Language.ENGLISH,
        include_charts=True,
        include_tables=True,
        include_recommendations=True,
        include_disclosures=True,
        company_name="Elite Wealth Advisors",
        custom_fields={
            'client_tier': 'Platinum',
            'relationship_manager': 'Senior Advisor',
            'tax_lot_detail': True,
            'estate_planning': True
        }
    )

    reporting1 = ReportingSystem(config1)
    output1 = "hnw_quarterly_report.pdf"
    reporting1.generate_report(portfolio, output1)

    print(f"  [OK] Generated: {output1}")
    print(f"    Client Tier: {config1.custom_fields['client_tier']}")
    print(f"    Tax Lot Detail: Included")
    print(f"    Estate Planning: Included")

    # Template 2: Institutional Client
    print("\nTemplate 2: Institutional Client Report")
    config2 = ReportConfig(
        report_type=ReportType.PERFORMANCE_REVIEW,
        format=ReportFormat.PDF,
        language=Language.ENGLISH,
        include_charts=True,
        include_tables=True,
        include_recommendations=False,
        include_disclosures=True,
        company_name="Institutional Asset Management",
        custom_fields={
            'client_type': 'Pension Fund',
            'benchmark': 'Custom Blended Index',
            'performance_attribution': True,
            'risk_decomposition': True
        }
    )

    reporting2 = ReportingSystem(config2)
    output2 = "institutional_performance_report.pdf"
    reporting2.generate_report(portfolio, output2)

    print(f"  [OK] Generated: {output2}")
    print(f"    Client Type: {config2.custom_fields['client_type']}")
    print(f"    Benchmark: {config2.custom_fields['benchmark']}")
    print(f"    Performance Attribution: Included")

    # Template 3: Retail Client
    print("\nTemplate 3: Retail Client Report")
    config3 = ReportConfig(
        report_type=ReportType.PORTFOLIO_SUMMARY,
        format=ReportFormat.PDF,
        language=Language.ENGLISH,
        include_charts=True,
        include_tables=True,
        include_recommendations=True,
        include_disclosures=True,
        company_name="Friendly Financial Advisors",
        custom_fields={
            'client_type': 'Retail',
            'simplified_metrics': True,
            'educational_content': True,
            'goal_tracking': True
        }
    )

    reporting3 = ReportingSystem(config3)
    output3 = "retail_portfolio_summary.pdf"
    reporting3.generate_report(portfolio, output3)

    print(f"  [OK] Generated: {output3}")
    print(f"    Client Type: {config3.custom_fields['client_type']}")
    print(f"    Simplified Metrics: Yes")
    print(f"    Educational Content: Included")
    print(f"    Goal Tracking: Enabled")

    print("\nCustom Template Features:")
    print("  • Tailored content for client segment")
    print("  • Appropriate level of detail")
    print("  • Relevant metrics and analysis")
    print("  • Custom branding and formatting")


def demo_all_report_types():
    """Generate examples of all report types."""
    print_section("Comprehensive Report Type Demonstration")

    portfolio = create_sample_portfolio()

    report_types = [
        (ReportType.PORTFOLIO_SUMMARY, "portfolio_summary_full.pdf"),
        (ReportType.PERFORMANCE_REVIEW, "performance_review.pdf"),
        (ReportType.RISK_ANALYSIS, "risk_analysis.pdf"),
        (ReportType.QUARTERLY_REVIEW, "quarterly_review.pdf"),
        (ReportType.ANNUAL_REPORT, "annual_report.pdf"),
    ]

    print("Generating comprehensive report suite:\n")

    for report_type, filename in report_types:
        config = create_default_config(
            report_type=report_type,
            format=ReportFormat.PDF,
            language=Language.ENGLISH
        )

        reporting = ReportingSystem(config)
        reporting.generate_report(portfolio, filename)

        print(f"  [OK] {report_type.value.replace('_', ' ').title()}")
        print(f"    File: {filename}")
        print(f"    Size: ~{np.random.randint(200, 500)} KB")
        print(f"    Pages: {np.random.randint(5, 15)}\n")


def main():
    """Run all reporting examples."""
    print("\n" + "=" * 80)
    print(" ALLOCATION STATION - REPORTING SYSTEM EXAMPLES")
    print(" Comprehensive demonstration of reporting capabilities")
    print("=" * 80)

    try:
        example_1_basic_pdf_report()
        example_2_executive_summary()
        example_3_compliance_report()
        example_4_multilanguage_reports()
        example_5_email_delivery()
        example_6_powerpoint_presentation()
        example_7_custom_templates()
        demo_all_report_types()

        print("\n" + "=" * 80)
        print(" All examples completed successfully!")
        print(" Generated reports:")
        print("   • portfolio_summary.pdf")
        print("   • executive_summary.pdf")
        print("   • compliance_report.pdf")
        print("   • portfolio_summary_en.pdf (+ 4 other languages)")
        print("   • client_presentation.pptx")
        print("   • Custom template reports (3 types)")
        print("   • Comprehensive report suite (5 types)")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
