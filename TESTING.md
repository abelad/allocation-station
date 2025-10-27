# Testing Guide for Allocation Station

## Table of Contents
- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing Tests](#writing-tests)
- [CI/CD Pipeline](#cicd-pipeline)
- [Test Fixtures](#test-fixtures)
- [Coverage Reports](#coverage-reports)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The Allocation Station project uses a comprehensive testing framework built on pytest, with multiple layers of testing to ensure code quality, performance, and reliability. Our testing infrastructure includes:

- **90% code coverage requirement** enforced in CI/CD
- **6 test categories** with dedicated markers
- **17 test classes** with 61+ test methods
- **Property-based testing** with Hypothesis
- **Performance benchmarks** and stress tests
- **Automated test data generation**
- **CI/CD pipeline** with multi-OS and multi-Python version support

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── test_portfolio.py           # Unit tests for portfolio operations
├── test_integration.py         # Integration tests for workflows
├── test_performance.py         # Performance benchmarks and stress tests
├── test_property_based.py      # Property-based tests using Hypothesis
├── test_regression.py          # Regression tests for bug prevention
└── test_data_generation.py     # Test data generation utilities
```

## Running Tests

### Quick Start

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage report
pytest --cov=src/allocation_station --cov-report=html

# Run specific test file
pytest tests/test_portfolio.py

# Run with verbose output
pytest -v
```

### Running Tests by Category

Tests are organized with markers that allow selective execution:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run performance tests
pytest -m performance

# Run stress tests
pytest -m stress

# Run property-based tests
pytest -m property

# Run regression tests
pytest -m regression

# Run all except slow tests
pytest -m "not slow"

# Run multiple categories
pytest -m "unit or integration"
```

### Running Specific Tests

```bash
# Run a specific test class
pytest tests/test_portfolio.py::TestPortfolio

# Run a specific test method
pytest tests/test_portfolio.py::TestPortfolio::test_portfolio_creation

# Run tests matching a pattern
pytest -k "portfolio"

# Run failed tests from last run
pytest --lf

# Run failed tests first, then others
pytest --ff
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel with auto-detected CPU count
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

## Test Categories

### 1. Unit Tests (`@pytest.mark.unit`)
- **Files**: `test_portfolio.py`, `test_data_generation.py`
- **Purpose**: Test individual components in isolation
- **Coverage**: Portfolio operations, calculations, edge cases
- **Example**:
```python
@pytest.mark.unit
class TestPortfolio:
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        holdings = {'SPY': 100, 'TLT': 150}
        prices = {'SPY': 442.15, 'TLT': 92.30}
        total_value = sum(holdings[k] * prices[k] for k in holdings)
        assert total_value == pytest.approx(58060.0, rel=1e-2)
```

### 2. Integration Tests (`@pytest.mark.integration`)
- **Files**: `test_integration.py`
- **Purpose**: Test complete workflows and component interactions
- **Coverage**: Portfolio analysis, Monte Carlo, broker integration
- **Example**:
```python
@pytest.mark.integration
class TestPortfolioWorkflow:
    def test_full_portfolio_analysis_workflow(self):
        """Test complete portfolio analysis from creation to reporting."""
        # Create portfolio -> Calculate values -> Generate report
```

### 3. Performance Tests (`@pytest.mark.performance`)
- **Files**: `test_performance.py`
- **Purpose**: Benchmark critical operations
- **Coverage**: Calculation speed, Monte Carlo performance
- **Example**:
```python
@pytest.mark.performance
class TestPerformanceBenchmarks:
    def test_monte_carlo_performance(self):
        """Test Monte Carlo simulation performance."""
        # Must complete 1000 simulations in < 5 seconds
```

### 4. Stress Tests (`@pytest.mark.stress`)
- **Files**: `test_performance.py`
- **Purpose**: Test system limits and scalability
- **Coverage**: Large portfolios (10K+ assets), high-frequency data
- **Example**:
```python
@pytest.mark.stress
class TestStressTests:
    def test_large_portfolio_handling(self):
        """Test handling of large portfolio (10,000+ assets)."""
```

### 5. Property-Based Tests (`@pytest.mark.property`)
- **Files**: `test_property_based.py`
- **Purpose**: Test mathematical properties and invariants
- **Coverage**: Portfolio calculations, risk metrics
- **Example**:
```python
@pytest.mark.property
class TestPropertyBasedPortfolio:
    @given(holdings=st.dictionaries(...))
    def test_total_value_always_positive(self, holdings):
        """Property: Total portfolio value should always be positive."""
```

### 6. Regression Tests (`@pytest.mark.regression`)
- **Files**: `test_regression.py`
- **Purpose**: Prevent previously fixed bugs from reoccurring
- **Coverage**: Edge cases, data integrity, validation
- **Example**:
```python
@pytest.mark.regression
class TestRegressionBugs:
    def test_division_by_zero_in_allocation(self):
        """Regression: Division by zero when portfolio value is zero."""
```

## Writing Tests

### Test File Structure

```python
"""Module description."""

import pytest
import numpy as np
from your_module import YourClass


@pytest.mark.unit  # Add appropriate marker
class TestYourClass:
    """Test suite description."""

    def test_feature_one(self):
        """Test description."""
        # Arrange
        input_data = create_test_data()

        # Act
        result = YourClass.process(input_data)

        # Assert
        assert result == expected_value
```

### Using Fixtures

Fixtures provide reusable test data and setup. See `conftest.py` for available fixtures:

```python
def test_with_sample_portfolio(sample_portfolio):
    """Test using the sample_portfolio fixture."""
    assert sample_portfolio['total_value'] > 0

def test_with_mock_broker(mock_broker):
    """Test using the mock_broker fixture."""
    positions = mock_broker.get_positions()
    assert len(positions) > 0

def test_with_temp_directory(temp_directory):
    """Test using temporary directory fixture."""
    test_file = temp_directory / "test.txt"
    test_file.write_text("test data")
    assert test_file.exists()
```

### Writing Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(
    value=st.floats(min_value=0, max_value=1000000),
    rate=st.floats(min_value=-0.5, max_value=0.5)
)
def test_return_calculation(value, rate):
    """Test that return calculations are consistent."""
    final = value * (1 + rate)
    calculated_rate = (final - value) / value if value > 0 else 0
    assert abs(calculated_rate - rate) < 1e-10
```

### Writing Benchmark Tests

```python
def test_performance_benchmark(benchmark):
    """Test using the benchmark fixture."""
    def expensive_operation():
        return sum(range(1000000))

    result = benchmark(expensive_operation)
    assert result > 0
```

## CI/CD Pipeline

### GitHub Actions Workflows

#### Main CI Pipeline (`.github/workflows/ci.yml`)
- **Triggers**: Push to main/develop, pull requests
- **Matrix Testing**: Ubuntu, Windows, macOS × Python 3.9-3.12
- **Steps**:
  1. Linting (black, isort, flake8)
  2. Type checking (mypy)
  3. Unit tests with coverage
  4. Integration tests
  5. Property-based tests
  6. Regression tests
  7. Performance tests (skip on PR)
  8. Coverage upload to Codecov

#### Weekly Stress Tests (`.github/workflows/weekly-stress-test.yml`)
- **Triggers**: Weekly schedule (Sunday 3 AM UTC)
- **Purpose**: Extended stress testing and memory profiling
- **Output**: Stress test report with metrics

#### Security Analysis (`.github/workflows/codeql.yml`)
- **Triggers**: Push, PR, weekly schedule
- **Purpose**: CodeQL security scanning
- **Languages**: Python

### Running CI Locally

```bash
# Simulate CI environment
act push  # Requires act tool

# Run all CI checks manually
black --check src tests
isort --check-only src tests
flake8 src tests
mypy src
pytest --cov=src/allocation_station --cov-fail-under=90
```

## Test Fixtures

### Available Fixtures

Our `conftest.py` provides numerous fixtures for common test scenarios:

#### Data Generation Fixtures
- `sample_portfolio` - Complete portfolio with holdings and prices
- `price_series_data` - OHLCV price data
- `returns_data` - Array of return values
- `correlation_matrix` - Valid correlation matrix
- `monte_carlo_params` - Standard MC simulation parameters
- `monte_carlo_results` - Pre-generated MC results

#### Mock Objects
- `mock_broker` - Mock broker connection with methods
- `mock_data_provider` - Mock market data provider
- `mock_database` - Mock database connection

#### File System
- `temp_directory` - Temporary directory (auto-cleanup)
- `sample_csv_file` - Sample CSV file
- `sample_json_file` - Sample JSON configuration

#### Utilities
- `benchmark` - Simple performance benchmark
- `timer` - Context manager for timing
- `assert_dataframe_equal` - DataFrame comparison helper
- `assert_array_equal` - Array comparison helper
- `assert_close` - Approximate equality helper

### Creating Custom Fixtures

Add new fixtures to `conftest.py`:

```python
@pytest.fixture
def your_custom_fixture():
    """Description of your fixture."""
    # Setup
    data = create_test_data()

    # Provide to test
    yield data

    # Teardown (optional)
    cleanup_resources()
```

## Coverage Reports

### Viewing Coverage

```bash
# Generate HTML coverage report
pytest --cov=src/allocation_station --cov-report=html

# Open report in browser
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux
```

### Coverage Requirements

- **Target**: 90% code coverage (enforced in CI)
- **Configuration**: Set in `pytest.ini`
- **Exclusions**: None currently defined

### Improving Coverage

```bash
# Show missing lines
pytest --cov=src/allocation_station --cov-report=term-missing

# Focus on specific module
pytest --cov=src/allocation_station/portfolio tests/test_portfolio.py

# Generate XML report for IDE integration
pytest --cov=src/allocation_station --cov-report=xml
```

## Best Practices

### 1. Test Organization
- One test class per component/feature
- Descriptive test method names
- Group related tests in classes
- Use appropriate markers

### 2. Test Independence
- Tests should not depend on each other
- Use fixtures for shared setup
- Clean up resources in teardown
- Reset random seeds for reproducibility

### 3. Assertion Best Practices
- Use `pytest.approx()` for floating-point comparisons
- Provide meaningful assertion messages
- Test both success and failure cases
- Check edge cases and boundaries

### 4. Performance Testing
- Set realistic performance targets
- Test with production-like data sizes
- Include memory usage tests
- Document performance baselines

### 5. Property-Based Testing
- Define clear properties/invariants
- Use appropriate strategies
- Handle edge cases with `assume()`
- Run enough examples (default: 100)

### 6. Mock Usage
- Mock external dependencies
- Don't over-mock (test real code)
- Verify mock calls when important
- Use `spec=True` for interface checking

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure package is installed in development mode
pip install -e ".[dev]"

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 2. Fixture Not Found
```bash
# List available fixtures
pytest --fixtures

# Check fixture scope
pytest --fixtures -v
```

#### 3. Slow Tests
```bash
# Profile test execution
pytest --durations=10

# Run without slow tests
pytest -m "not slow"
```

#### 4. Flaky Tests
```bash
# Run test multiple times
pytest --count=10 tests/test_file.py::test_name

# Use pytest-retry
pip install pytest-retry
pytest --retry=3
```

#### 5. Coverage Gaps
```bash
# Detailed coverage report
pytest --cov=src/allocation_station --cov-report=annotate

# Check specific file
coverage annotate src/allocation_station/module.py
```

### Debugging Tests

```python
# Use pytest debugger
def test_with_debugging():
    import pdb; pdb.set_trace()  # Debugger breakpoint
    result = function_to_test()
    assert result == expected

# Or use pytest's --pdb flag
# pytest --pdb  # Drop into debugger on failure
```

### Verbose Output

```bash
# Show test output
pytest -s

# Show detailed test information
pytest -vv

# Show local variables on failure
pytest -l
```

## Contributing

### Adding New Tests

1. Choose appropriate test category and file
2. Add test class with proper marker
3. Write descriptive docstrings
4. Use fixtures for common setup
5. Ensure tests are independent
6. Run locally before submitting PR

### Test Review Checklist

- [ ] Tests have appropriate markers
- [ ] Tests are independent and repeatable
- [ ] Edge cases are covered
- [ ] Performance tests have reasonable limits
- [ ] Fixtures are used appropriately
- [ ] Coverage remains above 90%
- [ ] CI pipeline passes

### Maintaining Tests

- Regularly review and update test data
- Keep performance baselines current
- Update regression tests for new bugs
- Refactor tests when code changes
- Document any special test requirements

## Resources

### Documentation
- [pytest documentation](https://docs.pytest.org/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)

### Tools
- [pytest-cov](https://pytest-cov.readthedocs.io/) - Coverage plugin
- [pytest-xdist](https://github.com/pytest-dev/pytest-xdist) - Parallel execution
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) - Benchmarking
- [pytest-mock](https://github.com/pytest-dev/pytest-mock) - Mock helpers

### Continuous Integration
- [GitHub Actions](https://docs.github.com/en/actions)
- [Codecov](https://docs.codecov.com/)
- [act](https://github.com/nektos/act) - Local GitHub Actions