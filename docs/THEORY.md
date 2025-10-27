# Theoretical Background

Mathematical and financial theory behind Allocation Station.

## Table of Contents

1. [Modern Portfolio Theory](#modern-portfolio-theory)
2. [Risk Metrics](#risk-metrics)
3. [Portfolio Optimization](#portfolio-optimization)
4. [Monte Carlo Simulation](#monte-carlo-simulation)
5. [Time Series Models](#time-series-models)
6. [Factor Models](#factor-models)
7. [Withdrawal Strategies](#withdrawal-strategies)
8. [Performance Attribution](#performance-attribution)

---

## Modern Portfolio Theory

### Overview

Modern Portfolio Theory (MPT), introduced by Harry Markowitz in 1952, provides a mathematical framework for constructing portfolios that maximize expected return for a given level of risk.

### Portfolio Return

The expected return of a portfolio is the weighted average of individual asset returns:

```
E[Rp] = Σ wi * E[Ri]
```

Where:
- `E[Rp]` = Expected portfolio return
- `wi` = Weight of asset i
- `E[Ri]` = Expected return of asset i
- `Σ wi = 1` (weights sum to 1)

### Portfolio Variance

Portfolio variance considers not just individual asset volatilities, but also their correlations:

```
σp² = Σ Σ wi * wj * σi * σj * ρij
```

Where:
- `σp²` = Portfolio variance
- `σi, σj` = Standard deviations of assets i and j
- `ρij` = Correlation between assets i and j
- `wi, wj` = Weights of assets i and j

In matrix notation:
```
σp² = w' Σ w
```

Where:
- `w` = Vector of weights
- `Σ` = Covariance matrix
- `w'` = Transpose of w

### Diversification Benefit

The key insight of MPT is that portfolio risk is less than the weighted average of individual risks when correlations are less than 1:

```
σp < Σ wi * σi    (when ρij < 1 for some i,j)
```

### Efficient Frontier

The efficient frontier represents portfolios with maximum return for each level of risk. Portfolios on the efficient frontier are optimal.

**Properties:**
- Convex curve in risk-return space
- Portfolios below the frontier are suboptimal
- Portfolios above the frontier are unattainable
- The tangency portfolio (maximum Sharpe ratio) is optimal for risk-averse investors

---

## Risk Metrics

### Volatility (Standard Deviation)

Measures dispersion of returns around the mean:

```
σ = √(1/n Σ (Ri - μ)²)
```

Where:
- `Ri` = Return in period i
- `μ` = Mean return
- `n` = Number of observations

**Annualization:**
```
σannual = σdaily * √252
σannual = σmonthly * √12
```

### Sharpe Ratio

Risk-adjusted return metric:

```
Sharpe Ratio = (E[Rp] - Rf) / σp
```

Where:
- `E[Rp]` = Expected portfolio return
- `Rf` = Risk-free rate
- `σp` = Portfolio standard deviation

**Interpretation:**
- Higher is better
- > 1 is good, > 2 is very good, > 3 is excellent
- Measures excess return per unit of risk

### Sortino Ratio

Similar to Sharpe but only considers downside volatility:

```
Sortino Ratio = (E[Rp] - Rf) / σdownside
```

Where:
```
σdownside = √(1/n Σ min(Ri - Rf, 0)²)
```

### Maximum Drawdown

Largest peak-to-trough decline:

```
MDD = min(Pt / Pmax - 1)
```

Where:
- `Pt` = Portfolio value at time t
- `Pmax` = Maximum portfolio value up to time t

### Value at Risk (VaR)

Maximum expected loss at a given confidence level:

```
VaR(α) = -F⁻¹(α)
```

Where:
- `F⁻¹` = Inverse cumulative distribution function
- `α` = Confidence level (e.g., 0.05 for 95% VaR)

**Methods:**
1. **Parametric (Normal)**: Assumes normal distribution
   ```
   VaR(α) = μ - z(α) * σ
   ```

2. **Historical**: Uses empirical distribution
   ```
   VaR(α) = α-th percentile of historical returns
   ```

3. **Monte Carlo**: Simulates many scenarios

### Conditional VaR (CVaR)

Expected loss given that VaR is exceeded:

```
CVaR(α) = E[Loss | Loss > VaR(α)]
```

Also called Expected Shortfall (ES). More conservative than VaR.

### Beta

Systematic risk relative to market:

```
β = Cov(Rp, Rm) / Var(Rm)
```

Where:
- `Rp` = Portfolio return
- `Rm` = Market return

**Interpretation:**
- β = 1: Moves with market
- β > 1: More volatile than market
- β < 1: Less volatile than market
- β < 0: Inverse relationship with market

### Alpha

Excess return above what CAPM predicts:

```
α = Rp - [Rf + β(Rm - Rf)]
```

Positive alpha indicates outperformance.

### Information Ratio

Risk-adjusted excess return relative to benchmark:

```
IR = (Rp - Rb) / TE
```

Where:
- `Rb` = Benchmark return
- `TE` = Tracking error = σ(Rp - Rb)

### Calmar Ratio

Return relative to maximum drawdown:

```
Calmar = CAGR / |MDD|
```

Higher is better; rewards return while penalizing drawdowns.

---

## Portfolio Optimization

### Mean-Variance Optimization

Maximize Sharpe ratio:

```
max w: (w'μ - Rf) / √(w'Σw)
subject to: Σ wi = 1
            wi ≥ 0
```

Or equivalently, minimize variance for target return:

```
min w: w'Σw
subject to: w'μ = Rtarget
            Σ wi = 1
            wi ≥ 0
```

### Minimum Variance Portfolio

```
min w: w'Σw
subject to: Σ wi = 1
```

**Solution:**
```
w = Σ⁻¹ 1 / (1' Σ⁻¹ 1)
```

Where `1` is a vector of ones.

### Maximum Return Portfolio

```
max w: w'μ
subject to: w'Σw ≤ σtarget²
            Σ wi = 1
```

### Risk Parity

Equal risk contribution from each asset:

```
RC_i = wi * (Σw)i / √(w'Σw) = constant for all i
```

Where `(Σw)i` is the i-th element of `Σw`.

**Iterative Solution:**
```
wi^(t+1) = wi^(t) * (target_RC / RC_i^(t))
```

### Black-Litterman Model

Combines market equilibrium with investor views.

**Prior (Equilibrium) Returns:**
```
Π = λ Σ wmkt
```

Where:
- `λ` = Risk aversion coefficient
- `wmkt` = Market capitalization weights

**Posterior Returns (with views):**
```
E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]
```

Where:
- `τ` = Scaling factor (typically 0.01-0.05)
- `P` = View matrix
- `Q` = View returns
- `Ω` = View uncertainty matrix

### Robust Optimization

Account for estimation uncertainty:

```
min w: w'Σw + κ * ||w - w₀||²
subject to: w'μ ≥ Rmin
            Σ wi = 1
```

Where:
- `κ` = Robustness parameter
- `w₀` = Reference weights

### Kelly Criterion

Optimal position sizing for maximum long-term growth:

```
f* = (p * b - q) / b
```

Where:
- `p` = Win probability
- `q` = Loss probability = 1 - p
- `b` = Win/loss ratio

For continuous returns:
```
f* = (μ - Rf) / σ²
```

This is equivalent to maximizing the Sharpe ratio.

---

## Monte Carlo Simulation

### Geometric Brownian Motion

Standard model for stock prices:

```
dS/S = μ dt + σ dW
```

Where:
- `μ` = Drift (expected return)
- `σ` = Volatility
- `dW` = Wiener process (random walk)

**Discrete Form:**
```
St+1 = St * exp((μ - σ²/2)Δt + σ√Δt * Z)
```

Where `Z ~ N(0,1)`.

### Multivariate Simulation

For multiple correlated assets:

```
R = μ + L * Z
```

Where:
- `R` = Vector of returns
- `μ` = Vector of expected returns
- `L` = Cholesky decomposition of covariance matrix (Σ = LL')
- `Z` = Vector of independent standard normals

**Process:**
1. Generate independent Z ~ N(0,1)
2. Correlate using Cholesky: X = L * Z
3. Scale to desired distribution: R = μ + X

### Importance Sampling

Focus simulations on important regions:

```
E[f(X)] = ∫ f(x) p(x) dx
        = ∫ f(x) [p(x)/q(x)] q(x) dx
        ≈ 1/n Σ f(Xi) [p(Xi)/q(Xi)]
```

Where:
- `p(x)` = Original distribution
- `q(x)` = Importance distribution
- `p/q` = Likelihood ratio

### Variance Reduction

**Antithetic Variates:**
```
For each Z, also simulate -Z
Final estimate = (f(Z) + f(-Z)) / 2
```

Reduces variance by exploiting symmetry.

**Control Variates:**
```
f̃ = f + c(g - E[g])
```

Where `g` is correlated with `f` but has known expectation.

### Simulation Accuracy

Standard error of Monte Carlo estimate:

```
SE = σ / √n
```

To halve standard error, need 4x simulations.

**Confidence Interval:**
```
CI = μ̂ ± z(α/2) * SE
```

---

## Time Series Models

### ARMA Models

AutoRegressive Moving Average:

```
Rt = c + Σ φi*Rt-i + Σ θj*εt-j + εt
```

Where:
- `φi` = AR coefficients
- `θj` = MA coefficients
- `εt` = White noise

### GARCH Models

Generalized AutoRegressive Conditional Heteroskedasticity:

```
Rt = μ + εt
εt = σt * Zt
σt² = ω + α*εt-1² + β*σt-1²
```

Where:
- `Zt ~ N(0,1)`
- `ω > 0, α ≥ 0, β ≥ 0`
- `α + β < 1` for stationarity

**Interpretation:**
- `α`: Impact of recent shocks
- `β`: Volatility persistence
- `α + β`: Volatility mean reversion speed

### Regime-Switching Models

Markov-switching model:

```
Rt | St = μ(St) + σ(St) * εt
P(St+1 = j | St = i) = pij
```

Where:
- `St` = State at time t
- `pij` = Transition probability from state i to j

**Transition Matrix:**
```
P = [p11  p12  ...  p1n]
    [p21  p22  ...  p2n]
    [...  ...  ...  ...]
    [pn1  pn2  ...  pnn]
```

Where each row sums to 1.

### Jump Diffusion

Merton jump diffusion model:

```
dS/S = μ dt + σ dW + J dN
```

Where:
- `J` = Jump size
- `dN` = Poisson process with intensity λ

**Discrete Form:**
```
St+1 = St * exp((μ - σ²/2)Δt + σ√Δt*Z + J*N)
```

Where:
- `Z ~ N(0,1)`
- `N ~ Poisson(λΔt)`
- `J ~ N(μJ, σJ²)`

### Stochastic Volatility

Heston model:

```
dS/S = μ dt + √vt dW1
dvt = κ(θ - vt) dt + σv√vt dW2
```

Where:
- `vt` = Variance at time t
- `κ` = Mean reversion speed
- `θ` = Long-run variance
- `σv` = Volatility of volatility
- `dW1, dW2` = Correlated Wiener processes

---

## Factor Models

### Single-Factor CAPM

```
E[Ri] = Rf + βi(E[Rm] - Rf)
```

Where:
- `βi` = Cov(Ri, Rm) / Var(Rm)

### Fama-French Three-Factor

```
Ri - Rf = αi + βM(RM - Rf) + βS*SMB + βV*HML + εi
```

Where:
- `SMB` = Small Minus Big (size factor)
- `HML` = High Minus Low (value factor)

### Carhart Four-Factor

Adds momentum:

```
Ri - Rf = αi + βM(RM - Rf) + βS*SMB + βV*HML + βMOM*MOM + εi
```

Where:
- `MOM` = Momentum factor (winners minus losers)

### Fama-French Five-Factor

Adds profitability and investment:

```
Ri - Rf = α + βM*MKT + βS*SMB + βV*HML + βP*RMW + βI*CMA + ε
```

Where:
- `RMW` = Robust Minus Weak (profitability)
- `CMA` = Conservative Minus Aggressive (investment)

### Arbitrage Pricing Theory (APT)

```
E[Ri] = Rf + Σ βik * λk
```

Where:
- `βik` = Sensitivity to factor k
- `λk` = Risk premium for factor k

No restriction on number or identity of factors.

### Principal Component Analysis (PCA)

Reduce dimensionality of returns:

```
R = U * Σ * V'
```

Where:
- `U` = Left singular vectors (time factors)
- `Σ` = Singular values (factor importance)
- `V` = Right singular vectors (asset loadings)

First few principal components often explain most variance.

---

## Withdrawal Strategies

### 4% Rule

```
W1 = P0 * 0.04
Wt = W1 * (1 + inflation)^(t-1)
```

Where:
- `W1` = First year withdrawal
- `P0` = Initial portfolio value

### Guyton-Klinger

Dynamic withdrawals with guardrails:

```
If Pt/P0 > (1 + upper_guard):
    Wt = Wt-1 * (1 + inflation) * (1 + adjustment)

If Pt/P0 < (1 - lower_guard):
    Wt = Wt-1 * (1 + inflation) * (1 - adjustment)

Otherwise:
    Wt = Wt-1 * (1 + inflation)
```

### Variable Percentage

```
Wt = Pt * withdrawal_rate
subject to: floor ≤ Wt ≤ ceiling
```

### Dynamic Programming

Optimal withdrawal solves:

```
V(Pt, t) = max Wt: U(Wt) + β * E[V(Pt+1, t+1)]
subject to: Pt+1 = (Pt - Wt) * Rt+1
```

Where:
- `V` = Value function
- `U` = Utility function (e.g., CRRA)
- `β` = Discount factor

**CRRA Utility:**
```
U(W) = W^(1-γ) / (1-γ)
```

Where `γ` = Risk aversion coefficient.

### Optimal Portfolio-Consumption

Merton's solution:

```
π* = (μ - r) / (γ * σ²)
C* = (r + π*²σ²/2 - ρ) / γ * W
```

Where:
- `π*` = Optimal risky asset weight
- `C*` = Optimal consumption
- `ρ` = Time preference rate
- `γ` = Risk aversion
- `W` = Wealth

---

## Performance Attribution

### Brinson Attribution

Decomposes performance vs benchmark:

```
Total Active Return = Allocation Effect + Selection Effect + Interaction Effect
```

**Allocation Effect:**
```
AA = Σ (wPi - wBi) * RBi
```

**Selection Effect:**
```
AS = Σ wBi * (RPi - RBi)
```

**Interaction Effect:**
```
AI = Σ (wPi - wBi) * (RPi - RBi)
```

Where:
- `wP, wB` = Portfolio and benchmark weights
- `RP, RB` = Portfolio and benchmark returns
- `i` = Asset or sector index

### Factor Attribution

Decompose returns into factor exposures:

```
Rt = αt + Σ βik * Fkt + εt
```

**Return Attribution:**
```
Total Return = α + Σ (βik * Fk) + ε
```

Each term represents contribution from:
- `α`: Manager skill
- `βik * Fk`: Factor k exposure
- `ε`: Idiosyncratic return

### Risk Attribution

Decompose portfolio variance:

```
σP² = Σ Σ wi * wj * Cov(Ri, Rj)
```

**Marginal Contribution to Risk:**
```
MCRi = ∂σP/∂wi = (Σw)i / σP
```

**Component Contribution to Risk:**
```
CCRi = wi * MCRi
```

And: `Σ CCRi = σP`

### Time-Weighted vs Money-Weighted Returns

**Time-Weighted (TWR):**
```
RTWR = ∏(1 + Rt) - 1
```

Geometric average of period returns. Measures manager performance independent of timing of cash flows.

**Money-Weighted (MWR/IRR):**
```
0 = -PV0 + Σ CFt/(1 + RMWR)^t + PVn/(1 + RMWR)^n
```

Internal rate of return. Includes impact of cash flow timing.

**Use Cases:**
- TWR: Manager evaluation
- MWR: Investor experience

---

## Statistical Tests

### Sharpe Ratio Significance

Test if Sharpe ratio significantly differs from zero:

```
t = SR * √n
```

Where `n` = number of observations.

Under H0 (SR = 0): `t ~ N(0,1)` for large n.

### Alpha Significance

Test if alpha is significant:

```
t = α / SE(α)
```

Where:
```
SE(α) = σε / √n
```

### Correlation Significance

Test if correlation significantly differs from zero:

```
t = r * √(n-2) / √(1-r²)
```

Under H0 (ρ = 0): `t ~ t(n-2)`.

---

## Assumptions and Limitations

### MPT Assumptions

1. **Normal Returns**: Returns are normally distributed
   - Reality: Fat tails, skewness
   - Solution: Use robust statistics, CVaR

2. **Constant Parameters**: μ and Σ are constant
   - Reality: Time-varying
   - Solution: Rolling windows, regime-switching

3. **Single Period**: Single-period optimization
   - Reality: Multi-period with path dependence
   - Solution: Dynamic programming, stochastic control

4. **No Constraints**: Can hold any weights
   - Reality: Short constraints, transaction costs
   - Solution: Constrained optimization

5. **No Taxes**: Tax-free trading
   - Reality: Taxes matter
   - Solution: After-tax optimization

### Estimation Error

Parameter estimates have uncertainty:

```
Var(μ̂) = Σ / n
Var(Σ̂) = complicated...
```

**Impact:**
- Optimization magnifies estimation errors
- Out-of-sample performance often disappoints

**Solutions:**
- Shrinkage estimators (Ledoit-Wolf)
- Robust optimization
- Regularization
- Bayesian methods (Black-Litterman)

### Model Risk

All models are approximations:

> "All models are wrong, but some are useful." - George Box

**Mitigation:**
- Use multiple models
- Stress test assumptions
- Regular backtesting
- Monitor out-of-sample performance

---

## Further Reading

### Classic Papers

1. Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*
2. Sharpe, W. (1964). "Capital Asset Prices." *Journal of Finance*
3. Black, F. & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal*
4. Fama, E. & French, K. (1993). "Common Risk Factors." *Journal of Financial Economics*

### Books

1. **Portfolio Theory:**
   - Merton, R. (1992). *Continuous-Time Finance*
   - Grinold, R. & Kahn, R. (1999). *Active Portfolio Management*

2. **Risk Management:**
   - Jorion, P. (2006). *Value at Risk*
   - McNeil, A. et al. (2015). *Quantitative Risk Management*

3. **Monte Carlo:**
   - Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*

4. **Empirical Finance:**
   - Campbell, J. et al. (1997). *The Econometrics of Financial Markets*

---

**Last Updated**: January 2025
**Version**: 0.1.0
