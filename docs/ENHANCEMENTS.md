## 1. Introduction

The primary objective of the `deep-quant` project is to develop a high-performance, production-grade pricing engine for American-style options under the computationally demanding class of rough stochastic volatility models, with a particular focus on the rough Bergomi model. While foundational to modern financial theory, simpler models fail to capture the steep volatility skews and term structures observed in the market for short-dated options. Rough volatility models, characterized by a low Hurst parameter ($H < 0.5$), provide a state-of-the-art framework for accurately modeling these empirical phenomena.

However, the mathematical sophistication of these models introduces significant numerical challenges that are not present in traditional frameworks. A naive implementation is prone to numerical instability, high variance in price estimates, and computationally prohibitive runtimes, rendering it impractical for real-world applications such as live calibration or comprehensive risk management.

The purpose of this document is to provide a detailed account of the key numerical hurdles encountered during the development of the `deep-quant` pricing engine. It will detail the hierarchy of sophisticated solutions that were systematically implemented to ensure the final engine is accurate, stable, and efficient.

## 2. The Core Challenge: Simulating Rough Volatility

### 2.1 The Problem: Path Instability in the Bergomi Model

The foundational challenge in pricing with the rough Bergomi model is the numerical simulation of its variance process. The model defines variance as an exponential of a fractional Brownian motion, as shown in the equation:

$$V_t = v_0 \exp\left(\eta W_t^H - \frac{1}{2}\eta^2 t^{2H}\right)$$

where $W_t^H$ is a fractional Brownian motion. A naive, vectorized implementation of this can lead to the exponent becoming very large, causing the variance to overflow to `inf` or `NaN` values. This catastrophic instability makes any subsequent pricing calculation impossible. Early attempts to mitigate this with a simple upper clamp on the variance were found to be suboptimal, as a fixed clamp can introduce significant bias, especially when simulating over a wide range of model parameters.

### 2.2 The Solution: A Stable Hybrid Simulation Scheme

The definitive solution was to implement a stable hybrid simulation scheme that is both mathematically sound and numerically robust. This approach completely abolishes the need for artificial clamps. The method consists of two key components:

1.  **Vectorized Log-Variance Path:** The variance process is handled by first simulating its logarithm, which is a numerically well-behaved process:
    $$\log(V_t) = \log(v_0) + \eta W_t^H - \frac{1}{2}\eta^2 t^{2H}$$
    The full path of $\log(V_t)$ is calculated in a single, vectorized operation. The final variance path is then obtained by exponentiating, $V_t = \exp(\log(V_t))$, which mathematically guarantees its positivity.

2.  **Iterative Log-Euler for Stock Price:** With a stable, positive variance path now available, the stock price path, which follows the SDE:
    $$\frac{dS_t}{S_t} = r dt + \sqrt{V_t} dB_t$$
    is simulated using a step-by-step `for` loop. The iterative Log-Euler scheme discretizes the logarithm of the stock price, which is more stable than a direct discretization of $S_t$:
    $$\log(S_{t+1}) = \log(S_t) + \left(r - \frac{1}{2}V_t\right)dt + \sqrt{V_t} \Delta B_t$$

This hybrid approach combines the speed of vectorization where it is safe (for the variance process) with the stability of an iterative loop where it is necessary (for the price process), ensuring the generation of robust and accurate paths.

```python
# --- Pseudocode for the Hybrid Scheme ---

# 1. Vectorized Log-Variance Generation
W_H = generate_fractional_brownian_path(...)
log_V_path = log(v0) + eta * W_H - 0.5 * eta**2 * t**(2*H)
V_path = exp(log_V_path)

# 2. Iterative Stock Price Generation
log_S = array_of_zeros
log_S[0] = log(s0)
for t in range(num_steps):
    sqrt_V = sqrt(V_path[t])
    log_S_increment = (r - 0.5 * V_path[t]) * dt + sqrt_V * delta_B[t]
    log_S[t+1] = log_S[t] + log_S_increment
S_path = exp(log_S)
```

## 3. Quantifying Uncertainty in the Primal and Dual Solvers

### 3.1 The Inherent Challenge: Irreducible Noise in Rough Simulations

The primary numerical challenge in this project stems not from a systematic bias (like overestimation) in the Longstaff-Schwartz algorithm, but from the fundamental nature of the rough volatility models we aim to simulate. Processes with a low Hurst parameter ($H < 0.5$), such as the rough Bergomi model, are characterized by extremely high variance and erratic paths.

As a consequence, even with advanced variance reduction techniques like Randomized Quasi-Monte Carlo (RQMC), the Monte Carlo estimates for the option price **do not converge to a single, fixed point**. Instead, as we increase the number of simulation paths, the price estimates for both the primal (`lower_bound`) and dual (`upper_bound`) solvers begin to **oscillate within a stable confidence interval**. This oscillation is not a sign of failure; it is a correct reflection of the high degree of inherent, irreducible uncertainty in the underlying financial model.

The challenge, therefore, is not to eliminate this oscillation, but to **accurately measure its bounds**.

### 3.2 The Solution: An Ensemble Method for Robust Estimation

To accurately estimate the range of the primal and dual values, we employ a robust **ensemble approach**. Instead of relying on the outcome of a single simulation, we run the entire high-quality simulation and pricing workflow multiple times in parallel to build a distribution of possible outcomes. This "poll of polls" approach acknowledges that a single stochastic simulation can be misleading.

The workflow is as follows:

1.  **Generate Independent Simulations**: We run the entire `simulate_paths_adaptive_rqmc` process `N` times (e.g., 10 times) in parallel. This produces `N` independent, high-quality, and internally converged sets of simulation paths.

2.  **Price Each Simulation**: For each of the `N` sets of paths, we run the full `PricingEngine` once. This involves executing the **`LinearPrimalSolver`** and the `DeepSignatureDualSolver` to obtain a single, high-quality estimate for the `lower_bound` and `upper_bound` for that specific run.

3.  **Create a Distribution**: At the end of this process, we have `N` distinct price intervals. For each run `i`, we calculate a midpoint price, $M_i = (\text{lower\_bound}_i + \text{upper\_bound}_i) / 2$. This gives us a final distribution of high-quality midpoint estimates, as well as distributions for the lower and upper bounds.



### 3.3 Deriving the Final Price and Uncertainty Bounds

This ensemble of results allows us to derive a final, definitive price estimate and a set of uncertainty bounds that correctly quantify the model's inherent randomness.

| Metric                    | Calculation                                               |
|:--------------------------|:----------------------------------------------------------|
| **Final Midpoint Price**  | **Average of the `N` individual midpoint estimates.**     |
| **Midpoint Price 95% CI** | 95% Confidence Interval of the `N` midpoint estimates.    |
| **Lower Bound 95% CI**    | 95% Confidence Interval of the `N` lower-bound estimates. |
| **Upper Bound 95% CI**    | 95% Confidence Interval of the `N` upper-bound estimates. |

This approach provides a single, stable price to use for practical applications, while simultaneously providing a clear and statistically sound measure of the full range within which the true price is expected to lie, directly addressing the oscillation problem.

## 4. Accelerating Monte Carlo Convergence

### 4.1 The Problem: Slow Convergence and Estimate Oscillation

Even with a stable path simulation, the inherent randomness of the Monte Carlo method presents a major performance bottleneck. The high variance of rough volatility paths means that a huge number of simulations are required for the price estimates to stabilize. Simply increasing the number of paths leads to impractically long runtimes, during which the estimates tend to oscillate in a noisy band without reaching a clear convergence point. The core challenge is that standard pseudo-random sampling is inefficient; it can create clusters of sample points and leave large areas of the probability space unexplored, requiring a massive number of paths to achieve even coverage.

### 4.2 The Solution: An Adaptive, High-Quality Sampling Framework

The solution to this performance bottleneck is a two-pronged strategy. Instead of relying on a fixed, brute-force number of simulations, we developed an adaptive framework that is both **smarter** and **more efficient**.

1.  **Efficiency (Better Samples):** We first improve the quality of each individual path by replacing standard pseudo-random numbers with a state-of-the-art sampling method.
2.  **Intelligence (Adaptive Stopping):** We then create a workflow that intelligently determines the minimum number of paths required to achieve a statistically stable result, avoiding any unnecessary computation.

These two components work together to dramatically accelerate the convergence of the simulation.

### 4.3 Technique 1: Advanced Sampling (RQMC + Antithetics)

To improve the quality of our samples, we replaced the standard pseudo-random number generator with a method that combines two powerful techniques: Randomized Quasi-Monte Carlo and antithetic variates.

#### The Randomized Quasi-Monte Carlo (RQMC) Technique

Formally, the fundamental improvement comes from replacing purely pseudo-random numbers with a **low-discrepancy sequence**, such as a Sobol sequence, and then randomizing it.

* **Quasi-Monte Carlo (QMC):** Standard Monte Carlo error decreases very slowly, at a rate of roughly $O(1/\sqrt{N})$, where $N$ is the number of paths. This is because pseudo-random points can form clusters and leave large areas of the probability space unexplored. QMC methods use deterministic, low-discrepancy sequences that are designed to fill the sample space as uniformly as possible. This leads to a much faster convergence rate, often approaching $O(1/N)$.
* **Randomization (The "R" in RQMC):** A pure QMC sequence is deterministic; running it twice yields the exact same points. This prevents the calculation of statistical error. RQMC solves this by applying a **random shift** to the entire sequence for each batch of paths. This ensures that while each individual batch is a highly uniform, low-discrepancy set, the batches themselves are statistically independent and random, which is a requirement for our adaptive stopping algorithm.



This RQMC method is then combined with **antithetic variates**, a technique that uses "mirror-image" paths to cancel out noise, further reducing the variance of each sample.

### 4.4 Technique 2: Intelligent Stopping (Stagnation Detection)

To avoid running the simulation longer than necessary, we implemented an intelligent adaptive stopping rule within the `simulate_paths_adaptive_rqmc` function. Instead of waiting for the confidence interval of the price estimate to shrink below an arbitrarily small tolerance (which may never happen due to the oscillation), the algorithm monitors the **rate of improvement**. It tracks the best confidence interval achieved so far and stops automatically if it fails to make a significant improvement after a set number of new path batches (`patience`). This "stagnation detection" correctly identifies when the simulation has reached the point of diminishing returns, saving a vast amount of computational effort.