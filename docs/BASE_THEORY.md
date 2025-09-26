# Base Theory: SDEs and Path Signatures

This document outlines the core mathematical and financial theories that form the foundation of the DeepQuant library. It covers the models for asset price simulation and the feature representation of price paths.

## 1. The Probabilistic Setup and American Options

The theoretical framework for modern quantitative finance is built upon a formal probabilistic setup. We begin by defining a **filtered probability space** $`(\Omega, \mathcal{F}, \{\mathcal{F}_t\}_{t \in [0,T]}, \mathbb{P})`$, where:
* $`\Omega`$ is the sample space of all possible outcomes.
* $`\mathcal{F}`$ is a $\sigma$-algebra representing the set of all possible events.
* $\mathbb{P}$ is the probability measure.
* $`\{\mathcal{F}_t\}_{t \in [0,T]}`$ is the **filtration**, an increasing family of $\sigma$-algebras where $\mathcal{F}_t$ represents the information available up to time $t$.

All stochastic processes, such as the asset price $S_t$, are assumed to be **adapted** to this filtration.

The pricing of an American option within this framework is an **optimal stopping problem**. The key challenge is to find the optimal time to exercise to maximize the expected payoff.

**Definition:** The **price of an American option ($V_0$)** is the supremum, or least upper bound, of the expected discounted payoff over the set of all possible stopping times adapted to the filtration $\{\mathcal{F}_t\}$.
*This means we're hunting for the absolute best possible moment to exercise, using only the information available at the time!*

Its value is given by the formula:

$$V_0 = \sup_{\tau \in \mathcal{T}_{[0,T]}} \mathbb{E}\left[e^{-r\tau} \text{Payoff}(S_\tau)\right]$$

where $\tau$ is a stopping time with respect to $\{\mathcal{F}_t\}$.

To solve this, the DeepQuant library uses a **primal-dual approach**. This method computes a mathematically rigorous price interval, trapping the true price between a lower and an upper bound. The difference, the **duality gap**, serves as a built-in measure of accuracy.

## 2. Modeling Financial Markets with SDEs

Stochastic Differential Equations (SDEs) are the mathematical language used to describe the random evolution of asset prices and their volatility. The library's adaptive framework selects the most appropriate SDE model based on market conditions, which are quantified by the **Hurst parameter (H)**.

### Mathematical Formulation of an SDE

**Definition:** A **Stochastic Differential Equation (SDE)** is a differential equation in which at least one term is a stochastic process, resulting in a solution that is also a stochastic process.
*It's a mathematical recipe for describing something that has both a predictable trend and a dose of pure randomness!*

The rigorous mathematical formulation of an SDE is expressed as an integral equation:

$$X_t = X_0 + \int_0^t a(s, X_s)ds + \int_0^t b(s, X_s)dW_s$$

This states that the value of the process at time $t$ is the sum of its initial value ($X_0$), a standard integral for the deterministic trend (drift), and an Itô integral for the cumulative random fluctuations (diffusion).

For convenience, the SDE is often written in its simpler, equivalent differential form:

$$dX_t = a(t, X_t)dt + b(t, X_t)dW_t$$

This shorthand clearly separates the deterministic **drift** ($a(t, X_t)dt$) from the random **diffusion** ($b(t, X_t)dW_t$).

### The Heston Model (Classical Volatility, $H \ge 0.5$)

For markets exhibiting classical random-walk behavior, the library employs the Heston model. It is defined by a system of two SDEs:

1.  **Asset Price ($S_t$)**: 

$$dS_t = r S_t dt + \sqrt{V_t} S_t dW^s_t$$

2.  **Variance ($V_t$)**: 

$$dV_t = \kappa(\theta - V_t)dt + \xi\sqrt{V_t}dW^v_t$$

where:
* $s_0$: The **initial stock price**.
* $v_0$: The **initial variance**.
* $\kappa$ (`kappa`): The **rate of mean reversion**, which controls how quickly the variance process returns to its long-term average.
* $\theta$ (`theta`): The **long-term mean of the variance**.
* $\xi$ (`xi`): The **volatility of variance**, or "vol of vol," which determines the volatility of the variance process itself.
* $\rho$ (`rho`): The **correlation** between the asset's random process ($W^s_t$) and the variance's random process ($W^v_t$).
* $r$: The **risk-free interest rate**.

### The Rough Bergomi Model (Rough Volatility, $H < 0.5$)

To capture the jagged, "rough" nature of volatility observed in real financial data, the library uses the modern rough Bergomi model. Its core component is the **Volterra process**:

1.  **Asset Price ($S_t$)**: 

$$dS_t = r S_t dt + \sqrt{V_t} S_t dB_t$$

2.  **Variance ($V_t$)**: 

$$V_t = V_0 \exp\left(\eta Y_t - \frac{1}{2}\eta^2 t^{2H}\right) \text{ where } Y_t = \int_0^t (t-s)^{H-1/2} \, dW^v_s$$

where:
* $s_0$: The **initial stock price**.
* $v_0$: The **initial forward variance**.
* $H$ (`H`): The **Hurst parameter**, which must be in the range (0, 0.5) to model rough volatility.
* $\eta$ (`eta`): The **volatility of volatility** parameter.
* $\rho$ (`rho`): The **correlation** between the volatility and price processes.
* $r$: The **risk-free interest rate**.

## 3. Path Signatures: A Modern Feature Representation

**Definition:** The **path signature** is a collection of iterated integrals of a path, providing a graded, hierarchical summary of its geometric properties.
*It's the ultimate path summary, turning a wiggly line into a precise list of its essential features!*

For a path $X: [0, T] \to \mathbb{R}^d$, its signature is the infinite sequence:

$$S(X)_{0,T} = \left(1, S^1(X), S^2(X), \dots \right)$$

In practice, the library uses the **truncated signature** up to a specified order. This provides a powerful, finite-dimensional feature vector that serves as the primary input for the library's deep learning solvers.

## 4. The SDE-to-Signature Transformation

The connection between a Stochastic Differential Equation (SDE) and a path signature is a formal mathematical mapping. The SDE generates a sample path, and the signature function transforms that path into a unique, structured feature vector. This document details the formal mathematics of that transformation.

### Step 1: Path Construction from a Discrete Simulation

The process begins with the output of an SDE simulation: a discrete time series of points $(t_0, X_0), (t_1, X_1), \dots, (t_N, X_N)$, where $X_t \in \mathbb{R}^d$.

To prepare this for the signature calculation, two mathematical steps are taken:
1.  The path is **augmented** with the time dimension to create a new path $Y_t \in \mathbb{R}^{d+1}$, where the mathematical definition is:

$$Y_t = (t, X_t)$$
    
3.  A continuous, **piecewise linear path** is constructed by connecting the points. For any time $s$ within a simulation interval $[t_i, t_{i+1}]$, the path is formally defined by the linear interpolation formula:

$$Y_s = Y_{t_i} + \frac{s - t_i}{t_{i+1} - t_i}(Y_{t_{i+1}} - Y_{t_i})$$

### Step 2: Signature Calculation

The signature of the constructed path $Y_s$ is a sequence of terms, where the $M$-th term, $S^M(Y)$, is formally defined as the **iterated integral** of the path up to order `M`:

$$ S^M(Y) = \int_{0< t_1 < \dots < t_M < T} dY_{t_1} \otimes \dots \otimes dY_{t_M} $$

While this is the general form, for a piecewise linear path, it can be calculated exactly. To simplify the notation, we first define the **increment** of the $i$-th linear segment as the vector $\Delta_i$:

$$\Delta_i = Y_{t_{i+1}} - Y_{t_i}$$

#### The First-Order Term: Displacement
The first-order term, $S^1(Y)$, resolves to the path's total displacement. For a piecewise linear path, this is the sum of the increments:

$$S^1(Y) = Y_T - Y_0 = \sum_{i=0}^{N-1} \Delta_i$$

This vector contains the total time elapsed and the total change in the asset's value.

#### A Brief Connection: Deriving the Second-Order Term
The second-order term, $S^2(Y)$, captures the path's area-like properties. The change in this term over a single segment is found to be:

$$\Delta S^2_i = S^1(Y)_{0, t_i} \otimes \Delta_i + \frac{1}{2} \Delta_i^{\otimes 2}$$

The total second-order term is the sum of these changes, $S^2(Y) = \sum_i \Delta S^2_i$.

#### The General Formula
This pattern generalizes to any level `M`. The $M$-th term of the signature is given by the exact recursive formula:

$$S^M(Y) = \sum_{i=0}^{N-1} \sum_{k=1}^{M} \frac{1}{k!} S^{M-k}(Y)_{0, t_i} \otimes \Delta_i^{\otimes k}$$

### Step 3: The Resulting Feature Vector

The final, usable signature is the **truncated signature**, a finite vector containing all computed terms up to level `M`. The mathematical representation of this final vector is the concatenation of the terms:

$$S^M(Y) = \text{concat}\left(S^0(Y), S^1(Y), S^2(Y), \dots, S^M(Y)\right)$$

This formal process provides a deterministic and unique mapping from any given SDE sample path to a rich feature vector.

## 5. The Primal-Dual Framework in Detail

### The Primal Problem: Finding the Lower Bound

The primal problem directly addresses the optimal stopping problem. Its theoretical foundation is the **Bellman principle of dynamic programming**. In a discrete-time setting, the value of the option $V_t$ at time $t$ is the greater of its intrinsic value (if exercised) or its continuation value (if held):

$$V_t = \max\left(\text{Payoff}(S_t), C_t\right)$$

The **continuation value ($C_t$)** is the central quantity. It is formally defined as the discounted expected value of the option at the next time step, conditional on all information available up to the current time, $\mathcal{F}_t$:

$$C_t = \mathbb{E}\left[e^{-r\Delta t} V_{t+\Delta t} | \mathcal{F}_t\right]$$

To solve for $V_0$, one must work backward from maturity ($T$), where $V_T = \text{Payoff}(S_T)$. At each preceding step, the key theoretical challenge is the evaluation of this conditional expectation to determine the optimal exercise decision.

#### The Lower Bound Formulation
The backward induction process yields an estimated continuation value $\hat{C}_t$ for each time step $t$. This allows for the definition of a near-optimal stopping time, $\hat{\tau}$, for each simulated path:

$$\hat{\tau} = \inf \{ t \in \{0, \dots, T\} \mid \text{Payoff}(S_t) \ge \hat{C}_t \}$$

where $\hat{C}_T$ is defined as the intrinsic payoff at maturity. This rule states that the option should be exercised at the first time its immediate payoff value is greater than or equal to the estimated value of holding it.

The resulting lower bound, $V_{\text{lower}}$, is the expected discounted payoff achieved by applying this exercise strategy across all paths:

$$V_{\text{lower}} = \mathbb{E}\left[e^{-r\hat{\tau}} \text{Payoff}(S_{\hat{\tau}})\right]$$

---
### The Dual Problem: Finding the Upper Bound

The dual problem provides a powerful method for finding a provable upper bound on the option's price, based on a key result from martingale theory.

#### Martingale Duality
For any **martingale** process $M_t$ (with $M_0=0$) adapted to the filtration $\{\mathcal{F}_t\}$, the true option price $V_0$ is bounded from above:

$$V_0 \le \mathbb{E}\left[\max_{t \in [0,T]} \left(e^{-rt}\text{Payoff}(S_t) - M_t\right)\right]$$

Since this holds for any martingale, the tightest possible upper bound is found by searching over the space of all valid martingales ($\mathcal{M}$) to find the one that minimizes this expectation:

$$V_{\text{upper}} = \inf_{M \in \mathcal{M}} \mathbb{E}\left[\max_{t \in [0,T]} \left(e^{-rt}\text{Payoff}(S_t) - M_t\right)\right]$$

The theoretical challenge is to solve this minimization problem over an infinite-dimensional space. This is typically approached by parameterizing a rich class of martingales. A general way to construct a martingale is as a stochastic integral $M_t = \int_0^t \alpha_s dW_s$, where the integrand process $\alpha_s$ is adapted to the filtration. The problem then becomes finding the optimal process $\alpha$.

---
### The Duality Gap

The final outputs of the primal and dual formulations provide the price interval $[V_{\text{lower}}, V_{\text{upper}}]$. The difference between these two values is the **duality gap**. A small duality gap is a strong theoretical indicator of an accurate and reliable price, as it means the lower and upper bounds have converged to a narrow range.
