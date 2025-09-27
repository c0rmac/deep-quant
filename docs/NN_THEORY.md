## 1. The Primal Solver: Approximating the Continuation Value

The goal of the primal solver is to find a practical, computable approximation for the theoretical **continuation value**, $C_t$. The derivation follows a three-step process: formalizing the theory, reformulating it with signatures, and then solving the reformulated problem using a function approximator.

---
### 1. The Theoretical Formulation

The starting point is the formal definition of the continuation value as a conditional expectation. It is an unknown function of all information available up to time $t$, represented by the filtration $\mathcal{F}_t$:
$$C_t = \mathbb{E}\left[e^{-r\Delta t} V_{t+\Delta t} | \mathcal{F}_t\right]$$

---
### 2. Reformulation via Signatures

The key to making this problem computable is to replace the abstract filtration $\mathcal{F}_t$ with a concrete numerical representation. This is achieved through a direct mathematical equivalence.

* **The Conditional Expectation is a Function**: The **Doob-Dynkin Lemma**, a fundamental theorem of probability, guarantees that the conditional expectation $C_t$ can be written as some unknown function, $f$, of the information it is conditioned on.
    $$C_t = f(\text{Information contained in } \mathcal{F}_t)$$
* **The Signature Represents the Information**: The path signature, $\text{sig}(\text{Path})_t$, provides a faithful and sufficient numerical representation of the information in the filtration.
* **The Resulting Equivalence**: By substituting the signature for the abstract information, we arrive at an equivalent formulation. The problem is now to find the specific, unknown function $f$ that maps the signature vector to the continuation value.
    $$\underbrace{\mathbb{E}\left[Y | \mathcal{F}_t\right]}_{\substack{\text{The abstract conditional} \\ \text{expectation}}} = \underbrace{f\left(\text{sig}(\text{Path})_t\right)}_{\substack{\text{An unknown function of the} \\ \text{concrete signature vector}}}$$

---
### 3. The Solver as a Function Approximator

With the problem reformulated, the final step is to find an approximation for the unknown function $f$. This is done using Monte Carlo regression.

Given a set of $N$ simulated paths, we can create a dataset for a regression task. For each path $i$:
* **Input ($x_i$)**: The signature of the path up to time $t$, $x_i = \text{sig}(\text{Path}_i)_t$.
* **Target ($y_i$)**: The realized discounted future value, $y_i = e^{-r\Delta t} V_{t+\Delta t}^{(i)}$.

The solver then employs a function approximator to learn the relationship between these inputs and targets. For example, a **Kernel RFF solver** uses a two-stage process:
1.  **Feature Mapping**: The signature, $x_i$, is transformed into a high-dimensional space, $z_i = \phi(x_i)$, using Random Fourier Features to approximate a non-linear kernel.
2.  **Linear Regression**: A simple regularized linear model is fit on these new features.

The trained model provides our best estimate for the unknown function $f$, which we denote as $\hat{f}$. Therefore, our final, computable approximation for the continuation value is the output of this learned function:
$$\hat{C}_t = \hat{f}(\text{sig}(\text{Path})_t) \approx C_t$$

### 4. The Algorithm: Backward Induction

The solver implements the theory by executing a backward induction algorithm. The process begins at maturity ($T$), where the option value for each path $i$ is initialized as $V_T^{(i)} = \text{Payoff}(S_T^{(i)})$. The algorithm then iterates backward for each time step $t = T-1, T-2, \dots, 0$.

At each step $t$, the following mathematical operations are performed:

#### Step 1: Define the Regression Set
The algorithm first identifies the set of path indices, $\mathcal{I}_t$, for which the option is in-the-money:
$$\mathcal{I}_t = \{i \mid \text{Payoff}(S_t^{(i)}) > 0\}$$
The regression is performed on the training set $\{ (x_i, y_i) \}_{i \in \mathcal{I}_t}$, where the inputs and targets are defined as:
* **Inputs ($x_i$)**: $x_i = \text{sig}(\text{Path}_i)_t$
* **Outputs ($y_i$)**: $y_i = e^{-r\Delta t} V_{t+1}^{(i)}$

#### Step 2: Approximate the Continuation Value
The solver finds the optimal function, $\hat{f}_t$, that approximates the continuation value by solving the least-squares problem:
$$\hat{f}_t = \arg\min_{f} \sum_{i \in \mathcal{I}_t} \left( y_i - f(x_i) \right)^2$$This learned function is then used to estimate the continuation value, $\hat{C}_t^{(i)}$, for **all** paths ($i = 1, \dots, N$):$$\hat{C}_t^{(i)} = \hat{f}_t(\text{sig}(\text{Path}_i)_t)$$

#### Step 3: Apply the Optimal Exercise Rule
The option's value for each path at time $t$, $V_t^{(i)}$, is updated according to the optimal exercise rule derived from the Bellman equation:
$$V_t^{(i)} = \begin{cases} \text{Payoff}(S_t^{(i)}) & \text{if } \text{Payoff}(S_t^{(i)}) > \hat{C}_t^{(i)} \text{ and } i \in \mathcal{I}_t \\ e^{-r\Delta t}V_{t+1}^{(i)} & \text{otherwise} \end{cases}$$
The resulting values, $\{V_t^{(i)}\}_{i=1}^N$, are then used as the known future values for the next step of the iteration at time $t-1$.

After the loop completes, the final option price is the sample mean of the values at time $t=0$:
$$V_{\text{lower}} = \frac{1}{N} \sum_{i=1}^{N} V_0^{(i)}$$

### 5. Note: Proving the Equivalence of the Algorithm and Theory

The algorithm's recursion is not just an arbitrary procedure; it is a direct computational method for solving the optimal stopping problem, first proposed by Longstaff and Schwartz (2001). The equivalence between its output and the theoretical $V_{\text{lower}}$ can be formally shown using a proof by **backward induction**.

The goal is to prove that the value $V_t$ calculated by the algorithm at each step is equal to the true theoretical value of the option at that time.

#### 1. The Base Case (at time $t=T-1$)
At the last step of the recursion, the algorithm calculates:
$$V_{T-1} = \max\left(\text{Payoff}(S_{T-1}), \mathbb{E}\left[e^{-r\Delta t}\text{Payoff}(S_T) \mid \mathcal{F}_{T-1}\right]\right)$$
This is precisely the theoretical value of the option at $T-1$, as the only two choices are to exercise now or to wait one period and receive the discounted expected payoff at maturity. The equivalence holds.

#### 2. The Inductive Hypothesis
Let's assume the equivalence holds at time $t+1$. This means the value computed by the algorithm, $V_{t+1}$, is the true theoretical value of the option from that point forward:
$$V_{t+1} = \sup_{\tau \in \mathcal{T}_{t+1,T}} \mathbb{E}\left[e^{-r(\tau-(t+1))} \text{Payoff}(S_\tau) \mid \mathcal{F}_{t+1}\right]$$

#### 3. The Inductive Step (at time $t$)
At time $t$, the algorithm computes the value by applying the Bellman equation:
$$V_t = \max\left(\text{Payoff}(S_t), \mathbb{E}\left[e^{-r\Delta t} V_{t+1} \mid \mathcal{F}_t\right]\right)$$By substituting our inductive hypothesis for $V_{t+1}$ and using the Tower Property of expectation, the second term becomes:$$\mathbb{E}\left[e^{-r\Delta t} V_{t+1} \mid \mathcal{F}_t\right] = \sup_{\tau \in \mathcal{T}_{t+1,T}} \mathbb{E}\left[e^{-r(\tau-t)} \text{Payoff}(S_\tau) \mid \mathcal{F}_t\right]$$This is the value of the "hold" strategy. Therefore, the algorithm's calculation for $V_t$ is:$$V_t = \max\left(\text{Payoff}(S_t), \quad \sup_{\tau \in \mathcal{T}_{t+1,T}} \mathbb{E}\left[e^{-r(\tau-t)} \text{Payoff}(S_\tau) \mid \mathcal{F}_t\right]\right)$$
This is exactly the definition of the true option value at time $t$, which is the maximum of the "exercise now" and "hold" strategies. The equivalence holds at time $t$.

By the principle of backward induction, the value $V_0$ computed by the algorithm is mathematically identical to the theoretical value from the optimal stopping problem. This proves the equivalence:
$$V_{\text{lower}} = \frac{1}{N} \sum_{i=1}^{N} V_0^{(i)} \approx \mathbb{E}\left[e^{-r\hat{\tau}} \text{Payoff}(S_{\hat{\tau}})\right]$$