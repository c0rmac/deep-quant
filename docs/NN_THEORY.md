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

$$\hat{f}_t = \arg\min_{f} \sum_{i \in \mathcal{I}_t} \left( y_i - f(x_i) \right)^2$$

This learned function is then used to estimate the continuation value, $\hat{C}_t^{(i)}$, for **all** paths ($i = 1, \dots, N$):

$$\hat{C}_t^{(i)} = \hat{f}_t(\text{sig}(\text{Path}_i)_t)$$

#### Step 3: Apply the Optimal Exercise Rule
The option's value for each path at time $t$, $V_t^{(i)}$, is updated according to the optimal exercise rule derived from the Bellman equation:

$$V_t^{(i)} = \text{Payoff}(S_t^{(i)}) \quad \text{if } \text{Payoff}(S_t^{(i)}) > \hat{C}_t^{(i)} \text{ and } i \in \mathcal{I}_t$$

$$V_t^{(i)} = e^{-r\Delta t}V_{t+1}^{(i)} \quad \text{otherwise}$$

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

$$V_t = \max\left(\text{Payoff}(S_t), \mathbb{E}\left[e^{-r\Delta t} V_{t+1} \mid \mathcal{F}_t\right]\right)$$

By substituting our inductive hypothesis for $V_{t+1}$ and using the Tower Property of expectation, the second term becomes:

$$\mathbb{E}\left[e^{-r\Delta t} V_{t+1} \mid \mathcal{F}_t\right] = \sup_{\tau \in \mathcal{T}_{t+1,T}} \mathbb{E}\left[e^{-r(\tau-t)} \text{Payoff}(S_\tau) \mid \mathcal{F}_t\right]$$

This is the value of the "hold" strategy. Therefore, the algorithm's calculation for $V_t$ is:

$$V_t = \max\left(\text{Payoff}(S_t), \quad \sup_{\tau \in \mathcal{T}_{t+1,T}} \mathbb{E}\left[e^{-r(\tau-t)} \text{Payoff}(S_\tau) \mid \mathcal{F}_t\right]\right)$$

This is exactly the definition of the true option value at time $t$, which is the maximum of the "exercise now" and "hold" strategies. The equivalence holds at time $t$.

By the principle of backward induction, the value $V_0$ computed by the algorithm is mathematically identical to the theoretical value from the optimal stopping problem. This proves the equivalence:

$$V_{\text{lower}} = \frac{1}{N} \sum_{i=1}^{N} V_0^{(i)} \approx \mathbb{E}\left[e^{-r\hat{\tau}} \text{Payoff}(S_{\hat{\tau}})\right]$$

## 2. The Dual Solver: Approximating the Martingale Integrand

The goal of the dual solver is to find a tight **upper bound** on the American option's price. It achieves this by constructing and optimizing a family of martingales, leveraging the principle of martingale duality for optimal stopping.

---
### 1. The Theoretical Formulation

The dual approach is based on the theoretical result that the true option price $V_0$ is bounded from above by the expected supremum of the payoff minus any suitable martingale process $M_t$. The tightest bound is found by searching over the entire space of valid martingales, $\mathcal{M}$:

$$V_{\text{upper}} = \inf_{M \in \mathcal{M}} \mathbb{E}\left[\sup_{t \in [0,T]} \left(e^{-rt}\text{Payoff}(S_t) - M_t\right)\right]$$

The main theoretical challenge is solving this minimization problem over the infinite-dimensional space of martingales. To make this tractable, martingales are constructed as stochastic integrals of the form $M_t = \int_0^t \alpha_s dW_s$, where $\alpha_s$ is a process adapted to the filtration $\mathcal{F}_s$ known as the martingale integrand.

### 2. Reformulation via Signatures and Neural Networks

The theoretical challenge is to find the optimal integrand process, $\alpha_s$, which is an abstract process adapted to the filtration $\mathcal{F}_s$. To make this problem computationally tractable, we reformulate it by specifying a concrete, parameterized form for $\alpha_s$.

The requirement that $\alpha_s$ be adapted to the filtration means its value at time $s$ can only depend on the history of the path up to that moment. As established in the primal problem, the **Doob-Dynkin Lemma** provides the rigorous justification for expressing this relationship as a function of the path signature. It guarantees the existence of some unknown function, $g$, such that:
$$\alpha_s = g\left(\text{sig}(\text{Path})_s\right)$$

#### The Neural Network as a Function Approximator
The function $g$ is unknown and potentially highly complex. The `DeepSignatureDualSolver` employs a **deep neural network**, $NN_{\theta}$, as a powerful, non-linear function approximator to model $g$. The integrand is therefore approximated as the output of the neural network, which takes the path signature as input:
$$\alpha_s \approx NN_{\theta}\left(\text{sig}(\text{Path})_s\right)$$

#### The Parameterized Martingale
By substituting this neural network approximation into the stochastic integral, we obtain the final, parameterized form of the martingale used by the solver. Its behavior is now controlled entirely by the learnable weights, $\theta$, of the neural network:
$$M_t(\theta) = \int_0^t NN_{\theta}\left(\text{sig}(\text{Path})_s\right) \, dW_s$$In the discrete-time setting of the algorithm, this integral is approximated as a sum of increments over the time steps of the simulation:$$M_t(\theta) = \sum_{j=0}^{t-1} NN_{\theta}(\text{sig}(\text{Path})_j) \cdot \Delta W_{j+1}$$
where $\Delta W_{j+1}$ is the Brownian increment over the interval $[t_j, t_{j+1}]$.

This reformulation successfully transforms the abstract problem of finding an optimal process $\alpha_s$ into a concrete, finite-dimensional optimization problem: finding the optimal network weights $\theta$.

### 3. The Algorithm: Gradient-Based Optimization

With the martingale parameterized by the neural network, the problem of finding the tightest upper bound becomes a finite-dimensional optimization problem over the network's weights, $\theta$:
$$\min_{\theta} \mathbb{E}\left[\max_{t=0,\dots,N} \left(\text{Payoff}_t - M_t(\theta)\right)\right]$$
To solve this in practice, the theoretical **objective function** (the expectation) is connected to a practical **loss function** through Monte Carlo approximation. The true expectation over all possible paths is approximated by the sample mean over the $N$ simulated paths. This sample mean *is* the loss function that the solver seeks to minimize:

$$\underbrace{\mathbb{E}\left[\max_{t} (\text{Payoff}_t - M_t(\theta))\right]}_{\text{Theoretical Objective}} \quad \xrightarrow{\text{approximated by}} \quad \underbrace{\frac{1}{N} \sum_{i=1}^{N} \max_{t} (\text{Payoff}_t^{(i)} - M_t^{(i)}(\theta))}_{\text{Practical Loss Function}}$$

The algorithm then finds the minimum of this loss function using the following steps:

1.  **Gradient Descent**: The solver uses a gradient-based optimizer, such as Adam, to find the optimal weights $\theta^*$ that minimize the loss function. In each step of the optimization, the gradient of the loss with respect to the network weights is computed via backpropagation and used to update the weights.
2.  **Early Stopping**: Training is monitored, and if the loss fails to improve for a set number of epochs, the optimization is stopped early to prevent overfitting and save computation time.

The final, minimized value of the loss function is the computed upper bound price, $V_{\text{upper}}$.

### 4. Choice of Neural Network Architecture

The solver employs a **Residual Network (ResNet)**, enhanced with **Squeeze-and-Excitation (SE)** blocks, to serve as the function approximator. This architecture is deliberately chosen to handle the complexity of the signature features and to ensure a stable and efficient training process.

#### The ResNet Backbone: Learning Hierarchical Refinements
A residual block computes its output, $x_{l+1}$, by adding its input, $x_l$, to a non-linear transformation, $F(x_l)$, of the input:
$$x_{l+1} = F(x_l) + x_l$$
This mathematical form is highly effective for signatures because the **skip connection** ($+ x_l$) acts as an "information highway." It allows the foundational information from the lower-order signature terms (e.g., overall displacement) to be perfectly preserved as it propagates to deeper layers. The network block, $F(x_l)$, is then free to focus only on learning the **refinement** or **perturbation** provided by the more complex, higher-order signature terms (e.g., area and curvature). This structure mirrors the natural hierarchy of the signature itself.

#### The SE Block: Adaptive Feature Recalibration
An SE block acts as an **attention mechanism** that allows the model to learn the dynamic, context-dependent importance of each feature in the signature vector. It performs this through a three-step mathematical process:

1.  **Squeeze**: The block first aggregates the global information from the signature features to produce a summary descriptor. For a vector input, this captures the overall state of the features.
2.  **Excite**: This descriptor is then passed through a small gating mechanism, which is a two-layer neural network with a final sigmoid activation, $\sigma$. This gate outputs a vector of importance scores, $s$, where each score is between 0 and 1. The mathematical form is:
    $$s = \sigma(W_2 \delta(W_1 X))$$
    where $W_1$ and $W_2$ are the learnable weights of the two layers and $\delta$ is a ReLU activation.
3.  **Recalibrate**: The final output of the block is obtained by element-wise multiplying the original signature features, $x_c$, by their learned importance scores, $s_c$:
    $$\tilde{x}_c = s_c \cdot x_c$$

This process of recalibration is what makes the SE block so powerful for signatures. It allows the network to learn the complex, non-linear interdependencies between the different signature terms. For instance, it can learn that if a Level 2 "area" term is high (indicating a volatile path), then a Level 4 "oscillatory" term is highly predictive. In response, the network will learn to output a high score ($s_c \approx 1$) for that Level 4 term, effectively **paying more attention** to it, while suppressing less relevant terms.


#### Architectural Advantage over a Simple MLP
Even if a shallow, single-block MLP had the same number of parameters as a deep ResNet, the ResNet's architecture gives it a significant advantage.

* **Shallow, Wide MLP**: This model attempts to learn the entire input-output relationship in **one single, massive transformation**. It has a higher tendency to simply memorize the training data (overfitting) rather than learning the hierarchical structure of the signature features.
* **Deep ResNet**: This model learns the relationship as a **sequence of simple, iterative refinements**. The first layers learn basic patterns, which are then combined by deeper layers to form more complex patterns. This hierarchical learning process is more efficient and leads to better generalization for structured data like path signatures.

This architectural difference also leads to a more **efficient learning process**. The skip connections in the ResNet create a smoother loss landscape, which allows the gradient-based optimizer to converge faster and more reliably. The SE blocks further enhance this efficiency by acting as a smart filter, directing the model's focus and gradient updates toward the most informative signature terms. This combination results in a model that not only generalizes better but also learns more quickly than a wide MLP, which often struggles with a more complex and difficult-to-navigate loss surface.