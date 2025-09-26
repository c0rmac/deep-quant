import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from typing import Callable, Optional
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

from .base_solver import AbstractPrimalSolver, AbstractDualSolver
from ..features.signature_calculator import calculate_signatures


# --- 1. Define the Neural Network Architecture ---

class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for function approximation.

    This network is a generic function approximator that will be used to model:
    1.  The continuation value in the `DeepSignaturePrimalSolver`.
    2.  The martingale integrand in the `DeepSignatureDualSolver`.

    The architecture consists of two hidden layers with ReLU activation functions.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add Dropout after activation

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add Dropout after activation

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# The SEBlock itself remains unchanged
class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c = x.shape
        y = self.squeeze(x.view(bs, c, 1)).view(bs, c)
        y = self.excitation(y).view(bs, c)
        return x * y


# UPGRADED: The ResidualBlock is now configurable
class ResidualBlock(nn.Module):
    def __init__(self, size: int, dropout_rate: float, use_se: bool = True):
        super().__init__()
        self.linear_1 = nn.Linear(size, size)
        self.norm_1 = nn.BatchNorm1d(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(size, size)
        self.norm_2 = nn.BatchNorm1d(size)

        # **THE FIX**: Conditionally create the SE block
        self.se_block = SEBlock(channels=size) if use_se else None

    def forward(self, x):
        identity = x
        out = F.relu(self.norm_1(self.linear_1(x)))
        out = self.dropout(out)
        out = self.norm_2(self.linear_2(out))

        # **THE FIX**: Conditionally apply the SE block
        if self.se_block:
            out = self.se_block(out)

        out += identity
        return F.relu(out)


# UPGRADED: The SimpleResNet now takes the flag and passes it down
class SimpleResNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int, dropout_rate: float,
                 use_se: bool = True):
        super().__init__()
        self.initial_layer = nn.Linear(input_dim, hidden_dim)

        # Pass the use_se flag down to each residual block
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate, use_se=use_se) for _ in range(num_blocks)]
        )
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.final_layer(x)

# --- 2. The Deep Primal Solver ---

class DeepSignaturePrimalSolver(AbstractPrimalSolver):
    """
    Primal solver using a deep neural network to approximate the continuation value.

    This solver follows the Longstaff-Schwartz algorithm but replaces the
    classical regression step (like linear or kernel regression) with a more
    powerful, non-linear function approximator in the form of a neural network.

    The network learns the complex relationship between the path's signature history
    and the option's expected future value. The performance and accuracy of this
    solver are highly dependent on the neural network's architecture and training
    parameters, which must be tuned to balance between learning the underlying
    signal and avoiding fitting to the noise of the Monte Carlo simulation.
    """

    def __init__(self, truncation_level: int, risk_free_rate: float, hidden_dim: int, epochs: int, batch_size: int,
                 lr: float):
        r"""
        Initializes the DeepSignaturePrimalSolver with its architecture and training parameters.

        The parameters set here control the behavior of the neural network used to
        approximate the continuation value, $C_t$, at each step of the backward
        induction. The continuation value is the key to the optimal stopping decision:

        $$ V_t = \max(\text{Payoff}(S_t), C_t) $$
        $$ C_t \approx NN(\text{sig}(\text{Path}_t); \theta) $$

        The training hyperparameters (`epochs_end`, `epochs_step`) implement a
        transfer learning heuristic that provides strong regularization to prevent
        the neural network from overfitting to Monte Carlo noise.

        Args:
            truncation_level (int): The level to which path signatures are truncated.
                This determines the input dimension of the neural network.
            risk_free_rate (float): The risk-free interest rate (r), used for
                discounting future values in the regression target.
            hidden_dim (int): The width (number of neurons) of the neural network's
                hidden layers. This controls the model's capacity to learn complex functions.
            epochs_end (int): The number of epochs for the full, initial training
                of the neural network at the final time step (T-dt).
            epochs_step (int): The number of epochs for the fine-tuning steps at all
                prior time steps (t < T-dt). This is typically set to 1 to implement
                the transfer learning heuristic.
            batch_size (int): The number of samples per batch used during the
                neural network's training.
            lr (float): The learning rate for the Adam optimizer, which controls
                the step size during gradient descent.
        """
        self.truncation_level = truncation_level
        self.r = risk_free_rate
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def solve(self, paths: torch.Tensor, payoff_fn: Callable, **kwargs) -> float:
        """
        Executes the adaptive deep signature-based Longstaff-Schwartz algorithm.

        This method takes the simulated path data and runs the full backward
        induction loop, training a new continuation value estimator at each time
        step to ultimately calculate the option price at time t=0.

        Args:
            paths (torch.Tensor): A 3D tensor of simulated asset paths.
            payoff_fn (Callable): The option's payoff function.
            **kwargs: Must contain 'T' (float), the time to maturity.

        Returns:
            The calculated primal (lower bound) price of the American option.
        """
        # --- 1. Initialization ---
        # Unpack necessary parameters and set up the environment.
        device = paths.device
        num_steps = paths.shape[1]
        T = kwargs.get('T')
        if T is None:
            raise ValueError("Maturity 'T' must be provided in kwargs.")

        # Calculate the time step size and the constant discount factor for one period.
        dt = T / (num_steps - 1)
        discount_factor = torch.exp(torch.tensor(-self.r * dt, device=device))

        # Isolate the asset price paths for calculating payoffs.
        asset_paths = paths[:, :, 0]

        # At the final time step (maturity), the option's value is exactly its intrinsic payoff.
        # This is the starting point for our backward induction.
        option_values = payoff_fn(asset_paths[:, -1])

        # --- 2. Backward Induction Loop ---
        # We step backwards in time from T-dt down to t=0, calculating the
        # option's value at each step.
        for t in range(num_steps - 2, -1, -1):

            # a. Calculate the value if exercised immediately at the current time t.
            intrinsic_value = payoff_fn(asset_paths[:, t])

            # b. Identify the paths where immediate exercise has a positive value.
            # The continuation value is only meaningful for these "in-the-money" paths.
            in_the_money_mask = intrinsic_value > 0

            # If no paths are in-the-money, there is no exercise decision to make.
            # The value of holding is simply the discounted value from the next step.
            if not torch.any(in_the_money_mask):
                option_values = option_values * discount_factor
                option_values = torch.maximum(intrinsic_value, option_values)
                continue

            # --- c. Continuation Value Estimation via Neural Network ---

            # The features for our regression are the signatures of the path history up to time t.
            # Slicing `[:, :t + 1, :]` is critical to prevent lookahead bias.
            paths_slice = paths[:, :t + 1, :]
            signatures = calculate_signatures(paths_slice, self.truncation_level)

            # Ensure the boolean mask is 1D for correct tensor indexing.
            final_mask = in_the_money_mask.squeeze()
            if final_mask.dim() > 1:
                final_mask = final_mask[:, 0]

            # Prepare the training data (X, Y) for the neural network.
            # X = The signatures of the in-the-money paths.
            # Y = The known, discounted future values for those same paths.
            X_train = signatures[final_mask]
            Y_train = (option_values[final_mask] * discount_factor).unsqueeze(1)

            # **Crucial Step**: Standardize the signature features. Neural networks are highly
            # sensitive to the scale of input data; this step is vital for stable training.
            scaler = StandardScaler()
            X_train_np = X_train.cpu().numpy()
            X_train_scaled_np = scaler.fit_transform(X_train_np)
            X_train_scaled = torch.from_numpy(X_train_scaled_np).to(device, dtype=torch.float32)

            # --- d. PyTorch Training Loop ---
            # At each time step, we train a new neural network from scratch to learn the
            # specific continuation value function for that time.
            sig_dim = signatures.shape[1]
            model = SimpleMLP(input_dim=sig_dim, hidden_dim=self.hidden_dim, output_dim=1, dropout_rate=0.5).to(device)
            #model = SimpleResNet(input_dim=sig_dim, hidden_dim=self.hidden_dim, output_dim=1, num_blocks=4,
            #                     dropout_rate=0.5).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
            loss_fn = nn.MSELoss()

            # Use a DataLoader to handle batching of the training data.
            dataset = TensorDataset(X_train_scaled, Y_train)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            print(f"  [Time Step t={t}] Training continuation value estimator...")
            model.train()  # Set the model to training mode (enables dropout, etc.)
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                num_batches = 0
                for x_batch, y_batch in loader:
                    optimizer.zero_grad()
                    y_pred = model(x_batch.to(torch.float32))  # Forward pass
                    loss = loss_fn(y_pred, y_batch.to(torch.float32))  # Compute loss
                    loss.backward()  # Backpropagate gradients
                    optimizer.step()  # Update weights
                    epoch_loss += loss.item()
                    num_batches += 1

                # Print the average training loss for this epoch at regular intervals.
                if (epoch + 1) % 20 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"    Epoch [{epoch + 1}/{self.epochs}], Regressor Loss (MSE): {avg_loss:.6f}")

            # --- e. Prediction and Optimal Decision ---

            # After training, use the model to predict the continuation value for ALL paths.
            model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
            with torch.no_grad():
                # Apply the SAME scaling that was learned from the training data.
                signatures_scaled_np = scaler.transform(signatures.cpu().numpy())
                signatures_scaled = torch.from_numpy(signatures_scaled_np).to(device, dtype=torch.float32)
                continuation_value = model(signatures_scaled).squeeze(-1)

            # The core Longstaff-Schwartz decision rule:
            # The value at time t is the intrinsic value if we exercise, otherwise it's the
            # actual (path-specific) discounted future value if we hold.
            option_values = torch.where(
                (intrinsic_value > continuation_value) & in_the_money_mask,
                intrinsic_value,
                option_values * discount_factor
            )

        # --- 3. Final Price Calculation ---
        # The final option price is the average of all path values at time t=0.
        final_price = torch.mean(option_values)
        return final_price.item()

# --- 3. The Deep Dual Solver ---

class DeepSignatureDualSolver(AbstractDualSolver):
    """
    Dual solver where the martingale integrand is modeled by a neural network.

    This implementation is optimized for performance using a vectorized, full-batch
    approach. It pre-computes all necessary non-anticipative signatures once,
    then uses fully vectorized operations within the main training loop to construct
    the martingale, avoiding the performance bottleneck of nested Python loops.
    """

    def __init__(self, truncation_level: int, hidden_dim: int, learning_rate: float, max_epochs: int, patience: int,
                 tolerance: float, batch_size: Optional[int] = None, num_res_net_blocks: int = 1):
        r"""
        Initializes the DeepSignatureDualSolver with its architecture and training parameters.

        This solver finds an upper bound for the American option price by constructing
        a martingale, $M_t$, and solving the dual optimization problem:

        $$ \text{Upper Bound} = \min_{\theta} \mathbb{E}\left[\max_{t=0,\dots,N} (\text{Payoff}_t - M_t(\theta))\right] $$

        The martingale is constructed as a stochastic integral where the integrand is
        a neural network, $NN(\cdot; \theta)$, that takes the path signature as input:

        $$ M_t(\theta) = \int_0^t NN(\text{sig}(\text{Path}_s); \theta) \, dW_s $$

        The optimizer's goal is to find the optimal neural network weights, $\theta$,
        that create the best martingale to minimize the expectation.

        Args:
            truncation_level (int): The signature truncation level.
            hidden_dim (int): The width of the neural network's hidden layers.
            learning_rate (float): The initial learning rate for the Adam optimizer.
            max_epochs (int): The maximum number of training epochs.
            patience (int): The number of epochs for early stopping.
            tolerance (float): The minimum change in loss for an improvement.
            batch_size (Optional[int]): If an integer is provided, the solver will use
                stochastic mini-batch gradient descent. If None, it will use the
                faster full-batch vectorized approach. Defaults to None.
        """
        self.truncation_level = truncation_level
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.num_res_net_blocks = num_res_net_blocks

    def solve(self, paths: torch.Tensor, dW: torch.Tensor, payoff_fn: Callable, **kwargs) -> float:
        """
        Executes the dual pricing algorithm using either a full-batch or mini-batch approach.
        """
        # --- 1. Initialization and Pre-computation ---
        device, num_steps, num_paths = paths.device, paths.shape[1], paths.shape[0]
        sig_dim = calculate_signatures(paths[:, :2, :], self.truncation_level).shape[1]

        # Pre-compute all non-anticipative signatures once for efficiency.
        print("Pre-computing all path signatures to create integrands...")
        all_integrands_unscaled = torch.zeros(num_paths, num_steps - 1, sig_dim, device=device, dtype=torch.float32)
        for t in range(num_steps - 1):
            paths_slice = paths[:, :t + 1, :]
            if paths_slice.shape[1] > 1:
                all_integrands_unscaled[:, t, :] = calculate_signatures(paths_slice, self.truncation_level)

        # Apply a stable, non-anticipative scaling
        final_time_integrands = all_integrands_unscaled[:, -1, :]
        mean = final_time_integrands.mean(dim=0, keepdim=True)
        std = final_time_integrands.std(dim=0, keepdim=True) + 1e-8
        all_integrands_scaled = (all_integrands_unscaled - mean.unsqueeze(1)) / std.unsqueeze(1)

        # --- 2. Optimization Setup ---
        model = SimpleResNet(input_dim=sig_dim, hidden_dim=self.hidden_dim, output_dim=1, num_blocks=self.num_res_net_blocks,
                             dropout_rate=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        asset_paths = paths[:, :, 0]
        payoffs = torch.stack([payoff_fn(asset_paths[:, t]) for t in range(num_steps)], dim=1)
        best_loss = float('inf')
        patience_counter = 0

        # --- 3. Select Training Mode ---
        if self.batch_size is None:
            # --- 3a. Full-Batch Vectorized Training (Fastest) ---
            print(f"Starting {self.__class__.__name__} optimization (Full-Batch mode)...")
            all_integrands_reshaped = all_integrands_scaled.view(-1, sig_dim)

            for epoch in range(self.max_epochs):
                optimizer.zero_grad()
                final_integrands_reshaped = model(all_integrands_reshaped)
                final_integrands = final_integrands_reshaped.view(num_paths, num_steps - 1, 1)
                increments = final_integrands * dW.unsqueeze(-1)
                martingale_path = torch.cat(
                    [torch.zeros(num_paths, 1, 1, device=device), torch.cumsum(increments, dim=1)], dim=1).squeeze(-1)

                loss = torch.mean(torch.max(payoffs - martingale_path, dim=1)[0])
                if torch.isnan(loss) or torch.isinf(loss): return best_loss if best_loss != float('inf') else float(
                    'nan')
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                if best_loss - loss.item() > self.tolerance:
                    best_loss, patience_counter = loss.item(), 0
                else:
                    patience_counter += 1
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch [{epoch + 1}/{self.max_epochs}], Loss (Upper Bound): {loss.item():.6f}")
                if patience_counter >= self.patience:
                    print(f"  -> Early stopping triggered at epoch {epoch + 1}.")
                    break
        else:
            # --- 3b. Mini-Batch Stochastic Training (More Granular) ---
            print(f"Starting {self.__class__.__name__} optimization (Mini-Batch size: {self.batch_size})...")
            path_indices = torch.arange(num_paths)
            dataset = TensorDataset(path_indices)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

            for epoch in range(self.max_epochs):
                model.train()
                for i, (batch_indices_tuple,) in enumerate(loader):
                    batch_indices = batch_indices_tuple.to(device)
                    optimizer.zero_grad()

                    batch_integrands = all_integrands_scaled[batch_indices]
                    batch_integrands_reshaped = batch_integrands.view(-1, sig_dim)

                    final_integrands_reshaped = model(batch_integrands_reshaped)
                    final_integrands = final_integrands_reshaped.view(len(batch_indices), num_steps - 1, 1)

                    batch_dW = dW[batch_indices]
                    increments = final_integrands * batch_dW.unsqueeze(-1)

                    martingale_path = torch.cat(
                        [torch.zeros(len(batch_indices), 1, 1, device=device), torch.cumsum(increments, dim=1)],
                        dim=1).squeeze(-1)

                    batch_payoffs = payoffs[batch_indices]
                    loss = torch.mean(torch.max(batch_payoffs - martingale_path, dim=1)[0])
                    if torch.isnan(loss) or torch.isinf(loss): continue

                    loss.backward()
                    optimizer.step()

                # Evaluate loss on the full dataset for stable scheduler and early stopping updates
                model.eval()
                with torch.no_grad():
                    full_integrands_reshaped = all_integrands_scaled.view(-1, sig_dim)
                    final_integrands_reshaped = model(full_integrands_reshaped)
                    final_integrands = final_integrands_reshaped.view(num_paths, num_steps - 1, 1)
                    increments = final_integrands * dW.unsqueeze(-1)
                    full_martingale = torch.cat(
                        [torch.zeros(num_paths, 1, 1, device=device), torch.cumsum(increments, dim=1)], dim=1).squeeze(
                        -1)
                    epoch_loss = torch.mean(torch.max(payoffs - full_martingale, dim=1)[0])

                scheduler.step(epoch_loss)
                if best_loss - epoch_loss.item() > self.tolerance:
                    best_loss, patience_counter = epoch_loss.item(), 0
                else:
                    patience_counter += 1
                if (epoch + 1) % 20 == 0:
                    print(
                        f"  Epoch [{epoch + 1}/{self.max_epochs}], Full-Batch Loss (Upper Bound): {epoch_loss.item():.6f}")
                if patience_counter >= self.patience:
                    print(f"  -> Early stopping triggered at epoch {epoch + 1}.")
                    break

        return best_loss