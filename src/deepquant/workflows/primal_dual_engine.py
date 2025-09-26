import torch
from typing import Dict, Optional

from src.deepquant.models.sde import SDEModel
from src.deepquant.solvers.base_solver import AbstractPrimalSolver, AbstractDualSolver
from src.deepquant.utils import payoff_factory


class PricingEngine:
    """
    Orchestrates the execution of modular primal and dual solvers to price an American option.

    This class encapsulates the end-to-end workflow for a single pricing run.
    It is initialized with a specific SDE model and concrete solver implementations
    (e.g., LinearPrimalSolver, LinearDualSolver). Its main `run` method simulates
    the required paths and then invokes the primal (lower bound) and dual (upper bound)
    solvers to compute the final price interval and duality gap.

    This modular design, based on the Strategy Pattern, allows for easy comparison
    of different solver methodologies (linear, kernel, deep learning) without
    changing the core engine logic.
    """

    def __init__(
            self,
            sde_model: SDEModel,
            primal_solver: AbstractPrimalSolver,
            dual_solver: AbstractDualSolver,
            option_type: str,
            strike: float,
            device: Optional[str] = None
    ):
        """
        Initializes the pricing engine with modular solver components.

        Args:
            sde_model (SDEModel): An instantiated SDE model (e.g., HestonModel) that
                                  can simulate paths and Brownian increments.
            primal_solver (AbstractPrimalSolver): A concrete implementation of a primal solver.
            dual_solver (AbstractDualSolver): A concrete implementation of a dual solver.
            option_type (str): The type of option to price, either 'put' or 'call'.
            strike (float): The strike price of the option.
            device (Optional[str]): The PyTorch device to use ('cuda', 'cpu', or 'mps').
                                    If None, it's auto-detected.
        """
        self.model = sde_model
        self.primal_solver = primal_solver
        self.dual_solver = dual_solver
        self.payoff_fn = payoff_factory(option_type, strike)

        # Auto-detect device if not explicitly provided
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if torch.backends.mps.is_available(): self.device = 'mps'
        else:
            self.device = device

        print("Primal-Dual Engine Initialized.")
        print(f"  -> Using device: {self.device}")

    def run(self, num_paths: int, num_steps: int, T: float) -> Dict[str, float]:
        """
        Runs a full pricing simulation to calculate the primal and dual bounds.

        Args:
            num_paths (int): The number of Monte Carlo paths to simulate.
            num_steps (int): The number of time steps in each simulated path.
            T (float): The time to maturity of the option.

        Returns:
            A dictionary containing the calculated 'lower_bound', 'upper_bound',
            and the resulting 'duality_gap'.
        """
        print(f"\nRunning engine for {num_paths:,} paths...")

        # 1. Simulate asset paths and Brownian increments from the SDE model.
        # The dual solver requires the `dW` increments.
        paths, dW = self.model.simulate_paths(num_paths, num_steps, T)
        paths = paths.to(self.device, dtype=torch.float32)
        dW = dW.to(self.device, dtype=torch.float32)

        # 2. Calculate the Lower Bound by invoking the provided primal solver.
        print("Executing Primal Solver (Lower Bound)...")
        lower_bound = self.primal_solver.solve(paths=paths, payoff_fn=self.payoff_fn, T=T)
        print(f"  -> Primal Price (Lower Bound): {lower_bound:.4f}")

        # 3. Calculate the Upper Bound by invoking the provided dual solver.
        print("Executing Dual Solver (Upper Bound)...")
        upper_bound = self.dual_solver.solve(paths=paths, dW=dW, payoff_fn=self.payoff_fn)
        print(f"  -> Dual Price (Upper Bound): {upper_bound:.4f}")

        # 4. Calculate the Duality Gap, the key measure of pricing accuracy.
        duality_gap = upper_bound - lower_bound
        print(f"  -> Duality Gap: {duality_gap:.4f}")

        return {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "duality_gap": duality_gap
        }