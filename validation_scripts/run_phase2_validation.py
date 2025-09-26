import torch
import matplotlib.pyplot as plt

from src.sde_simulator import HestonModel
from src.primal_dual_engine import PrimalDualEngine
from src.solvers.linear_solver import LinearPrimalSolver, LinearDualSolver


def run_validation():
    """
    Executes the validation suite for Phase 2 of the project.

    This script analyzes the duality gap by running both the primal and dual
    solvers using the refactored, modular PrimalDualEngine. It plots the
    convergence of the lower bound (primal) and upper bound (dual) prices
    against a known benchmark price for an American option under the Heston model.

    The primary output is the duality gap convergence plot, which establishes the
    baseline performance of the linear signature models.
    """
    # --- 1. Configuration ---
    print("Starting Phase 2 Validation: Primal-Dual Duality Gap Analysis")

    # Option & Model Parameters
    K = 100.0
    T = 1.0
    r = 0.05
    s0 = 100.0
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7

    # Simulation & Solver Parameters
    num_steps = 50
    truncation_level = 4
    # device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = 'cpu'

    # Analysis parameters
    path_counts = [10_000, 20_000, 40_000]
    benchmark_price = 7.145

    print(f"Using device: {device}")
    print(f"Signature Truncation Level: {truncation_level}")

    # --- 2. Setup ---

    # Instantiate the Heston Model to generate financial data
    heston_model = HestonModel(s0=s0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, r=r)

    # Instantiate the specific solver objects we want to test
    primal_solver = LinearPrimalSolver(
        truncation_level=truncation_level,
        risk_free_rate=r
    )
    dual_solver = LinearDualSolver(
        truncation_level=truncation_level,
        learning_rate=0.005,
        max_epochs=500,  # Increased max epochs to ensure convergence
        patience=15,  # Increased patience slightly for stability
        tolerance=1e-5
    )

    # Pass the modular solver objects to the engine
    engine = PrimalDualEngine(
        sde_model=heston_model,
        primal_solver=primal_solver,
        dual_solver=dual_solver,
        option_type='put',
        strike=K,
        device=device
    )

    # --- 3. Main Simulation Loop ---

    lower_bounds = []
    upper_bounds = []
    gaps = []

    for num_paths in path_counts:
        results = engine.run(num_paths=num_paths, num_steps=num_steps, T=T)
        lower_bounds.append(results["lower_bound"])
        upper_bounds.append(results["upper_bound"])
        gaps.append(results["duality_gap"])

    # --- 4. Visualization ---
    print("\nGenerating Duality Gap plot...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot calculated prices
    ax.plot(path_counts, lower_bounds, marker='o', linestyle='-', color='b', label='Primal Price (Lower Bound)')
    ax.plot(path_counts, upper_bounds, marker='^', linestyle='-', color='g', label='Dual Price (Upper Bound)')

    # Plot benchmark price
    ax.axhline(y=benchmark_price, color='r', linestyle='--', label=f'Benchmark Price ({benchmark_price:.4f})')

    # Shade the duality gap, the key result of this phase
    ax.fill_between(path_counts, lower_bounds, upper_bounds, color='gray', alpha=0.2, label='Duality Gap')

    # Formatting the plot
    ax.set_title('Primal-Dual Price Convergence and Duality Gap (Heston Model)', fontsize=16)
    ax.set_xlabel('Number of Simulation Paths', fontsize=12)
    ax.set_ylabel('Calculated Option Price ($)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Save the figure to the specified directory
    output_path = 'visualizations/phase2_duality_gap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Display the plot
    plt.show()


if __name__ == "__main__":
    run_validation()