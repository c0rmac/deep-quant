import torch
import matplotlib.pyplot as plt
import pandas as pd

from src.sde_simulator import BergomiModel
from src.primal_dual_engine import PrimalDualEngine
from src.solvers.linear_solver import LinearPrimalSolver, LinearDualSolver
from src.solvers.kernel_rff_solver import KernelRFFPrimalSolver, KernelRFFDualSolver
from src.solvers.deep_signature_solver import DeepSignaturePrimalSolver, DeepSignatureDualSolver


def run_comparison():
    """
    Executes the final methodology comparison for Phase 3 of the project.

    This script compares the performance of the Linear, Kernel RFF, and Deep Signature
    solver pairs on pricing an American option under the rough Bergomi model.
    It generates a bar chart comparing their final duality gaps, which serves as the
    key result of the project.
    """
    # --- 1. Configuration ---
    print("Starting Phase 3: Solver Methodology Comparison")

    # Option & Model Parameters for the rough Bergomi model
    K = 100.0
    T = 1.0
    r = 0.05
    s0 = 100.0
    v0 = 0.04
    H = 0.1
    eta = 1.9
    rho = -0.9

    # Simulation & Solver Parameters
    num_steps = 50
    num_paths = 5000
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    # device = 'cpu'
    print(f"Using device: {device}")

    # --- 2. Setup ---

    bergomi_model = BergomiModel(s0=s0, v0=v0, r=r, H=H, eta=eta, rho=rho)

    # Define the configurations for each solver pair
    solver_configs = {
        "Linear": {
            "primal": LinearPrimalSolver(truncation_level=4, risk_free_rate=r),
            "dual": LinearDualSolver(truncation_level=4, learning_rate=0.005, max_epochs=600, patience=15,
                                     tolerance=1e-5)
        },
        "Kernel RFF": {
            "primal": KernelRFFPrimalSolver(truncation_level=4, risk_free_rate=r, n_rff=1024*4, gamma='scale'),
            "dual": KernelRFFDualSolver(truncation_level=4, learning_rate=0.005, max_epochs=600, patience=15,
                                        tolerance=1e-5, n_rff=256, gamma='scale')
        },
        "Deep Signature": {
            "primal": DeepSignaturePrimalSolver(truncation_level=4, risk_free_rate=r, hidden_dim=1024, epochs=50,
                                                batch_size=256, lr=0.008),
            "dual": DeepSignatureDualSolver(truncation_level=4, hidden_dim=1024, learning_rate=0.001, max_epochs=600,
                                            patience=15, tolerance=1e-5)
        },
        "Optimal": {
            "primal": KernelRFFPrimalSolver(truncation_level=4, risk_free_rate=r, n_rff=1024 * 2, gamma='scale'),
            "dual": DeepSignatureDualSolver(truncation_level=4, hidden_dim=256, learning_rate=0.009, max_epochs=800,
                                            patience=15, tolerance=1e-7)
        }
    }

    results_log = {}

    solver_configs = {"Optimal": solver_configs["Optimal"]}

    # --- 3. Main Simulation & Comparison Loop ---

    print(f"\nGenerating {num_paths:,} paths from the Bergomi model...")
    paths, dW = bergomi_model.simulate_paths(num_paths, num_steps, T)

    for name, config in solver_configs.items():
        print(f"\n--- Testing Solver: {name} ---")

        engine = PrimalDualEngine(
            sde_model=bergomi_model,
            primal_solver=config["primal"],
            dual_solver=config["dual"],
            option_type='put',
            strike=K,
            device=device
        )

        # We manually call the solvers with the pre-generated paths for a fair comparison
        print("Executing Primal Solver (Lower Bound)...")
        lower_bound = engine.primal_solver.solve(paths=paths.clone(), payoff_fn=engine.payoff_fn, T=T)
        print(f"  -> Primal Price (Lower Bound): {lower_bound:.4f}")

        print("Executing Dual Solver (Upper Bound)...")
        upper_bound = engine.dual_solver.solve(paths=paths.clone(), dW=dW.clone(), payoff_fn=engine.payoff_fn)
        print(f"  -> Dual Price (Upper Bound): {upper_bound:.4f}")

        duality_gap = upper_bound - lower_bound
        print(f"  -> Duality Gap: {duality_gap:.4f}")

        results_log[name] = {
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Duality Gap": duality_gap
        }

    # --- 4. Results Summary and Visualization ---

    results_df = pd.DataFrame.from_dict(results_log, orient='index')
    print("\n--- Comparison Summary ---")
    print(results_df)

    print("\nGenerating Duality Gap comparison plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    gaps = results_df["Duality Gap"]
    colors = ['skyblue', 'salmon', 'lightgreen']
    bars = ax.bar(gaps.index, gaps.values, color=colors[:len(gaps)])

    ax.set_ylabel('Duality Gap ($)', fontsize=12)
    ax.set_title('Duality Gap Comparison Across Solvers (Bergomi Model)', fontsize=16)
    ax.set_xticks(gaps.index)
    ax.set_xticklabels(gaps.index, fontsize=12)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

    output_path = 'visualizations/phase3_duality_gap_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    run_comparison()