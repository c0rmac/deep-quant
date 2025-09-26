import torch
import matplotlib.pyplot as plt
import pandas as pd
import QuantLib as ql
import time

from src.deepquant.models.sde import HestonModel
from src.deepquant.solvers.deep_signature_solver import DeepSignaturePrimalSolver, DeepSignatureDualSolver
from src.deepquant.solvers.kernel_rff_solver import KernelRFFPrimalSolver, KernelRFFDualSolver
from src.deepquant.solvers.linear_solver import LinearPrimalSolver, LinearDualSolver
from src.deepquant.workflows.price_deducer import PriceDeducer
from src.deepquant.workflows.primal_dual_engine import PricingEngine


def format_price(price):
    return f"{price:.4f}"


def run_heston_comparison():
    """
    Executes a comprehensive comparison of all solvers against the industry-standard
    QuantLib library on the Heston model.
    """
    # --- 1. Configuration ---
    print("Starting Final Validation: All Solvers vs. QuantLib (Heston Model)")

    # Option & Heston Model Parameters
    K = 100.0
    T = 1.0
    r = 0.05
    s0 = 100.0
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7
    dividend_yield = 0.0

    # Simulation & Solver Parameters
    num_steps = 50
    num_paths = 20_000
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Setup Solvers ---
    heston_model = HestonModel(s0=s0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, r=r)
    price_deducer = PriceDeducer()

    solver_configs = {
        "Linear": {
            "primal": LinearPrimalSolver(truncation_level=4, risk_free_rate=r),
            "dual": LinearDualSolver(truncation_level=4, learning_rate=0.005, max_epochs=500, patience=15,
                                     tolerance=1e-5)
        },
        "Kernel RFF": {
            "primal": KernelRFFPrimalSolver(truncation_level=4, risk_free_rate=r, n_rff=256, gamma='scale'),
            "dual": KernelRFFDualSolver(truncation_level=4, learning_rate=0.005, max_epochs=500, patience=15,
                                        tolerance=1e-5, n_rff=256, gamma='scale')
        },
        "Deep Signature": {
            "primal": DeepSignaturePrimalSolver(truncation_level=5, risk_free_rate=r, hidden_dim=64, epochs=10,
                                                batch_size=512, lr=0.001),
            "dual": DeepSignatureDualSolver(truncation_level=5, hidden_dim=64, learning_rate=0.001, max_epochs=200,
                                            patience=20, tolerance=1e-6)
        },
        "Deep Signature 2": {
            "primal": KernelRFFPrimalSolver(truncation_level=4, risk_free_rate=r, n_rff=1024 * 2, gamma='scale'),
            "dual": DeepSignatureDualSolver(truncation_level=5, hidden_dim=64, learning_rate=0.001, max_epochs=200,
                                            patience=20, tolerance=1e-6)
        }
    }

    solver_configs = {"Deep Signature 2": solver_configs["Deep Signature 2"]}

    results_log = {}

    # --- 3. Price with Your Engine ---
    print(f"\n--- Pricing with Custom Engine ---")
    print(f"Generating {num_paths:,} paths from the Heston model...")
    paths, dW = heston_model.simulate_paths(num_paths, num_steps, T)

    for name, config in solver_configs.items():
        print(f"\n--- Testing Solver: {name} ---")
        engine = PricingEngine(
            sde_model=heston_model,
            primal_solver=config["primal"],
            dual_solver=config["dual"],
            option_type='put',
            strike=K,
            device=device
        )
        engine_results = engine.run(num_paths=num_paths, num_steps=num_steps, T=T)
        price_info = price_deducer.deduce(engine_results)
        results_log[name] = {
            "Lower Bound": engine_results["lower_bound"],
            "Upper Bound": engine_results["upper_bound"],
            "Deduced Price": price_info["deduced_price"]
        }

    # --- 4. Price with QuantLib ---
    print("\n--- Pricing with QuantLib (Finite Difference Method) ---")
    start_time = time.time()
    calculation_date = ql.Date(24, 9, 2025)  # Using today's date
    ql.Settings.instance().evaluationDate = calculation_date
    maturity_date = calculation_date + ql.Period(1, ql.Years)
    exercise = ql.AmericanExercise(calculation_date, maturity_date)
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    option = ql.VanillaOption(payoff, exercise)
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(s0))
    risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, r, ql.Actual365Fixed()))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_yield, ql.Actual365Fixed()))
    heston_process = ql.HestonProcess(risk_free_ts, dividend_ts, spot_handle, v0, kappa, theta, xi, rho)
    ql_engine = ql.FdHestonVanillaEngine(ql.HestonModel(heston_process), tGrid=200, xGrid=200, vGrid=100)
    option.setPricingEngine(ql_engine)
    quantlib_price = option.NPV()
    end_time = time.time()
    print(f"QuantLib Calculation Time: {end_time - start_time:.2f} seconds")

    # --- 5. Final Comparison Summary ---
    results_df = pd.DataFrame.from_dict(results_log, orient='index')
    results_df['QuantLib Price'] = quantlib_price
    results_df['Error vs QL'] = results_df['Deduced Price'] - quantlib_price
    results_df['Is Bracketed'] = results_df.apply(
        lambda row: '✅ Yes' if row['Lower Bound'] <= quantlib_price <= row['Upper Bound'] else '❌ No', axis=1)

    print("\n--- Final Comparison Summary ---")
    print(results_df)

    print("\n--- Conclusion ---")
    if all(results_df['Is Bracketed'] == '✅ Yes'):
        print("✅ Success! All solvers successfully bracketed the industry-standard QuantLib price.")
    else:
        print("⚠️  Action Required: One or more solvers failed to bracket the QuantLib price.")


if __name__ == "__main__":
    run_heston_comparison()