import yfinance as yf
import pandas as pd
from pathlib import Path

from src.deepquant.data.loader import YFinanceLoader
from src.deepquant.workflows.elemtary_pricing_workflow import ElementaryPricingWorkflow

# --- 1. Setup ---
asset_ticker = '^GSPC'
evaluation_date = '2023-01-03'
maturity_date = '2024-01-03'

try:
    spot_price = yf.Ticker(asset_ticker).history(start=evaluation_date, end='2023-01-04')['Close'][0]
except IndexError:
    print(f"Could not fetch price for {asset_ticker} on {evaluation_date}. Exiting.")
    exit()

strike_price = round(spot_price / 50) * 50

# --- 2. Price the Option with Advanced Configuration ---
data_loader = YFinanceLoader(ticker=asset_ticker, end_date=evaluation_date)

workflow = ElementaryPricingWorkflow(
    data_loader=data_loader,
    models_dir=Path.cwd() / "models",
    risk_free_rate=0.05,

    primal_learning_scale=8, # Scale up the learning ability of the primal.
    # Higher scale results in a tighter bound, but require more computational resources and
    # may be prone to overfitting if the num_paths and num_steps are not sufficiently large enough.
    dual_learning_depth=2, # Deepen the learning ability of the dual.
    # Greater depth results in a tighter bound, but requires more random access memory and
    # computational resources. Setting the depth too high without sufficient paths or steps
    # may result in wasted resources.

    force_model='bergomi', # Override the forecast and force the rough model
    bergomi_static_params = { 'H': 0.1, "eta": 1.9, "rho": -0.9 } # Override the bergomi
    # simulation parameters.
)

# Run the pricing process with custom, high-fidelity simulation parameters.
price_info, engine_results = workflow.price_option(
    strike=strike_price,
    maturity=maturity_date,
    option_type='put',

    exchange='NYSE',        # <-- Specify the exchange for which the asset is traded.

    num_paths=25_000,       # <-- Specify the number of volatility paths to compute.
    num_steps=70,           # <-- Specify the number of steps each volatility path should take.
    # Warning: Scaling num_steps beyond 100 and num_paths beyond 30_000 is
    # random access memory-resource intensive even if dual_learning_depth=1

    evaluation_date=evaluation_date
)

# --- 3. Display Results ---
results = {"Asset": asset_ticker, "Spot Price": spot_price, **price_info, **engine_results}
print("\n--- FINAL PRICING RESULT (Advanced Backtest) ---")
print(pd.Series(results).to_string())