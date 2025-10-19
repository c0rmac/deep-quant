import yfinance as yf
import pandas as pd
from pathlib import Path

from deepquant.data.loader import YFinanceLoader
from deepquant.workflows.elemtary_pricing_workflow import ElementaryPricingWorkflow

# --- 1. Setup ---
asset_ticker = 'SPY'
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
    retrain_hurst_interval_days=30,

    force_model='bergomi', # Override the forecast and force the rough model
    # bergomi_static_params = { 'H': 0.4, "eta": 1.9, "rho": -0.9 } # Override the bergomi simulation parameters.
)

# Run the pricing process with custom, high-fidelity simulation parameters.
price_info, engine_results = workflow.price_option(
    strike=strike_price,
    maturity=maturity_date,
    option_type='put',
    primal_uncertainty=0.8,

    exchange='NYSE',        # <-- Specify the exchange for which the asset is traded.
    evaluation_date=evaluation_date,

    max_num_paths=300,       # <-- Specify the number of volatility paths to compute.
    max_num_steps=5000,      # <-- Specify the number of steps each volatility path should take.
    # Reduce these paramters in order to reduce resource usage.

    # Note: Smaller values may mean that the primal process will have to run for longer in order to
    # obtain a sufficiently small primal uncertainty on the confidence interval. It may also
    # induce significant bias (ie: miss-pricing the deduced price). Use with caution
)

# --- 3. Display Results ---
results = {"Asset": asset_ticker, "Spot Price": spot_price, **price_info, **engine_results}
print("\n--- FINAL PRICING RESULT (Advanced Backtest) ---")
print(pd.Series(results).to_string())