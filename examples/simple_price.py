from pathlib import Path

import yfinance as yf
import pandas as pd

from src.deepquant.data.loader import YFinanceLoader
from src.deepquant.workflows.elemtary_pricing_workflow import ElementaryPricingWorkflow

# --- 1. Setup ---
asset_ticker = 'AAPL'
try:
    latest_price = yf.Ticker(asset_ticker).history(period='1d')['Close'][0]
except IndexError:
    print(f"Could not fetch price for {asset_ticker}. Exiting.")
    exit()

strike_price = round(latest_price) # At-the-money

# --- 2. Price the Option ---
# Instantiate the data loader and the main workflow.
data_loader = YFinanceLoader(ticker=asset_ticker)
workflow = ElementaryPricingWorkflow(
    data_loader=data_loader,
    models_dir=Path.cwd() / "models",
    risk_free_rate=0.05
)

# Run the pricing process using default simulation parameters.
price_info, engine_results = workflow.price_option(
    strike=strike_price,
    maturity=252, # 1 year in trading days
    option_type='put'
)

# --- 3. Display Results ---
results = {"Asset": asset_ticker, "Spot Price": latest_price, **price_info, **engine_results}
print("\n--- FINAL PRICING RESULT ---")
print(pd.Series(results).to_string())