from pathlib import Path

import yfinance as yf
import pandas as pd

from deepquant.data.loader import YFinanceLoader
from deepquant.workflows.elemtary_pricing_workflow import ElementaryPricingWorkflow

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
    option_type='put',

    # Defines within what monetary range the primal's price must be.
    primal_uncertainty=0.05
    # Since the primal must be computed on a stochastic process,
    # there is uncertainty on each primal computation. The process
    # will generate paths and run the primal until the mean is within
    # a 95% confidence interval of width 2 * primal_uncertainty.
    #
    # For example, if the deduced option price is $2.05, and primal-uncertainty is $0.05,
    # the process will stop once the deduced price's 95%-confidence interval has shrunk to ($2, $2.10).
)

# --- 3. Display Results ---
results = {"Asset": asset_ticker, "Spot Price": latest_price, **price_info, **engine_results}
print("\n--- FINAL PRICING RESULT ---")
print(pd.Series(results).to_string())