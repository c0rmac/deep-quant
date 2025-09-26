from pathlib import Path

import yfinance as yf
import pandas as pd
from datetime import date
import numpy as np
from typing import Union

# The user of your library only needs to import the high-level components.
from src.deepquant.data.loader import YFinanceLoader
from src.deepquant.workflows.elemtary_pricing_workflow import ElementaryPricingWorkflow


def price_american_option(
        ticker: str,
        strike: float,
        maturity: Union[int, float, str, date],
        evaluation_date: Union[str, date] = None,
        option_type: str = 'put',
        risk_free_rate: float = 0.05,
        num_paths: int = 25_000,
        num_steps: int = 70
) -> dict:
    """
    High-level API to price an American option on a real-world asset
    as of a specific evaluation date.

    This function encapsulates the entire hybrid workflow:
    1.  Fetches historical data for the specified ticker up to the evaluation date.
    2.  Forecasts the market's future roughness (Hurst parameter).
    3.  Selects the appropriate SDE model (Heston or Bergomi) based on the forecast.
    4.  Calibrates the chosen model to the historical data.
    5.  Runs the advanced primal-dual engine with the Deep Signature solver.
    6.  Returns a final price, uncertainty estimate, and detailed bounds.
    """

    # Instantiate the data loader, configured with the evaluation date.
    # This ensures no future data is used in calibration or forecasting.
    data_loader = YFinanceLoader(ticker=ticker, end_date=evaluation_date)

    # Instantiate the main pricing workflow, injecting the data loader.
    workflow = ElementaryPricingWorkflow(
        data_loader=data_loader,
        models_dir=Path.cwd() / "models",
        risk_free_rate=risk_free_rate,
        retrain_hurst_interval_days=30,
        primal_learning_scale=24,
        dual_learning_depth=1,
        # force_model='bergomi'
    )

    # Run the end-to-end pricing workflow.
    price_info, engine_results = workflow.price_option(
        strike=strike,
        maturity=maturity,
        option_type=option_type,
        num_paths=num_paths,
        num_steps=num_steps,
        evaluation_date=evaluation_date
    )

    # Combine and return a comprehensive set of results.
    s0 = data_loader.get_spot_price()

    final_results = {
        "Asset": ticker,
        "Spot Price": s0,
        "Strike": strike,
        # 'maturity' is handled inside the workflow, we can add it back to the report
        "Maturity": maturity,
        "Evaluation Date": evaluation_date or date.today().strftime('%Y-%m-%d'),
        "Option Type": option_type,
        **price_info,
        **engine_results
    }

    return final_results


if __name__ == "__main__":
    # This is an example of how a user would call the main function.

    # Price a 1-year, at-the-money put option on the S&P 500 Index.
    asset_ticker = '^GSPC'  # The Yahoo Finance ticker for the S&P 500

    # Get the latest price to set a reasonable at-the-money strike
    latest_price = yf.Ticker(asset_ticker).history(period='1d')['Close'][0]
    # Round the strike to a clean number, e.g., nearest 50 points
    strike_price = round(latest_price / 50) * 50

    # Run the full pricing workflow for a 1-year (252 trading days) option
    results = price_american_option(
        ticker=asset_ticker,
        strike=strike_price,
        maturity=252  # Specify maturity in trading days
    )

    # Print a clean summary of the final results
    print("\n--- FINAL PRICING RESULT ---")
    results_series = pd.Series(results)
    print(results_series.to_string())
