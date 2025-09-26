# DeepQuant: An Adaptive Deep Learning Library for Quantitative Finance ⚖️

[![PyPI version](https://badge.fury.io/py/deep-quant-lib.svg)](https://badge.fury.io/py/deep-quant-lib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/example/deep-quant)

**DeepQuant** is a modern, high-performance Python library for pricing American options under advanced stochastic volatility models. It combines state-of-the-art techniques from deep learning and rough path theory to provide accurate, reliable price bounds for complex derivatives.

## In-Depth Documentation 📚

For a deeper dive into the methodologies and advanced usage of the library, please refer to the following documents:

* [**`Base Theory: SDEs and Path Signatures`**](./docs/BASE_THEORY.md)**:** The technical details motivating much of the library's foundations.

* [**`Neural Network Solvers`**](https://www.google.com/search?q=NN_THEORY.md)**:** The technical details driving the implementation of the neural network and kernel solvers.

* [**`Advanced Usage and Examples`**](https://www.google.com/search?q=ADVANCED_EXAMPLES.md)**:** Details other advanced usages of the library beyond what the `ElementaryPricingWorkflow` provides.

## Installation ⚙️

### For Users

DeepQuant is available on PyPI and can be installed with pip:

```
pip install deep-quant-lib

```

### For Developers

If you wish to contribute to the project, you can clone the repository and install it in editable mode:

```
git clone [https://github.com/your_username/deep-quant.git](https://github.com/your_username/deep-quant.git)
cd deep-quant
pip install -e .

```

## Usage 🚀

Here are two examples demonstrating how to use the `deepquant` library.

### Quick Start: Simple Example

This example shows the simplest way to price a 1-year, at-the-money put option on Apple (`AAPL`). The workflow uses sensible, heuristically chosen defaults for the simulation parameters (`num_paths` and `num_steps`).

```python
# examples/simple_price.py
import yfinance as yf
import pandas as pd
from pathlib import Path
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
    option_type='put'
)

# --- 3. Display Results ---
results = {"Asset": asset_ticker, "Spot Price": latest_price, **price_info, **engine_results}
print("\n--- FINAL PRICING RESULT ---")
print(pd.Series(results).to_string())
```

### Advanced Usage: Backtesting and Full Configuration

This example showcases the full power of the library for a research use case. We will price a 1-year put option on the S&P 500 (`^GSPC`) as if we were on **January 3rd, 2023**. We will **force the use of the rough Bergomi model** and override the default simulation parameters for a **high-fidelity** run.

```python
# examples/advanced_backtest.py
import yfinance as yf
import pandas as pd
from pathlib import Path
from deepquant.data.loader import YFinanceLoader
from deepquant.workflows.elemtary_pricing_workflow import ElementaryPricingWorkflow

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
```

## License ©️

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
