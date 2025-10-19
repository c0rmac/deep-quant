# DeepQuant: An Adaptive Deep Learning Library for Quantitative Finance ⚖️

[![PyPI version](https://badge.fury.io/py/deep-quant-lib.svg)](https://badge.fury.io/py/deep-quant-lib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/example/deep-quant)

**DeepQuant** is a modern, high-performance Python library for pricing American options under advanced stochastic volatility models. It combines state-of-the-art techniques from deep learning and rough path theory to provide accurate, reliable price bounds for complex derivatives.

## In-Depth Documentation 📚

For a deeper dive into the methodologies and advanced usage of the library, please refer to the following documents:

* [**`Base Theory: SDEs and Path Signatures`**](./docs/BASE_THEORY.md)**:** The technical details motivating much of the library's foundations.

* [**`Neural Network Solvers`**](./docs/NN_THEORY.md)**:** The technical details driving the implementation of the neural network and kernel solvers.

* [**`Challenges & Enhancements`**](./docs/ENHANCEMENTS.md)**:** The technical details describing how challenges were uncovered and how enhancements were made to solve them.

* [**`Advanced Usage and Examples`**](https://www.google.com/search?q=ADVANCED_EXAMPLES.md)**:** Details other advanced usages of the library beyond what the `ElementaryPricingWorkflow` provides.

## Installation ⚙️

### For Users

**⚠️ Important: PyTorch Dependency**

This package depends on PyTorch. Due to the way PyTorch is built for different hardware (NVIDIA GPUs, Apple Silicon, etc.), you **must install PyTorch manually before installing `deepquant`**.

If you try to install `deepquant` directly, `pip` may install a version of PyTorch that is incompatible with your system or with `deepquant`'s other dependencies, which can cause low-level crashes (like a `SIGSEGV` error).

Please follow the guide for your preferred package manager.

---

#### Option 1: Mamba / Conda (Recommended)

This is the safest and most reliable method. It ensures all your packages (PyTorch, NumPy, etc.) are 100% binary-compatible.

1.  **Create a new, strict environment.**
    This command creates a new environment, sets `conda-forge` as the *only* source, and installs Python.
    ```bash
    mamba create -n deepquant-env -c conda-forge --strict-channel-priority python=3.10
    ```

2.  **Activate the environment.**
    ```bash
    mamba activate deepquant-env
    ```

3.  **Install PyTorch**
    Install all your complex binary packages from `conda-forge` first.
    ```bash
    mamba install pytorch -c pytorch
    ```

4.  **Install `deepquant` (forcing a source build).**
    This final step uses `pip` to install our package, but the `--no-binary :all:` flag forces it to compile against the PyTorch you just installed, guaranteeing compatibility.
    ```bash
    pip install deep-quant-lib
    ```

---

#### Option 2: `pip` and `venv`

This method works, but you must be careful to install the correct PyTorch build.

1.  **Create and activate a virtual environment.**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate   # On Windows
    ```

2.  **Install PyTorch FIRST.**
    Go to the **[Official PyTorch Website](https://pytorch.org/get-started/locally/)** and select the correct command for your system (macOS, Linux, Windows) and hardware (CPU, NVIDIA CUDA, Apple MPS).

    *For example, for a Mac with an M2 chip, you would run:*
    ```bash
    pip install torch
    ```
    *For a Linux machine with CUDA 12.1, you would run:*
    ```bash
    pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

3.  **Install `deepquant`.**
    Once PyTorch is successfully installed, you can safely install `deepquant`.
    ```bash
    pip install deep-quant-lib
    ```

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
    bergomi_static_params = { 'H': 0.4, "eta": 1.9, "rho": -0.9 } # Override the bergomi simulation parameters.
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
```

## License ©️

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
