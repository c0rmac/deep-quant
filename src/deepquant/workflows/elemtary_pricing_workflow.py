import math
from enum import Enum
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from typing import Union, Optional
import pandas_market_calendars as mcal

# --- Imports for other deepquant modules ---
from ..models.sde import SDEFactory
from .primal_dual_engine import PricingEngine
from .price_deducer import PriceDeducer
from ..calibration.hurst_forecaster import HurstForecaster
from ..calibration.heston_calibrator import HestonCalibrator
from ..solvers.deep_signature_solver import DeepSignaturePrimalSolver, DeepSignatureDualSolver, \
    estimate_dual_solver_memory
from ..data.base_loader import AbstractDataLoader
from ..solvers.hilbert_operator_solver import HilbertOperatorPrimalSolver
from ..solvers.kernel_rff_solver import KernelRFFPrimalSolver
from ..solvers.linear_solver import LinearPrimalSolver

def _get_signature_dimension(truncation_level: int, path_dimension: int) -> int:
    """
    Helper function to calculate the dimension of a truncated signature for a
    path of a given dimension.
    """
    # For a 1D path, the signature dimension is simply the truncation level.
    if path_dimension == 1:
        return truncation_level

    # For d > 1, the dimension is the sum of a geometric series.
    # Formula: (d^(M+1) - 1) / (d - 1)
    dimension = (path_dimension ** (truncation_level + 1) - 1) // (path_dimension - 1)
    return dimension


def get_hidden_dim_heuristic(truncation_level: int, path_dimension: int = 2) -> int:
    """
    Calculates a heuristic for the hidden layer dimension based on the
    signature TRUNCATION LEVEL.

    It first computes the signature's dimension and then chooses the nearest
    power of two as the recommended hidden dimension.

    Args:
        truncation_level (int): The truncation level of the signature.
        path_dimension (int, optional): The dimension of the path being analyzed.
                                       Defaults to 2 (e.g., time + 1D asset price).

    Returns:
        int: The recommended hidden dimension for the neural network.
    """
    if truncation_level < 1:
        raise ValueError("Truncation level must be 1 or greater.")

    # 1. Calculate the signature dimension from the truncation level.
    signature_dimension = _get_signature_dimension(truncation_level, path_dimension)

    # 2. Apply the power-of-two heuristic to the calculated dimension.
    log_dim = math.log2(signature_dimension)
    rounded_log_dim = round(log_dim)
    hidden_dim = int(math.pow(2, rounded_log_dim))

    return hidden_dim

def get_num_layers_heuristic(truncation_level: int) -> int:
    """
    Calculates a heuristic for the number of ResNet blocks based on the
    signature truncation level.

    The heuristic is (Truncation Level - 1), with a minimum of 1 block.
    This scales the model's depth with the complexity of the signature.

    Args:
        truncation_level (int): The truncation level of the signature.

    Returns:
        int: The recommended number of ResNet blocks.
    """
    if truncation_level < 1:
        raise ValueError("Truncation level must be 1 or greater.")

    # The number of layers is the truncation level minus one, but at least 1.
    num_layers = (max(1, truncation_level - 1) / 2).__ceil__()

    return num_layers

class ElementaryPricingWorkflow:
    """
    Orchestrates the end-to-end pricing of an American option using the
    adaptive, hybrid framework, with an option to force a specific model.

    This workflow represents the highest level of abstraction in the library.
    It encapsulates the entire decision-making and pricing process:
    1.  It loads market data via an injected, exchangeable data loader.
    2.  It forecasts the market's volatility "roughness" by estimating the
        future Hurst parameter, H(t).
    3.  Based on the forecast, it performs a "regime switch," selecting the most
        appropriate SDE model (smooth Heston for H >= 0.5, rough Bergomi for H < 0.5).
    4.  It calibrates the chosen model to the historical data.
    5.  It runs the powerful primal-dual engine with the best deep learning solver.
    6.  It deduces a final, actionable price with a quantified uncertainty.
    """

    def __init__(
            self,
            data_loader: AbstractDataLoader,
            models_dir: Path,
            risk_free_rate: float,
            retrain_hurst_interval_days: int = 30,
            force_model: Optional[str] = None,
            bergomi_static_params: dict = { 'H': 0.49, "eta": 1.9, "rho": -0.7 },
            heston_static_params: dict = { 'H': 0.5, "rho": -0.7 },
    ):
        """
        Initializes the HybridPricingWorkflow.

        Args:
            data_loader (AbstractDataLoader): A concrete data loader object
                (e.g., YFinanceLoader) that provides the historical market data.
            models_dir (Path): Path to the directory where the models are stored.
            risk_free_rate (float): The risk-free interest rate to use for pricing.
            retrain_hurst_interval_days (int): The number of days to retrain the HurstForecaster.
                Set to 0 for forced retraining.
                Set to never retrain.
            force_model (Optional[str]): If set to 'heston' or 'bergomi', overrides
                the Hurst forecast and forces the use of the specified model for
                the lifetime of this workflow instance. Defaults to None (adaptive mode).
        """
        self.data_loader = data_loader
        self.models_dir = models_dir
        self.r = risk_free_rate
        self.retrain_hurst_interval_days = retrain_hurst_interval_days
        self.force_model = force_model
        self.bergomi_static_params = bergomi_static_params
        self.heston_static_params = heston_static_params

        dual_truncation_level = 3
        self.dual_truncation_level = dual_truncation_level

        # resnet_depth = min(get_num_layers_heuristic(dual_truncation_level), dual_max_learning_depth)
        resnet_depth = 2
        self.resnet_depth = resnet_depth

        # hidden_dim = get_hidden_dim_heuristic(dual_truncation_level)
        hidden_dim = 64
        self.hidden_dim = hidden_dim

        # For the final, effective library, we use our most powerful solver.
        # The hyperparameters are set to robust, well-tuned values.
        self.primal_solver = HilbertOperatorPrimalSolver(risk_free_rate=self.r)
        self.dual_solver = DeepSignatureDualSolver(
            hidden_dim=hidden_dim,
            learning_rate=0.009,
            max_epochs=800,
            patience=50,
            tolerance=1e-3,
            num_res_net_blocks=resnet_depth,
            batch_size=128
        )

    def price_option(
            self,
            strike: float,
            maturity: Union[int, float, str, date],
            option_type: str,
            primal_uncertainty: float,
            exchange: str = 'NYSE',
            evaluation_date: Union[str, date] = None,
            max_num_paths: int = 100_000,
            max_num_steps: int = 100_000
    ):
        """
        Executes the full, adaptive pricing workflow.

        Args:
            strike (float): The option's strike price.
            maturity (Union[int, float, str, date]): The option's time to maturity.
                Can be an integer/float (number of trading days) or a specific
                date (as a 'YYYY-MM-DD' string or a datetime.date object).
            option_type (str): The type of option ('put' or 'call').
            primal_uncertainty (float): Defines within what monetary range the primal's price must be.

                Since the primal must be computed on a stochastic process,
                there is uncertainty on each primal computation. The process
                will generate paths and run the primal until the mean is within
                a 95% confidence interval of width 2 * primal_uncertainty.

                For example, if the deduced option price is $2.05, and primal-uncertainty is $0.05,
                the process will stop once the deduced price's 95%-confidence interval has shrunk to ($2, $2.10).
            exchange (str): The stock exchange calendar to use for counting
                            trading days (e.g., 'NYSE'). Defaults to 'NYSE'.
            evaluation_date (Union[str, date], optional): The date for the valuation. Defaults to today.
            max_num_paths (int, optional): The maximum number of paths the simulation is allowed to generate.
                Reduce this in order to reduce resource usage.
                Note: Smaller values may mean that the primal process will have to run for longer in order to
                obtain a sufficiently small primal uncertainty on the confidence interval. It may also
                induce significant bias (ie: miss-pricing the deduced price). Use with caution
            max_num_steps (int, optional): The maximum number of steps the simulation is allowed to generate.
                Reduce this in order to reduce resource usage.
                Note: Smaller values may mean that the deduced primal will be significantly biased
                (ie: miss-pricing the deduced price). Use with caution.

        Returns:
            A tuple containing the deduced price dictionary and the full engine results.
        """

        # --- Step 1: Handle Dates ---
        # This section makes the API user-friendly by handling multiple date formats.
        # It determines the valuation date and calculates the option's time to maturity
        # in both trading days and annualized years.

        if evaluation_date:
            eval_date = pd.to_datetime(evaluation_date).date()
        else:
            eval_date = date.today()

        calendar = mcal.get_calendar(exchange)

        # **THE FIX**: This logic now correctly handles all specified maturity types.
        if isinstance(maturity, (int, float)):
            # If given a number, assume it's a number of trading days and find the future date.
            schedule = calendar.schedule(start_date=eval_date,
                                         end_date=eval_date + pd.Timedelta(days=int(maturity * 1.8)))
            maturity_date_obj = schedule.index[int(maturity) - 1].date()
        elif isinstance(maturity, str):
            # If it's a string, convert to a standard date object.
            maturity_date_obj = pd.to_datetime(maturity).date()
        elif isinstance(maturity, date):
            # If it's already a date object, use it directly.
            maturity_date_obj = maturity
        else:
            raise TypeError("maturity must be an int/float (days), a string 'YYYY-MM-DD', or a date object.")

        if maturity_date_obj <= eval_date:
            raise ValueError("Maturity date must be after the evaluation date.")

        # Use the market calendar to get the precise number of trading days.
        trading_schedule = calendar.schedule(start_date=eval_date, end_date=maturity_date_obj)
        maturity_in_days = len(trading_schedule)

        # Determine the number of trading days in the specific year
        # of the option's life for a more accurate annualization.
        year_schedule = calendar.schedule(start_date=f"{eval_date.year}-01-01", end_date=f"{eval_date.year}-12-31")
        trading_days_per_year = len(year_schedule)

        # Convert to an annualized year fraction.
        maturity_in_years = maturity_in_days / trading_days_per_year

        print(f"Evaluation Date: {eval_date.strftime('%Y-%m-%d')}")
        print(f"Maturity Date:   {maturity_date_obj.strftime('%Y-%m-%d')} ({maturity_in_days} trading days)")

        # --- Step 2: Data Loading and Setup ---
        # The workflow uses the injected data loader, making it independent of the data source.
        log_returns = self.data_loader.load()
        s0 = self.data_loader.get_spot_price()
        sde_factory = SDEFactory()
        calibrator = HestonCalibrator(log_returns=log_returns)
        calibrated_params = calibrator.calibrate()
        model_params = {'s0': s0, 'r': self.r, 'rho': self.heston_static_params['rho'], **calibrated_params}

        # --- Step 3: Model Selection (Regime Switch) ---
        # This is the core "brain" of the adaptive framework. It decides whether
        # the market is likely to be rough or smooth and selects the best model.

        use_bergomi = False # Default to Heston (smooth regime)

        if self.force_model:
            # If a model is forced, we bypass the forecast.
            print(f"-> Model Override: Forcing use of {self.force_model.title()} model.")
            if self.force_model.lower() == 'bergomi':
                use_bergomi = True
            elif self.force_model.lower() != 'heston':
                raise ValueError("force_model must be 'heston' or 'bergomi'")
        else:
            # If no model is forced, use the adaptive Hurst forecast logic.
            h_forecaster = HurstForecaster(log_returns=log_returns)
            h_forecast = h_forecaster.forecast(
                horizon=maturity_in_days,
                models_dir=self.models_dir,
                retrain_interval_days=self.retrain_hurst_interval_days,
                force_retrain=self.retrain_hurst_interval_days == 0
            )
            model_params['H'] = h_forecast
            if h_forecast < 0.5:
                use_bergomi = True

        # Set the final model parameters based on the decision
        if use_bergomi:
            if 'H' not in model_params or self.force_model == 'bergomi':
                # model_params['H'] = self.bergomi_static_params['H']
                model_params.update(self.bergomi_static_params)
            else:
                model_params['eta'] = self.bergomi_static_params['eta']
                model_params['rho'] = self.bergomi_static_params['rho']
            print(f"-> Regime: ROUGH market (H={model_params['H']:.3f}). Selecting Bergomi model.")
        else:
            if self.force_model == 'heston':
                # model_params['H'] = self.bergomi_static_params['H']
                model_params.update(self.heston_static_params)
            print(f"-> Regime: SMOOTH market (H={model_params['H']:.3f}). Selecting Heston model.")
            model_params['H'] = 0.5 # Ensure H is exactly 0.5 for Heston

        sde_model = sde_factory.create_model(**model_params)

        # --- Step 4: Run Pricing Engine ---
        # With the model and parameters set, we pass everything to the core
        # primal-dual engine to perform the heavy lifting of the simulation and pricing.

        engine = PricingEngine(
            sde_model=sde_model,
            primal_solver=self.primal_solver,
            dual_solver=self.dual_solver,
            option_type=option_type,
            strike=strike,
            r=self.r
        )

        # num_steps = int(1200 * maturity_in_years)
        engine_results = engine.run(
            T=maturity_in_years,
            primal_uncertainty=primal_uncertainty,
            max_num_paths=max_num_paths,
            max_num_steps=max_num_steps
        )

        # --- Step 5: Deduce the Final Price ---
        # We take the raw bounds from the engine and calculate a single, actionable
        # price point and its associated uncertainty.
        price_deducer = PriceDeducer()
        final_price_info = price_deducer.deduce(engine_results)

        return final_price_info, engine_results