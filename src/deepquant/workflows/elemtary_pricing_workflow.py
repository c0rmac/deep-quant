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
from ..solvers.deep_signature_solver import DeepSignaturePrimalSolver, DeepSignatureDualSolver
from ..data.base_loader import AbstractDataLoader
from ..solvers.kernel_rff_solver import KernelRFFPrimalSolver


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
            primal_learning_scale: int=1,
            dual_learning_depth: int=1,
            retrain_hurst_interval_days: int = 30,
            force_model: Optional[str] = None,
            bergomi_static_params: dict = { 'H': 0.1, "eta": 1.9, "rho": -0.9 },
            heston_static_params: dict = { "rho": -0.7 },
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

        # For the final, effective library, we use our most powerful solver.
        # The hyperparameters are set to robust, well-tuned values.
        self.primal_solver = KernelRFFPrimalSolver(truncation_level=6, risk_free_rate=self.r, n_rff=64 * primal_learning_scale, gamma='scale')
        # self.primal_solver = LinearPrimalSolver(truncation_level=5, risk_free_rate=self.r)
        self.dual_solver = DeepSignatureDualSolver(
            truncation_level=6,
            hidden_dim=64,
            learning_rate=0.009,
            max_epochs=2000,
            patience=20,
            tolerance=1e-6,
            num_res_net_blocks=dual_learning_depth
        )

    def price_option(
            self,
            strike: float,
            maturity: Union[int, float, str, date],
            option_type: str,
            exchange: str = 'NYSE',
            num_paths: int = 20_000,
            num_steps: int = 50,
            evaluation_date: Union[str, date] = None,
    ):
        """
        Executes the full, adaptive pricing workflow.

        Args:
            strike (float): The option's strike price.
            maturity (Union[int, float, str, date]): The option's time to maturity.
                Can be an integer/float (number of trading days) or a specific
                date (as a 'YYYY-MM-DD' string or a datetime.date object).
            option_type (str): The type of option ('put' or 'call').
            exchange (str): The stock exchange calendar to use for counting
                            trading days (e.g., 'NYSE'). Defaults to 'NYSE'.
            num_paths (int): The number of Monte Carlo paths to simulate.
            num_steps (int): The number of time steps for the simulation.
            evaluation_date (Union[str, date], optional): The date for the valuation. Defaults to today.

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
            model_params['H'] = self.bergomi_static_params['H']
            print(f"-> Regime: ROUGH market (H={model_params['H']:.3f}). Selecting Bergomi model.")
            model_params.update(self.bergomi_static_params)
        else:
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
            strike=strike
        )
        engine_results = engine.run(num_paths=num_paths, num_steps=num_steps, T=maturity_in_years)

        # --- Step 5: Deduce the Final Price ---
        # We take the raw bounds from the engine and calculate a single, actionable
        # price point and its associated uncertainty.
        price_deducer = PriceDeducer()
        final_price_info = price_deducer.deduce(engine_results)

        return final_price_info, engine_results