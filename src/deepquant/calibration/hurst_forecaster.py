import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from xgboost import XGBRegressor
from datetime import datetime, timedelta


class HurstForecaster:
    r"""
    Estimates and forecasts the time-varying Hurst parameter H(t) from a
    provided time series of log returns, with an intelligent, time-based
    retraining mechanism.

    This class provides the core logic for the adaptive, hybrid framework. The
    process involves two main stages:

    1.  **Historical Estimation:** A rolling window estimation of the Hurst
        parameter is performed on the historical log returns using the classic
        Rescaled Range (R/S) analysis. This produces a time series, H(t),
        representing the evolution of the market's "roughness."

    2.  **Forecasting:** A multi-step XGBoost model is trained to forecast the
        future path of H(t). The features for this model are the lagged (most
        recent) values of the historical H(t) series.

    The final output is the average of the forecasted H path, which is used by
    the `HybridPricingWorkflow` to select the appropriate SDE model (Heston for
    H >= 0.5, Bergomi for H < 0.5).
    """

    def __init__(self, log_returns: pd.Series, power: int = 6, k: int = 5):
        """
        Initializes the HurstForecaster.

        Args:
            log_returns (pd.Series): A time series of historical log returns for the asset.
            power (int): Log2 of the rolling window size for the R/S analysis
                         (e.g., power=6 corresponds to a 2^6 = 64-day window).
            k (int): The number of lagged Hurst values to use as features for the
                     XGBoost forecasting model.
        """
        self.log_returns = log_returns
        self.power = power
        self.k = k

        print("Computing historical Hurst series...")
        self.hurst_series = self._compute_rolling_hurst(self.log_returns)
        print("Historical Hurst series computed successfully.")

    def _compute_rolling_hurst(self, returns: pd.Series) -> pd.Series:
        """
        Calculates the Hurst exponent using R/S analysis over a rolling window.
        This method is a direct adaptation of the logic in the provided reference scripts.
        """
        returns_np = returns.values
        n = 2 ** self.power  # The size of the main rolling window
        if len(returns_np) < n:
            raise ValueError(f"Not enough data for Hurst calculation. Need at least {n} data points.")

        hursts = []
        # The exponents for the sub-windows in the R/S analysis
        exponents = np.arange(2, self.power + 1)

        # Slide the main window across the entire time series
        for t in range(n, len(returns_np) + 1):
            window = returns_np[t - n:t]
            rs_log = []
            # For each sub-window size...
            for exp in exponents:
                m = 2 ** exp
                s = n // m
                segments = window.reshape(s, m)

                # Calculate the rescaled range (R/S) for each segment
                dev = np.cumsum(segments - segments.mean(axis=1, keepdims=True), axis=1)
                R = dev.max(axis=1) - dev.min(axis=1)
                S = segments.std(axis=1)
                rs = np.where(S != 0, R / S, 0)

                # Take the average log2 of the R/S values
                rs_log.append(np.log2(rs.mean()))

            # The Hurst exponent is the slope of the line fitting log(R/S) vs log(window_size)
            hursts.append(np.polyfit(exponents, rs_log, 1)[0])

        return pd.Series(np.array(hursts), index=returns.index[n - 1:])

    def train_forecaster(self, horizon: int, models_dir: Path, xgb_params: dict = None, train_frac: float = 0.8):
        """
        Trains and saves a set of XGBoost models and a metadata file with the training date.

        Args:
            horizon (int): The number of future steps (days) to forecast.
            models_dir (Path): The path to save the model to.
            xgb_params (dict, optional): Hyperparameters for the XGBoost model.
            train_frac (float): The fraction of historical data to use for training.
        """
        if xgb_params is None:
            xgb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'objective': 'reg:squarederror'}

        split = int(len(self.hurst_series) * train_frac)
        train_values = self.hurst_series.iloc[:split].values

        models = []
        # We train one separate model for each step in the forecast horizon
        for step in range(horizon):
            X, y = [], []
            # Create the lagged feature set
            for i in range(self.k, len(train_values) - step):
                X.append(train_values[i - self.k:i])
                y.append(train_values[i + step])

            model = XGBRegressor(**xgb_params)
            model.fit(np.array(X), np.array(y))
            models.append(model)

        model_path = models_dir / f'hurst_forecaster_h{horizon}.pkl'
        meta_path = models_dir / f'hurst_forecaster_h{horizon}_meta.json'

        # Save the trained model
        joblib.dump(models, model_path)

        # Save metadata with the current timestamp
        metadata = {'train_date': datetime.now().isoformat()}
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)

        print(f"Trained and saved {horizon}-step Hurst forecaster and metadata.")
        return models

    def forecast(self, horizon: int, models_dir: Path, force_retrain: bool = False, retrain_interval_days: int = 30) -> float:
        """
        Forecasts the average Hurst parameter, automatically retraining the model if it's stale.

        By default, it will load a pre-trained model if available. If the model is
        older than `retrain_interval_days` or if `force_retrain` is True, it will
        always train a new model on the latest available data.

        Args:
            horizon (int): The number of future steps (days) to forecast.
            models_dir (Path): The path to save the model to.
            force_retrain (bool): If True, a new model is trained even if one exists.
            retrain_interval_days (int): The maximum age of a model in days before a retrain is forced.

        Returns:
            The average of the forecasted Hurst path, H_bar.
        """
        model_path = models_dir / f'hurst_forecaster_h{horizon}.pkl'
        meta_path = models_dir / f'hurst_forecaster_h{horizon}_meta.json'

        # --- Intelligent Retraining Logic ---
        should_retrain = force_retrain
        if not should_retrain and model_path.exists():
            try:
                # Check the age of the existing model
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                train_date = datetime.fromisoformat(metadata['train_date'])
                if datetime.now() - train_date > timedelta(days=retrain_interval_days) and retrain_interval_days != -1:
                    print(f"Saved model is older than {retrain_interval_days} days. Forcing retrain.")
                    should_retrain = True
            except (FileNotFoundError, json.JSONDecodeError):
                print("Model metadata not found or corrupt. Forcing retrain.")
                should_retrain = True

        if not model_path.exists():
            should_retrain = True

        if should_retrain:
            print("Training a new forecaster model...")
            models = self.train_forecaster(horizon, models_dir)
        else:
            print(f"Loading pre-trained Hurst forecaster from {model_path}")
            models = joblib.load(model_path)
        # --- End of Logic ---

        # Use the last k values of the historical Hurst series as the initial input
        last_window = self.hurst_series.values[-self.k:]

        # Each model predicts one step further into the future
        predictions = [model.predict(last_window.reshape(1, -1))[0] for model in models]

        avg_h_forecast = np.mean(predictions)
        print(f"  -> Forecasted {horizon}-day Hurst path: {[f'{p:.3f}' for p in predictions]}")
        print(f"  -> Average Forecasted H: {avg_h_forecast:.4f}")
        return avg_h_forecast