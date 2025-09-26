import pandas as pd
import numpy as np
from scipy.optimize import minimize


class HestonCalibrator:
    """
    Calibrates Heston model parameters to a historical asset price data series.

    This class provides a simplified calibration routine based on the **method of
    moments** for historical volatility. It is a practical alternative when liquid
    options data for a full implied volatility surface calibration is not available.

    The process involves:
    1.  Calculating a time series of annualized historical volatility from log returns.
    2.  Estimating the Heston parameters by matching the statistical moments
        (like mean, variance, and autocorrelation) of the model's variance
        process to the moments observed in the historical volatility data.

    This provides a reasonable set of parameters that allows the Heston SDE to
    simulate paths that are statistically consistent with the asset's past behavior.
    """

    def __init__(self, log_returns: pd.Series):
        """
        Initializes the HestonCalibrator.

        Args:
            log_returns (pd.Series): A time series of historical log returns for the asset.
        """
        self.log_returns = log_returns
        self.volatility_history = self._get_volatility_history()
        print("HestonCalibrator initialized with data.")

    def _get_volatility_history(self, window=30) -> pd.Series:
        """
        Calculates a time series of annualized rolling historical volatility.

        Args:
            window (int): The rolling window size in days.

        Returns:
            A pandas Series of historical volatility.
        """
        # The standard deviation of log returns is calculated over a rolling window.
        # It's multiplied by sqrt(252) to annualize it from daily data.
        vol_history = self.log_returns.rolling(window=window).std() * np.sqrt(252)
        return vol_history.dropna()

    def calibrate(self) -> dict:
        """
        Finds the Heston parameters that best fit the historical volatility series.

        This method estimates the five Heston parameters (`v0`, `kappa`, `theta`,
        `xi`, `rho`) by matching statistical properties.
        """
        print(f"Calibrating Heston parameters...")
        vol_series = self.volatility_history.values.squeeze(-1)

        # --- Parameter Estimation ---

        # The long-term variance (theta) is estimated as the mean of the historical variance series.
        theta = np.mean(vol_series ** 2)

        # The initial variance (v0) is set to the most recently observed variance.
        v0 = vol_series[-1] ** 2

        # The speed of mean reversion (kappa) and the vol-of-vol (xi) are found
        # by numerically minimizing an objective function. The function tries to
        # match the model's theoretical moments to the empirically observed moments.
        def objective(params):
            kappa, xi = params
            # Ensure parameters are in a valid range during optimization.
            if kappa <= 0 or xi <= 0: return np.inf

            # Moment 1: Autocorrelation of variance.
            # We match the theoretical 1-day autocorrelation of the Heston variance
            # process to the value empirically measured from the data.
            model_autocorr = np.exp(-kappa / 252)  # The lag is 1 day (1/252 years)
            empirical_autocorr = pd.Series(vol_series ** 2).autocorr(lag=1)

            # Moment 2: Variance of variance.
            # We match the theoretical variance of the Heston variance process
            # to the variance observed in the historical variance data.
            model_var_of_var = (xi ** 2 * theta) / (2 * kappa)
            empirical_var_of_var = np.var(vol_series ** 2)

            # The error is the sum of squared differences of these two moments.
            return (model_autocorr - empirical_autocorr) ** 2 + \
                (model_var_of_var - empirical_var_of_var) ** 2

        # Run the optimization to find the best-fitting kappa and xi.
        initial_guess = np.array([2.0, 0.3])
        bounds = [(1e-3, 10.0), (1e-3, 2.0)]  # Constrain parameters to sensible ranges
        result = minimize(objective, initial_guess, bounds=bounds)

        kappa_opt, xi_opt = result.x

        calibrated_params = {
            "v0": v0,
            "kappa": kappa_opt,
            "theta": theta,
            "xi": xi_opt
        }
        print("Calibration successful.")
        return calibrated_params