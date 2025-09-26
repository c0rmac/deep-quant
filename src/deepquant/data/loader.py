import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from typing import Union

# Import the common interface that all data loaders must follow.
from .base_loader import AbstractDataLoader


class YFinanceLoader(AbstractDataLoader):
    """
    A concrete data loader for fetching historical price data from Yahoo Finance.

    This class handles the connection to the yfinance API, data download,
    and caching to provide the necessary inputs for the pricing workflow.
    """

    def __init__(self, ticker: str, period: str = "10y", end_date: Union[str, date] = None):
        """
        Initializes the YFinanceLoader.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL', '^GSPC').
            period (str, optional): The historical data period to download (e.g., "5y", "10y").
                                    Defaults to "10y" to ensure enough data for Hurst calculation.
            end_date (Union[str, date], optional): The end date for the historical data query.
                                                   If None, data is fetched up to the current date.
                                                   This is used for backtesting. Defaults to None.
        """
        self.ticker = ticker
        self.period = period
        self.end_date = end_date
        self._data = None  # Internal cache to avoid redundant downloads

    def _fetch_data(self):
        """
        Private method to fetch and cache the historical data from Yahoo Finance
        if it has not already been loaded.
        """
        if self._data is None:
            print(f"Fetching historical data for {self.ticker} up to {self.end_date or 'today'}...")
            self._data = yf.download(
                self.ticker,
                period=self.period,
                interval="1d",
                end=self.end_date,
                progress=False  # Suppress download progress bar
            )
            if self._data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            print("Data loaded successfully.")

    def load(self) -> pd.Series:
        """
        Loads the data and computes the log returns.

        This method fulfills the `AbstractDataLoader` interface.

        Returns:
            A pandas Series of historical log returns.
        """
        self._fetch_data()
        log_returns = np.log(self._data['Close'] / self._data['Close'].shift(1))
        return log_returns.dropna()

    def get_spot_price(self) -> float:
        """
        Returns the most recent spot price from the loaded data.

        This method fulfills the `AbstractDataLoader` interface.

        Returns:
            The latest available adjusted closing price.
        """
        self._fetch_data()
        return self._data['Close'].to_numpy()[-1][0]


class CsvLoader(AbstractDataLoader):
    """
    An example of a concrete data loader for a local CSV file.

    This class demonstrates how a user could create their own custom loader
    to integrate their private data sources with the `deepquant` library.
    """

    def __init__(self, path: str, date_col: str, price_col: str):
        """
        Initializes the CsvLoader.

        Args:
            path (str): The file path to the CSV file.
            date_col (str): The name of the column containing the dates.
            price_col (str): The name of the column containing the asset prices.
        """
        self.path = path
        self.date_col = date_col
        self.price_col = price_col
        self._data = pd.read_csv(path, parse_dates=[self.date_col], index_col=self.date_col)

    def load(self) -> pd.Series:
        """
        Loads the data from the CSV and computes the log returns.

        Returns:
            A pandas Series of historical log returns.
        """
        log_returns = np.log(self._data[self.price_col] / self._data[self.price_col].shift(1))
        return log_returns.dropna()

    def get_spot_price(self) -> float:
        """
        Returns the most recent spot price from the CSV data.

        Returns:
            The latest available price.
        """
        return self._data[self.price_col][-1]