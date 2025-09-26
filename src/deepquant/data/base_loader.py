from abc import ABC, abstractmethod
import pandas as pd


class AbstractDataLoader(ABC):
    """
    Abstract base class for all data loaders.

    This class defines the common interface that all concrete data loaders
    (e.g., for yfinance, CSV, a database) must implement. This ensures
    they can be used interchangeably by the pricing workflows.
    """

    @abstractmethod
    def load(self) -> pd.Series:
        """
        Loads data from a source and returns it as a pandas Series of log returns.
        """
        pass

    @abstractmethod
    def get_spot_price(self) -> float:
        """
        Returns the most recent spot price from the data source.
        """
        pass