# =============================================================================
# base_strategy.py - Abstract Base Class for All Strategies
# =============================================================================

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class that all strategies must inherit from.

    Every strategy must implement:
        - name()
        - calculate_indicators(df)
        - get_signal(df, index)
        - get_params()
        - get_min_candles()
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        pass

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicator columns to the DataFrame.

        Args:
            df: OHLCV DataFrame with columns:
                timestamp, open, high, low, close, volume

        Returns:
            DataFrame with additional indicator columns added
        """
        pass

    @abstractmethod
    def get_signal(self, df: pd.DataFrame, index: int):
        """
        Get trading signal at a specific candle index.

        Args:
            df: DataFrame with indicators already calculated
            index: Current candle index to evaluate

        Returns:
            "BUY"  - enter long
            "SELL" - enter short
            None   - no signal
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Return strategy parameters as a dictionary.
        Used for display in backtest summary and chart titles.

        Returns:
            dict of parameter names and values
        """
        pass

    @abstractmethod
    def get_min_candles(self) -> int:
        """
        Return minimum number of candles required before
        signals can be generated reliably.

        Returns:
            int: minimum candle count
        """
        pass

    def __str__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.name}({params})"
