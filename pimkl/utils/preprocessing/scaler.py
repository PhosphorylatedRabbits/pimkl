"""Data scaling utilities."""
import pandas as pd
import numpy as np
from .core import enforce_pandas_dataframe_on_second_argument


class Scaler(object):
    """Object for data scaling."""

    scales = None

    def __init__(self, min_target=.0, max_target=1., fill_value=0.0):
        """Instantiate a Scaler."""
        self.min_target = .0
        self.max_target = 1.
        self.fill_value = fill_value

    @enforce_pandas_dataframe_on_second_argument
    def _apply(self, data):
        data_std = (data - self.scales['min']) / (
            self.scales['max'] - self.scales['min']
        )
        return data_std * (self.max_target - self.min_target) + self.min_target

    @enforce_pandas_dataframe_on_second_argument
    def _apply_and_fillna(self, data):
        transformed = self._apply(data)
        to_be_filled = self.scales['max'] == self.scales['min']
        to_be_filled_slice = to_be_filled.index[to_be_filled.values]
        transformed[to_be_filled_slice] = (
            transformed[to_be_filled_slice].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(self.fill_value)
        )
        return transformed

    @enforce_pandas_dataframe_on_second_argument
    def apply(self, data):
        """Learn and apply the scaling."""
        minimums = data.min()
        maximums = data.max()
        self.scales = pd.DataFrame(
            {'min': minimums, 'max': maximums}
        )
        return self._apply_and_fillna(data)

    @enforce_pandas_dataframe_on_second_argument
    def reapply(self, data):
        """Re-apply the scaling."""
        return self._apply_and_fillna(data)

    @enforce_pandas_dataframe_on_second_argument
    def unapply(self, data):
        """Unapply the scaling."""
        if self.scales is not None:
            data_std = (data - self.min_target) / (
                self.max_target - self.min_target
            )
            return data_std * (
                self.scales['max'] - self.scales['min']
            ) + self.scales['min']
        else:
            return data
