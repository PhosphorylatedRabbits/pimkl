"""Data standardization utilities."""
import pandas as pd
import numpy as np
from .core import enforce_pandas_dataframe_on_second_argument


class Standardizer(object):
    """Object for data standardization."""

    parameters = None

    @enforce_pandas_dataframe_on_second_argument
    def _apply(self, data):
        return (data - self.parameters['mean']) / self.parameters['std']

    @enforce_pandas_dataframe_on_second_argument
    def _apply_and_fillna(self, data):
        to_be_filled = self.parameters['std'] == 0.0
        transformed = self._apply(data)
        to_be_filled_slice = to_be_filled.index[to_be_filled.values]
        transformed[to_be_filled_slice] = (
            transformed[to_be_filled_slice].replace([np.inf, -np.inf],
                                                    np.nan).fillna(0.0)
        )
        return transformed

    @enforce_pandas_dataframe_on_second_argument
    def apply(self, data):
        """Learn and apply the standardization."""
        stds = data.std()
        means = data.mean()
        self.parameters = pd.DataFrame({'mean': means, 'std': stds})
        return self._apply_and_fillna(data)

    @enforce_pandas_dataframe_on_second_argument
    def reapply(self, data):
        """Re-apply the standardization."""
        return self._apply_and_fillna(data)

    @enforce_pandas_dataframe_on_second_argument
    def unapply(self, data):
        """Unapply the standardization."""
        if self.parameters is not None:
            return (data * self.parameters['std'] + self.parameters['mean'])
        else:
            return data
