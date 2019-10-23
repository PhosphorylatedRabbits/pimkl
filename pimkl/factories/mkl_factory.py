from pymimkl import EasyMKL, UMKLKNN, AverageMKL
import logging
import numpy as np
logger = logging.getLogger(__name__)


class WeightedAverageMKL(AverageMKL):
    """small wrapping of AverageMKL where the additional cunstructor parameter
    kernels_weights is used to predict a final kernel rather than the average.

    The applied weights are corrected to sum up to one.
    """

    def __init__(self, kernels_weights, *args, **kwargs):
        if np.any(kernels_weights < 0):
            raise ValueError(
                'kernels_weights contains negative values:\n{}'.
                format(kernels_weights)
            )
        self.given_kernels_weights = kernels_weights
        super().__init__(*args, **kwargs)

    def fit(self, X, y=None):
        # set kernels (and average weights), tracefactors and so on
        AverageMKL.fit(self, X, y)
        # resetting the weights, that should be a convex sum
        kernels_weights = np.array(
            self.given_kernels_weights, order='F', dtype=np.float64
        )
        kernels_weights = kernels_weights / kernels_weights.sum()
        self.weights = kernels_weights
        self.kernels_weights = kernels_weights


MKL_FACTORY = {
    'EasyMKL': EasyMKL,
    'UMKLKNN': UMKLKNN,
    'AverageMKL': AverageMKL,
    'WeightedAverageMKL': WeightedAverageMKL
}
