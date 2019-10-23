"""Pathway Induced Multiple Kernel Learning."""
import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from ..factories import MKL_FACTORY, ESTIMATOR_FACTORY, INDUCTION_FACTORY
from ..utils.objects import is_sequence, is_sequence_of_sequence

logger = logging.getLogger(__name__)


def _update_kernels(
    kernels, lhs, rhs, induction, inducer, induction_parameters
):
    kernel = induction(lhs, rhs, inducer, **induction_parameters)
    kernels.append(kernel)


def _update_kernels_from_inducers(
    inducers, kernels, lhs, rhs, induction, induction_parameters
):
    # optimize computation of the kernels
    lhs = np.array(lhs, order='F', dtype=np.float64)
    rhs = np.array(rhs, order='F', dtype=np.float64)
    for inducer in inducers:
        _update_kernels(
            kernels, lhs, rhs, induction, inducer, induction_parameters
        )


def _update_kernels_multiple_data(
    inducers, kernels, lhs, rhs, induction, induction_parameters
):
    dict_mode = (
        isinstance(lhs, dict) and isinstance(rhs, dict)
        and isinstance(inducers, dict)
    )
    if dict_mode:
        for key in lhs:
            a_lhs = lhs[key]
            a_rhs = rhs[key]
            corresponding_inducers = inducers[key]
            _update_kernels_from_inducers(
                corresponding_inducers, kernels, a_lhs, a_rhs, induction,
                induction_parameters
            )
    else:
        for a_lhs, a_rhs in zip(lhs, rhs):
            _update_kernels_from_inducers(
                inducers, kernels, a_lhs, a_rhs, induction,
                induction_parameters
            )


class PIMKL(BaseEstimator, ClassifierMixin):
    """Pathway Induced Multiple Kernel Learning
    with choice of MKL and estimator algorithm.
    Estimator is only trained when MKL is not an estimator itself."""

    def __init__(
        self,
        inducers,
        induction='induce_linear_kernel',
        mkl='UMKLKNN',
        estimator='EasyMKL',
        induction_parameters={},
        mkl_parameters={
            'k': 5,
            'epsilon': 0.0001,
            'maxiter_qp': 100000,
            'kernel_normalization': True,
            'precompute': True
        },
        estimator_parameters={
            'lam': 0.2,
            'epsilon': 1e-5,
            'regularization_factor': False,
            'kernel_normalization': False,
            'precompute': True
        }
    ):
        """Instantiate a PIMKL object."""
        self.inducers = inducers
        self.induction = induction
        self.mkl = mkl
        self.estimator = estimator
        self.induction_parameters = induction_parameters
        self.mkl_parameters = mkl_parameters
        self.estimator_parameters = estimator_parameters

    def get_params(self, deep=True):
        """Get model parameters."""
        return {
            'inducers': self.inducers,
            'induction': self.induction,
            'mkl': self.mkl,
            'estimator': self.estimator,
            'induction_parameters': self.induction_parameters,
            'mkl_parameters': self.mkl_parameters,
            'estimator_parameters': self.estimator_parameters
        }

    def set_params(self, **parameters):
        """Set model parameters."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def set_mkl_params(self, **parameters):
        """Set model parameters."""
        for parameter, value in parameters.items():
            self.mkl_parameters[parameter] = value

    def set_estimator_params(self, **parameters):
        """Set model parameters."""
        for parameter, value in parameters.items():
            self.estimator_parameters[parameter] = value

    def fit(self, X, y=None):
        """Fit the model.
        Estimator is only trained when MKL is not an estimator."""
        logger.debug('PIMKL.fit() start')
        self.mkl_model_ = MKL_FACTORY[self.mkl](**self.mkl_parameters)
        self.lhs_ = X
        self.y_ = y
        # prepare the kernels
        kernels = self._get_kernels(self.lhs_)

        # fit mkl
        self.mkl_model_.fit(kernels, self.y_)
        self.kernels_weights = self.mkl_model_.kernels_weights
        # in case of single binary problem, ensure kernels_weights is 1D
        try:
            binary_problems = self.kernels_weights.shape[1]
            if binary_problems == 1:
                self.kernels_weights = self.kernels_weights[:, 0]
        except IndexError:
            pass

        # when mkl is not an estimator, fit estimator
        if hasattr(self.mkl_model_, 'predict_proba'):
            self.estimator_model_ = None
            logger.debug('PIMKL.fit() done, is estimator and fitted already')
            return self

        if self.y_ is not None:
            logger.debug('train given estimator')
            self.estimator_parameters['trace_normalization'] = False
            self.estimator_parameters['precompute'] = True
            self.estimator_model_ = ESTIMATOR_FACTORY[self.estimator](
                **self.estimator_parameters
            )
            self.estimator_model_.fit(
                [self.mkl_model_.get_optimal_kernel()], self.y_
            )
            logger.debug('given estimator done')
        else:
            self.estimator_model_ = None
        logger.debug('PIMKL.fit() done')
        return self

    def predict(self, X):
        """
        Predict using trained model.

        It returns the optimal kernel using learned weights or,
        in case labels were fitted in training, the predicted labels.
        """
        # prepare the kernels
        kernels = self._get_kernels(self.lhs_, X)
        # predict
        try:
            return np.argmax(self.mkl_model_.predict_proba(kernels), axis=1)
        except AttributeError:
            if self.estimator_model_ is not None:
                return self.estimator_model_.predict(
                    [self.mkl_model_.predict(kernels)]
                )
            else:
                return self.mkl_model_.predict(kernels)

    def predict_proba(self, X):
        """Predict probabilities using trained model."""
        # predict
        if hasattr(self.mkl_model_, 'predict_proba'):
            kernels = self._get_kernels(self.lhs_, X)
            return self.mkl_model_.predict_proba(kernels)

        if self.estimator_model_ is not None:
            # prepare the kernels
            kernels = self._get_kernels(self.lhs_, X)
            # predict probabilities
            return self.estimator_model_.predict_proba(
                [self.mkl_model_.predict(kernels)]
            )
        else:
            raise RuntimeError(
                'predict_proba valid only if trained passing labels'
            )

    def _get_kernels(self, lhs, rhs=None):
        logger.debug('_get_kernels() start')
        if rhs is None:
            rhs = lhs

        kernels = []

        multiple_inductions = is_sequence(self.induction)
        multiple_induction_parameters = is_sequence_of_sequence(
            self.induction_parameters
        )
        multiple_data = is_sequence(lhs) and is_sequence(rhs)
        if multiple_data:
            if len(lhs) != len(rhs):
                raise ValueError(
                    'Mismatch in lenght of lhs:{} and rhs:{}'.format(
                        len(lhs), len(rhs)
                    )
                )

        if multiple_induction_parameters:
            # TODO given a set of different induction_parameters to be used:
            raise NotImplementedError

        # prepare the combined kernels
        if multiple_inductions and multiple_data:
            for induction in self.induction:
                _update_kernels_multiple_data(
                    self.inducers, kernels, lhs, rhs,
                    INDUCTION_FACTORY[induction], self.induction_parameters
                )

        elif multiple_data:
            _update_kernels_multiple_data(
                self.inducers, kernels, lhs, rhs,
                INDUCTION_FACTORY[self.induction], self.induction_parameters
            )

        elif multiple_inductions:
            for induction in self.induction:
                _update_kernels_from_inducers(
                    self.inducers, kernels, lhs, rhs,
                    INDUCTION_FACTORY[induction], self.induction_parameters
                )
        else:
            _update_kernels_from_inducers(
                self.inducers, kernels, lhs, rhs,
                INDUCTION_FACTORY[self.induction], self.induction_parameters
            )
        logger.debug('_get_kernels() done')
        return kernels

    # Pickling support

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        state.pop('mkl_model_', None)
        state.pop('estimator_model_', None)
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        # Restore the previously models state.
        if hasattr(self, 'y_') and hasattr(self, 'lhs_'):
            self.fit(self.lhs_, self.y_)
