from pymimkl import EasyMKL  # TODO KOMD
from sklearn.svm import SVC
ESTIMATOR_FACTORY = {
    'EasyMKL': EasyMKL,
    'SVC': SVC
}
