from .models import PIMKL
from .evaluation import roc_analysis
from .utils.preprocessing import (
    labels_to_one_hot_code_using_dict, Standardizer
)
from .data import get_learning_data_in_dict_mode, get_learning_data
import logging

logger = logging.getLogger('run_pimkl')


def fold_generator(
    number_of_folds,
    data,
    labels,
    max_per_class,
    transformer_class=Standardizer
):
    """generate class balanced splits of data and labels"""
    for fold in range(number_of_folds):
        if isinstance(data, dict):
            data_type_labels = list(data.keys())
            if labels is None:
                X_train, X_test = get_learning_data_in_dict_mode(
                    data,
                    labels=labels,
                    data_types=data_type_labels,
                    max_per_class=max_per_class
                )
                y_train = None
                y_test = None
            else:
                X_train, y_train, X_test, y_test = get_learning_data_in_dict_mode(  # noqa
                    data,
                    labels=labels,
                    data_types=data_type_labels,
                    max_per_class=max_per_class
                )
            for data_type in data_type_labels:
                # learn normalization only on train data
                transformer = transformer_class()
                X_train[data_type] = transformer.apply(X_train[data_type])
                X_test[data_type] = transformer.reapply(X_test[data_type])
        else:
            if labels is None:
                X_train, X_test = get_learning_data(
                    data, labels=labels, max_per_class=max_per_class
                )
                y_train = None
                y_test = None
            else:
                X_train, y_train, X_test, y_test = get_learning_data(
                    data, labels=labels, max_per_class=max_per_class
                )
            X_train = transformer.apply(X_train)
            X_test = transformer.reapply(X_test)
        yield {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'fold': fold
        }


def run_model(
    inducers, induction_name, mkl_name, estimator_name, mkl_parameters,
    estimator_parameters, induction_parameters, inducers_extended_names,
    fold_parameters
):
    """Run a single fold of the model with data splits from fold_generator.

    Arguments are those to PIMKL and then the inducer_names and a dict
    containing the fold specific arguments.
    In junction with partial and the fold_generator it can be used for
    running folds in parallel:
    ```list(pool.imap(run_fold, fold_generator(...)))```
    """
    X_train, y_train, X_test, y_test, fold = map(
        fold_parameters.get,
        ('X_train', 'y_train', 'X_test', 'y_test', 'fold')
    )
    try:
        logger.debug('Training fold {}.'.format(fold))
        model = PIMKL(
            inducers=inducers,
            induction=induction_name,
            mkl=mkl_name,
            estimator=estimator_name,
            induction_parameters=induction_parameters,
            mkl_parameters=mkl_parameters,
            estimator_parameters=estimator_parameters
        )
        if y_train is None or y_test is None:
            model.fit(X_train)
            aucs = {'class': 0.0}
            weights = {
                name: weight
                for name, weight in
                zip(inducers_extended_names, model.kernels_weights)
            }
            trace_factors = {
                name: trace_factor
                for name, trace_factor in
                zip(inducers_extended_names, model.mkl_model_.trace_factors)
            }
        else:
            model.fit(X_train, y_train)
            label_to_index = {
                label: index
                for index, label in enumerate(
                    model.mkl_model_.get_classes_order(
                    ) if hasattr(model.mkl_model_, 'get_classes_order') else
                    model.estimator_model_.get_classes_order()
                )
            }
            index_to_label = {
                index: label
                for label, index in label_to_index.items()
            }
            y_test_one_hot_code = labels_to_one_hot_code_using_dict(
                y_test, label_to_index
            )
            # prediction
            y_score = model.predict_proba(X_test)
            _, _, _, aucs = roc_analysis(y_test_one_hot_code, y_score)
            aucs = {
                index_to_label.get(index, index): value
                for index, value in aucs.items()
            }
            # results
            if len(model.kernels_weights.shape) == 2:  # EasyMKL 1vRest
                weights = [
                    (
                        index_to_label.get(index, index), {
                            name: weight
                            for name, weight in zip(
                                inducers_extended_names, class_kernels_weights
                            )
                        }
                    ) for index, class_kernels_weights in
                    enumerate(model.kernels_weights.T)
                ]
            else:
                weights = {
                    name: weight
                    for name, weight in
                    zip(inducers_extended_names, model.kernels_weights)
                }
            trace_factors = {
                name: trace_factor
                for name, trace_factor in
                zip(inducers_extended_names, model.mkl_model_.trace_factors)
            }

    except Exception as exc:
        print(str(exc))
        logger.exception(exc)
        print('Problem in training for fold {}.'.format(fold))
        return None
    return aucs, weights, trace_factors
