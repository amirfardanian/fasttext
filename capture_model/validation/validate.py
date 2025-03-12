import pandas as pd
import matplotlib.pyplot as plt
import itertools
import copy
import numpy as np
from capture_model.validation.utils import _get_pred_and_actual_labels, _confusion_matrix


def validate(df_test, models, validation_functions, verbose=True):
    validation_results = _validate_models(df_test, models, validation_functions)
    for model_name, validation_result in validation_results.items():
        if verbose:
            print('\nModel: {}'.format(model_name))
            df_val = pd.DataFrame(_format_validation_result_for_dataframe(validation_result))
            print(df_val)
    return validation_results


def _validate_models(df_test, models, validation_functions):
    validation_results = {}
    for model in models:
        validation_results[model.model_name] = {}
        for validation_function in validation_functions:
            val_score_dict = validation_function(model, df_test)
            validation_method_name = _get_name(validation_function)
            validation_results[model.model_name][validation_method_name] = val_score_dict
    return validation_results


def _get_name(validation_function):
    if hasattr(validation_function, '__name__'):
        name = validation_function.__name__
    elif hasattr(validation_function, '__class__') and hasattr(validation_function.__class__, '__name__'):
        name = validation_function.__class__.__name__
    else:
        name = 'validation function name not found'
    return name


def _format_validation_result_for_dataframe(validation_result):
    _validation_result = copy.deepcopy(validation_result)
    contains_label_score = any([isinstance(_validation_result[k], dict) for k in _validation_result])
    if not contains_label_score:
        for key, val_res in _validation_result.items():
            if isinstance(val_res, float) or isinstance(val_res, np.float64):
                _validation_result[key] = [val_res]
    return _validation_result


def confusion_table(df_test, model, normalize=False):
    """Confusion matrix as a DataFrame with labels on index and columns.
    Rows correspond to actual category, columns to predicted category."""
    actual_labels, predicted_labels, unique_labels = _get_pred_and_actual_labels(df_test, model)
    conf_matrix = _confusion_matrix(actual_labels, predicted_labels, unique_labels, normalize)
    conf_table = pd.DataFrame(conf_matrix, index=pd.Series(unique_labels, name='actual_group'), columns=unique_labels)

    return conf_table


def plot_confusion_matrix(df_test, model, normalize=False, filename=None,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    actual_labels, predicted_labels, unique_labels = _get_pred_and_actual_labels(df_test, model)
    conf_matrix = _confusion_matrix(actual_labels, predicted_labels, unique_labels, normalize)
    pretty_labels = _prettify_labels(unique_labels)
    if not title:
        title = 'Normalized confusion matrix: {}' if normalize else 'Confusion matrix: {}'
        title = title.format(model.model_name)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = list(range(len(pretty_labels)))
    plt.xticks(tick_marks, pretty_labels, rotation=90)
    plt.yticks(tick_marks, pretty_labels)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if filename:
        plt.savefig('{}.png'.format(filename))
    else:
        plt.show()


def _prettify_labels(labels):
    return [label.replace('__label__', '') for label in labels]
