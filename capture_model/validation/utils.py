import numpy as np
import functools
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


# @functools.lru_cache(maxsize=4)
def _get_pred_and_actual_labels(df_test, model):
    input_names = model.input_names
    output_names = model.output_names
    actual_labels = df_test[output_names[0]]
    input_vars = df_test[input_names[0]]
    predicted_labels = model.predict_many_best(input_vars)
    unique_labels = get_unique_labels(model)
    return actual_labels, predicted_labels, unique_labels


# @functools.lru_cache(maxsize=4)
def _get_predicted_probs_and_indicator_array(df_test, model):
    input_names = model.input_names
    output_names = model.output_names
    actual_labels = np.array(df_test[output_names[0]])
    input_vars = df_test[input_names[0]]
    probs_array = model.predict_proba(input_vars)
    unique_labels = np.array(get_unique_labels(model))
    indic_array = (actual_labels.reshape(-1, 1) == unique_labels.reshape(1, -1)).astype(int)
    return indic_array, probs_array


# @functools.lru_cache(maxsize=4)
def confusion_matrix(df_test, model, normalize=False):
    actual_labels, predicted_labels, unique_labels = _get_pred_and_actual_labels(df_test, model)
    return _confusion_matrix(actual_labels, predicted_labels, unique_labels, normalize)


def _confusion_matrix(actual_labels, predicted_labels, unique_labels, normalize=False):
    conf_matrix = sk_confusion_matrix(actual_labels, predicted_labels, labels=unique_labels)
    if normalize:
        return conf_matrix/conf_matrix.sum(axis=1).reshape(-1, 1)
    else:
        return conf_matrix


def get_unique_labels(model):
    unique_labels = model.get_labels()
    unique_labels = sorted((label if label is not None else 'None' for label in unique_labels))
    return unique_labels