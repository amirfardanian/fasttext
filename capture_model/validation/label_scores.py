from capture_model.validation.utils import confusion_matrix, get_unique_labels, _get_predicted_probs_and_indicator_array
import numpy as np


class ConfusionMatrixScoringFunction:

    def __init__(self, scorer_fun, name):
        self.scorer_fun = scorer_fun
        self.__name__ = name

    def score(self, model, df_test):
        conf_matrix = confusion_matrix(df_test, model, normalize=False)
        unique_labels = get_unique_labels(model)
        return self.scorer_fun(conf_matrix, unique_labels)

    def __call__(self, model, df_test):
        return self.score(model, df_test)


class ProbabilityScoringFunction:

    def __init__(self, scorer_fun, name):
        self.scorer_fun = scorer_fun
        self.__name__ = name

    def score(self, model, df_test):
        predicted_probs, actual_labels_indic_array = _get_predicted_probs_and_indicator_array(df_test, model)
        unique_labels = get_unique_labels(model)
        return self.scorer_fun(predicted_probs, actual_labels_indic_array, unique_labels)

    def __call__(self, model, df_test):
        return self.score(model, df_test)


def _precision(conf_matrix, unique_labels):
    return {label: conf_matrix[i, i]/conf_matrix[:, i].sum() for i, label in enumerate(unique_labels)}


def _recall(conf_matrix, unique_labels):
    return {label: conf_matrix[i, i]/conf_matrix[i, :].sum() for i, label in enumerate(unique_labels)}


def f1_score(model, df_test):
    _recall = recall(model, df_test)
    _precision = precision(model, df_test)
    unique_labels = get_unique_labels(model)
    return {label: _f_beta_score_value(_recall[label], _precision[label]) for label in unique_labels}


def _f_beta_score_value(recall_value, precision_value, beta=1.0):
    return (1 + beta**2)*(precision_value * recall_value)/(beta**2*precision_value + recall_value)


def _count_actual_labels(conf_matrix, unique_labels):
    return {label: conf_matrix[i, :].sum() for i, label in enumerate(unique_labels)}


def _count_predicted_labels(conf_matrix, unique_labels):
    return {label: conf_matrix[:, i].sum() for i, label in enumerate(unique_labels)}


def _brier_score(predicted_probs, actual_labels_indic_array, unique_labels):
    labels_scores = np.nanmean((predicted_probs - actual_labels_indic_array) ** 2, axis=0) # over observations
    return {label: labels_scores[i] for i, label in enumerate(unique_labels)}


precision = ConfusionMatrixScoringFunction(_precision, 'precision')
recall = ConfusionMatrixScoringFunction(_recall, 'recall')
count_actual_labels = ConfusionMatrixScoringFunction(_count_actual_labels, 'count_actual_labels')
brier_score = ProbabilityScoringFunction(_brier_score, 'brier_score')
