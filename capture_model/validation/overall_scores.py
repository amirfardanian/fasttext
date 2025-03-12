import numpy as np
from capture_model.validation.label_scores import _f_beta_score_value
from capture_model.validation.utils import confusion_matrix, _get_predicted_probs_and_indicator_array


def overall_accuracy(model, df_test):
    conf_matrix = confusion_matrix(df_test, model, normalize=False)
    return _overall_accuracy(conf_matrix)


def _overall_accuracy(conf_matrix):
    total_hits = np.trace(conf_matrix)
    total_obs = conf_matrix.sum()
    return total_hits/total_obs


def overall_precision(model, df_test):
    conf_matrix = confusion_matrix(df_test, model, normalize=False)
    return _overall_precision(conf_matrix)


def _overall_precision(conf_matrix):
    dim = conf_matrix.shape[0]
    return sum([np.nan_to_num(conf_matrix[i, i])/np.nansum(conf_matrix[:, i]) for i in range(dim)])/dim


def overall_recall(model, df_test):
    conf_matrix = confusion_matrix(df_test, model, normalize=False)
    return _overall_recall(conf_matrix)


def _overall_recall(conf_matrix):
    dim = conf_matrix.shape[0]
    return sum([np.nan_to_num(conf_matrix[i, i])/np.nansum(conf_matrix[i, :]) for i in range(dim)])/dim


def overall_f1_score(model, df_test):
    _recall = overall_recall(model, df_test)
    _precision = overall_precision(model, df_test)
    return _f_beta_score_value(_recall, _precision, beta=1.0)


def _overall_f1_score(conf_matrix):
    _recall = _overall_recall(conf_matrix)
    _precision = _overall_precision(conf_matrix)
    return _f_beta_score_value(_recall, _precision, beta=1.0)


def overall_brier_score(model, df_test):
    indic_array, probs_array = _get_predicted_probs_and_indicator_array(df_test, model)
    return _overall_brier_score(probs_array, indic_array)


def _overall_brier_score(probs_array, indic_array):
    return np.nanmean(np.nansum((probs_array-indic_array)**2, axis=1))
