import unittest
import pandas as pd
import os
import copy
import numpy as np
import capture_model.validation as val
from capture_model.validation.validate import _get_pred_and_actual_labels, _confusion_matrix, _validate_models
from capture_model.validation.utils import _get_predicted_probs_and_indicator_array
import capture_model.validation.label_scores as label_scores
import capture_model.validation.overall_scores as overall_scores
from capture_model.modelling.fasttext_model import FasttextClassifierModel


class TestValidate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_train = pd.read_csv('./fixtures/fasttext_training_data.csv', index_col=0)
        cls.df_test = pd.read_csv('./fixtures/fasttext_test_data.csv', index_col=0, dtype={'label': str})

        model = FasttextClassifierModel('validation_model_1')
        model.train(cls.df_train,
                    input_names=['words'],
                    output_names=['label'],
                    verbose=False,
                    label_prefix='__label__')
        model.to_json()

        model = FasttextClassifierModel('validation_model_2')
        model.train(cls.df_train,
                    input_names=['words'],
                    output_names=['label'],
                    verbose=False,
                    label_prefix='__label__')
        model.to_json()

        # WORKAROUND! Because fasttext model removes the prefix from the labels if the model isn't saved an loaded
        cls.model_1 = FasttextClassifierModel.from_json('validation_model_1.json')
        cls.model_2 = FasttextClassifierModel.from_json('validation_model_2.json')

        cls.actual_labels_list = ['__label__a', '__label__b', '__label__c', '__label__a', '__label__c']
        cls.predicted_labels_list_all_a = ['__label__a', '__label__a', '__label__a', '__label__a', '__label__a']
        cls.predicted_labels_list_mixed = ['__label__a', '__label__b', '__label__c', '__label__a', '__label__b']

        cls.actual_labels_series = pd.Series(['__label__a', '__label__b', '__label__c', '__label__a', '__label__c'])
        cls.predicted_labels_series_all_a = pd.Series(['__label__a', '__label__a', '__label__a', '__label__a', '__label__a'])
        cls.predicted_labels_series_mixed = pd.Series(['__label__a', '__label__b', '__label__c', '__label__a', '__label__b'])

    @classmethod
    def tearDownClass(cls):
        os.remove('validation_model_1.json')
        os.remove('validation_model_2.json')

    def assertNumpyArrayEqual(self, array1, array2):
        self.assertTrue((array1 == array2).all(), msg='array1={} not equal to array1={}'.format(array1, array2))

    def assertValidLabelScores(self, label_score_dict):
        self.assertIsInstance(label_score_dict, dict)
        for key, score in label_score_dict.items():
            # with self.subTest(msg=key):
            self.assertIsInstance(score, np.float)

    def assertValidOverallScore(self, overall_score):
        self.assertIsInstance(overall_score, np.float)


class TestPredictionFunctions(TestValidate):

    def test_get_actual_and_pred_labels(self):
        actual_labels, predicted_labels, unique_labels = _get_pred_and_actual_labels(self.df_test, self.model_1)
        self.assertIsInstance(actual_labels, pd.Series)
        self.assertIsInstance(predicted_labels, pd.Series)
        self.assertIsInstance(actual_labels[actual_labels.index[0]], str)
        self.assertIsInstance(predicted_labels[predicted_labels.index[0]], str)
        self.assertListEqual(unique_labels, ['1', '2', '3'])

    def test_get_actual_and_pred_labels_None(self):
        df_error = pd.DataFrame({'label': ['1', '2', '3', None], 'words': ['dog', 'house', 'ocean', 'car']})
        actual_labels, predicted_labels, unique_labels = _get_pred_and_actual_labels(df_error, self.model_1)
        self.assertIsInstance(actual_labels, pd.Series)
        self.assertIsInstance(predicted_labels, pd.Series)
        self.assertIsInstance(actual_labels[actual_labels.index[0]], str)
        self.assertIsInstance(predicted_labels[predicted_labels.index[0]], str)
        self.assertListEqual(unique_labels, ['1', '2', '3'])

    def test_get_predicted_probs_and_indicator_array(self):
        indic_matrix, pred_probs_matrix = _get_predicted_probs_and_indicator_array(self.df_test, self.model_1)
        expected_indic_matrix = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
        ], dtype=int)
        self.assertNumpyArrayEqual(expected_indic_matrix, indic_matrix)
        self.assertIsInstance(pred_probs_matrix, np.ndarray)
        self.assertEqual(pred_probs_matrix.shape, (6, 3))


class TestLabelScoringFunctions(TestValidate):

    def test__confusion_matrix_all_a_list(self):
        unique_labels = ['__label__a', '__label__b', '__label__c']
        conf_matrix = _confusion_matrix(self.actual_labels_list, self.predicted_labels_list_all_a, unique_labels)
        expected_conf_matrix = np.array([[2, 0, 0], [1, 0, 0], [2, 0, 0]])
        self.assertTrue((conf_matrix == expected_conf_matrix).all())

    def test__confusion_matrix_all_None(self):
        unique_labels = ['__label__a', '__label__b', 'None']
        actual_labels = ['__label__a', '__label__b', 'None', '__label__a', 'None']
        predicted_labels =['__label__a', '__label__a', '__label__a', '__label__a', '__label__a']
        conf_matrix = _confusion_matrix(actual_labels, predicted_labels, unique_labels)
        expected_conf_matrix = np.array([[2, 0, 0], [1, 0, 0], [2, 0, 0]])
        self.assertTrue((conf_matrix == expected_conf_matrix).all())

    def test__confusion_matrix_mixed_list(self):
        unique_labels = ['__label__a', '__label__b', '__label__c']
        conf_matrix = _confusion_matrix(self.actual_labels_list, self.predicted_labels_list_mixed, unique_labels)
        expected_conf_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 1, 1]])
        self.assertTrue((conf_matrix == expected_conf_matrix).all(), msg='\n' + str(conf_matrix == expected_conf_matrix))

    def test__confusion_matrix_all_a_series(self):
        unique_labels = ['__label__a', '__label__b', '__label__c']
        conf_matrix = _confusion_matrix(self.actual_labels_series, self.predicted_labels_series_all_a, unique_labels)
        expected_conf_matrix = np.array([[2, 0, 0], [1, 0, 0], [2, 0, 0]])
        self.assertTrue((conf_matrix == expected_conf_matrix).all(), msg='\n' + str(conf_matrix == expected_conf_matrix))

    def test__confusion_matrix_mixed_series(self):
        unique_labels = ['__label__a', '__label__b', '__label__c']
        conf_matrix = _confusion_matrix(self.actual_labels_series, self.predicted_labels_series_mixed, unique_labels)
        expected_conf_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 1, 1]])
        self.assertTrue((conf_matrix == expected_conf_matrix).all(), msg='\n' + str(conf_matrix == expected_conf_matrix))

    def test__confusion_matrix_mixed_series_normalized(self):
        unique_labels = ['__label__a', '__label__b', '__label__c']
        conf_matrix = _confusion_matrix(self.actual_labels_series, self.predicted_labels_series_mixed, unique_labels, normalize=True)
        expected_conf_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]])
        self.assertTrue((conf_matrix == expected_conf_matrix).all(), msg='\n' + str(conf_matrix == expected_conf_matrix))

    def test_confusion_matrix(self):
        conf_matrix = val.confusion_matrix(self.df_test, self.model_1)
        self.assertIsInstance(conf_matrix, np.ndarray)
        self.assertEqual(conf_matrix.shape, (3, 3))

    def test_confusion_table(self):
        conf_table = val.confusion_table(self.df_test, self.model_1)
        self.assertIsInstance(conf_table, pd.DataFrame)
        self.assertEqual(conf_table.shape, (3, 3))

    def test__precision(self):
        unique_labels = ['__label__a', '__label__b', '__label__c']
        conf_matrix = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
        precision = label_scores._precision(conf_matrix, unique_labels)
        self.assertIsInstance(precision, dict)
        self.assertAlmostEqual(precision['__label__a'], 1.0)
        self.assertAlmostEqual(precision['__label__b'], 1 / 3)
        self.assertAlmostEqual(precision['__label__c'], 1.0)

    def test__precision_zero_hits_and_zero_predictions(self):
        unique_labels = ['__label__a', '__label__b', '__label__c']
        conf_matrix = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 0]])
        precision = label_scores._precision(conf_matrix, unique_labels)
        self.assertTrue(np.isnan(precision['__label__a']))
        self.assertAlmostEqual(precision['__label__b'], 0.0)
        self.assertAlmostEqual(precision['__label__c'], 0.0)

    def test_precision_happy_path(self):
        precision = label_scores.precision(self.model_1, self.df_test)
        self.assertValidLabelScores(precision)

    def test_precision_none_input(self):
        with self.assertRaises(Exception):
            label_scores.precision(None, self.df_test)
        with self.assertRaises(Exception):
            label_scores.precision(self.model_1, None)

    def test_precision_corrupted_model(self):
        model = copy.copy(self.model_1)
        delattr(model, 'internal_model')
        with self.assertRaises(Exception):
            label_scores.precision(model, self.df_test)

    def test__recall(self):
        unique_labels = ['__label__a', '__label__b', '__label__c']
        conf_matrix = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
        recall = label_scores._recall(conf_matrix, unique_labels)
        self.assertIsInstance(recall, dict)
        self.assertAlmostEqual(recall['__label__a'], 1 / 2)
        self.assertAlmostEqual(recall['__label__b'], 1.0)
        self.assertAlmostEqual(recall['__label__c'], 1 / 2)

    def test__recall_zero_hits_and_zero_predictions(self):
        unique_labels = ['__label__a', '__label__b', '__label__c']
        conf_matrix = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]])
        recall = label_scores._recall(conf_matrix, unique_labels)
        self.assertIsInstance(recall, dict)
        self.assertTrue(np.isnan(recall['__label__a']))
        self.assertAlmostEqual(recall['__label__b'], 0.0)
        self.assertAlmostEqual(recall['__label__c'], 0.0)

    def test_recall_happy_path(self):
        recall = label_scores.recall(self.model_1, self.df_test)
        self.assertValidLabelScores(recall)

    def test_recall_none_input(self):
        with self.assertRaises(Exception):
            label_scores.recall(None, self.df_test)
        with self.assertRaises(Exception):
            label_scores.recall(self.model_1, None)

    def test_recall_corrupted_model(self):
        model = copy.copy(self.model_1)
        delattr(model, 'internal_model')
        with self.assertRaises(Exception):
            label_scores.recall(model, self.df_test)

    def test_f1_score(self):
        f1s = label_scores.f1_score(self.model_1, self.df_test)
        self.assertValidLabelScores(f1s)

    def test__brier_score(self):
        unique_labels = ['__label__a', '__label__b', '__label__c']

        # Perfect prediction
        predicted_probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        actual_labels_indic_array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        brier_scores = label_scores._brier_score(predicted_probs, actual_labels_indic_array, unique_labels)
        self.assertIsInstance(brier_scores, dict)
        self.assertAlmostEqual(brier_scores['__label__a'], 0.0)
        self.assertAlmostEqual(brier_scores['__label__b'], 0.0)
        self.assertAlmostEqual(brier_scores['__label__c'], 0.0)

        # Worst prediction
        predicted_probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        actual_labels_indic_array = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        brier_scores = label_scores._brier_score(predicted_probs, actual_labels_indic_array, unique_labels)
        self.assertIsInstance(brier_scores, dict)
        self.assertAlmostEqual(brier_scores['__label__a'], 2/3)
        self.assertAlmostEqual(brier_scores['__label__b'], 2/3)
        self.assertAlmostEqual(brier_scores['__label__c'], 2/3)

        # Random prediction
        predicted_probs = np.random.uniform(size=(3, 3))
        predicted_probs = predicted_probs * 1/predicted_probs.sum(axis=0)
        actual_labels_indic_array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        brier_scores = label_scores._brier_score(predicted_probs, actual_labels_indic_array, unique_labels)
        self.assertIsInstance(brier_scores, dict)
        self.assertGreater(brier_scores['__label__a'], 0.0)
        self.assertGreater(brier_scores['__label__b'], 0.0)
        self.assertGreater(brier_scores['__label__c'], 0.0)
        self.assertLess(brier_scores['__label__a'], 1.0)
        self.assertLess(brier_scores['__label__b'], 1.0)
        self.assertLess(brier_scores['__label__c'], 1.0)

    def test_brier_score(self):
        brier_scores = label_scores.f1_score(self.model_1, self.df_test)
        self.assertValidLabelScores(brier_scores)


class TestOverallScoringFunctions(TestValidate):

    def test__overall_accuracy(self):
        # sum(true positives) / N
        conf_matrix = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
        accuracy = overall_scores._overall_accuracy(conf_matrix)
        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual(accuracy, 3 / 5)

    def test__overall_precision(self):
        # mean(true positives / (true positives + false positives))
        conf_matrix = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
        precision = overall_scores._overall_precision(conf_matrix)
        self.assertAlmostEqual(precision, 7 / 9)

    def test__overall_recall(self):
        # mean(true positives / (true positives + false negatives))
        conf_matrix = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
        recall = overall_scores._overall_recall(conf_matrix)
        self.assertAlmostEqual(recall, 2 / 3)

    def test__overall_f1_score(self):
        # 2 / (1/precision + 1/recall)
        conf_matrix = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
        f1_score = overall_scores._overall_f1_score(conf_matrix)
        self.assertAlmostEqual(f1_score, 28 / 39)

    def test__overall_brier_score(self):
        predicted_probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        actual_labels_indic_array = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        brier_score = overall_scores._overall_brier_score(predicted_probs, actual_labels_indic_array)
        self.assertAlmostEqual(brier_score, 2.0)
        predicted_probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        actual_labels_indic_array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        brier_score = overall_scores._overall_brier_score(predicted_probs, actual_labels_indic_array)
        self.assertAlmostEqual(brier_score, 0.0)

    def test_overall_accuracy(self):
        accuracy = overall_scores.overall_accuracy(self.model_1, self.df_test)
        self.assertIsInstance(accuracy, float)

    def test_overall_precision(self):
        precision = val.overall_precision(self.model_1, self.df_test)
        self.assertIsInstance(precision, float)

    def test_overall_recall(self):
        recall = val.overall_recall(self.model_1, self.df_test)
        self.assertIsInstance(recall, float)

    def test_overall_f1_score(self):
        # 2 / (1/precision + 1/recall)
        f1_score = val.overall_f1_score(self.model_1, self.df_test)
        self.assertIsInstance(f1_score, float)

    def test_overall_brier_score(self):
        brier_score = val.overall_brier_score(self.model_1, self.df_test)
        self.assertIsInstance(brier_score, float)


class TestValidateFunction(TestValidate):
    """Test validate method on label scoring functions and overall scoring functions"""

    def assertValidValidationDict(self, val_dict, n_models, n_val_functions):
        self.assertIsInstance(val_dict, dict)
        self.assertEqual(len(val_dict), n_models)
        for model, score_dict in val_dict.items():
            self.assertEqual(len(score_dict), n_val_functions)
            for score_name, score in score_dict.items():
                # with self.subTest(msg='model: , scoring function:'.format(model, score_name)):
                if score_name.startswith('overall'):
                    self.assertValidOverallScore(score)
                else:
                    self.assertValidLabelScores(score)

    def test__validate_models(self):
        models = [self.model_1, self.model_2]
        val_functions = [val.precision, val.recall]
        validation_results = _validate_models(self.df_test, models, val_functions)
        self.assertValidValidationDict(validation_results, n_models=2, n_val_functions=2)

    def test_validate(self):
        models = [self.model_1, self.model_2]
        val_functions = [val.precision, val.recall]
        validation_results = val.validate(self.df_test, models, val_functions)
        self.assertValidValidationDict(validation_results , n_models=2, n_val_functions=2)

    # Test validate method on overall scoring functions
    def test__validate_models_with_overall_scores(self):
        models = [self.model_1, self.model_2]
        val_functions = [val.overall_precision, val.overall_recall]
        validation_results = _validate_models(self.df_test, models, val_functions)
        self.assertValidValidationDict(validation_results, n_models=2, n_val_functions=2)

    def test_validate_with_both_overall_and_label_scores(self):
        models = [self.model_1, self.model_2]
        val_functions = [val.overall_accuracy, val.precision, val.recall]
        validation_results = val.validate(self.df_test, models, val_functions)
        self.assertValidValidationDict(validation_results, n_models=2, n_val_functions=3)

    def test_validate_with_only_overall_scores(self):
        models = [self.model_1, self.model_2]
        val_functions = [val.overall_accuracy, val.overall_precision, val.overall_recall, val.overall_f1_score]
        validation_results = val.validate(self.df_test, models, val_functions)
        self.assertValidValidationDict(validation_results, n_models=2, n_val_functions=4)

if __name__ == '__main__':
    unittest.main()
