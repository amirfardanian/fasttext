import unittest
from capture_model.modelling import FasttextClassifierModel
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json


class TestFasttextModel(unittest.TestCase):

    def setUp(self):
        pass

    def assertNumpyArrayEqual(self, array1, array2):
        self.assertTrue((array1 == array2).all(), msg='array1={} not equal to array1={}'.format(array1, array2))

    def test_init(self):
        model = FasttextClassifierModel('test_model')

        self.assertEqual(model.model_name, 'test_model')
        self.assertIsInstance(model.init_datetime, datetime)
        self.assertEqual(model.class_name, 'FasttextClassifierModel')

    def test_train_dataframe(self):
        model = FasttextClassifierModel('test_model')

        output_name = 'label'
        input_name = 'sentence'

        df = pd.DataFrame({output_name: [1, 1, 2, 2, 1, 3, 2],
                           input_name: ['cow', 'chicken', 'house', 'apartment', 'dog', 'ocean', 'flat']})

        result = model.train(df, input_names=[input_name], output_names=[output_name])

        expected_model_filepath = model.model_name + '.bin'
        self.assertEqual(result, 1)
        self.assertTrue(os.path.exists(expected_model_filepath))
        self.assertListEqual(model.input_names, [input_name])
        self.assertListEqual(model.output_names, [output_name])
        self.assertIsNotNone(model.internal_model)
        os.remove(expected_model_filepath)

    def test_train_file(self):
        model = FasttextClassifierModel('test_model')

        training_data_filepath = 'fixtures/fasttext_training_data.txt'

        result = model.train(dataframe=None, input_names=[], output_names=[], training_filepath=training_data_filepath)
        expected_model_filepath = model.model_name + '.bin'

        self.assertEqual(result, 1)
        self.assertTrue(os.path.exists(expected_model_filepath))
        self.assertIsNotNone(model.internal_model)
        os.remove(expected_model_filepath)

    def test_train_many_input_names(self):
        model = FasttextClassifierModel('test_model')
        training_data_filepath = 'fixtures/fasttext_training_data.txt'

        input_names = ['input1', 'input2']

        with self.assertRaises(ValueError):
            model.train(dataframe=None, input_names=input_names, output_names=[], training_filepath=training_data_filepath)

    def test_train_many_output_names(self):
        model = FasttextClassifierModel('test_model')
        training_data_filepath = 'fixtures/fasttext_training_data.txt'

        output_names = ['output1', 'output2']

        with self.assertRaises(ValueError):
            model.train(dataframe=None, input_names=[], output_names=output_names, training_filepath=training_data_filepath)

    def test_train_unlabelled_data_explicit(self):
        model = FasttextClassifierModel('test_model')
        training_data_filepath = 'fixtures/fasttext_training_data_unlabelled.txt'

        # Un-prefixed label data should be specified as un-prefixed. This is not allowed, though,
        # since Fasttext then will consider all words as labels.

        with self.assertRaises(ValueError):
            model.train(dataframe=None, input_names=[], output_names=[], training_filepath=training_data_filepath, label_prefix='')

    def test_train_unlabelled_data_implicit(self):
        # If the label_prefix for un-prefixed data is not explicitly stated specified as '',
        # the FasttextClassifierModel will not catch it and Fasttext will consider all words labels.
        # The model will thus train, but be a meaningless model.

        model = FasttextClassifierModel('test_model')
        training_data_filepath = 'fixtures/fasttext_training_data_unlabelled.txt'

        # Un-prefixed label data should be specified as un-prefixed. This is not allowed, though,
        # since Fasttext then will consider all words as labels.

        result = model.train(dataframe=None, input_names=[], output_names=[], training_filepath=training_data_filepath)
        expected_model_filepath = model.model_name + '.bin'
        self.assertEqual(result, 1)
        self.assertTrue(os.path.exists(expected_model_filepath))
        self.assertIsNotNone(model.internal_model)
        os.remove(expected_model_filepath)

    def test_to_json(self):
        model = FasttextClassifierModel('test_model')
        training_data_filepath = 'fixtures/fasttext_training_data.txt'
        model.train(dataframe=None, input_names=[], output_names=[], training_filepath=training_data_filepath)
        model.to_json()

        expected_internal_model_filepath = model.model_name + '.bin'
        expected_model_filepath = model.model_name + '.json'

        self.assertTrue(os.path.exists(expected_internal_model_filepath))
        self.assertTrue(os.path.exists(expected_model_filepath))

        with open(expected_model_filepath, 'r') as model_file:
            model_dict = json.load(model_file)

        self.assertEqual(model_dict['model_name'], model.model_name)
        self.assertEqual(model_dict['input_names'], model.input_names)
        self.assertEqual(model_dict['output_names'], model.output_names)
        self.assertEqual(model_dict['storage_dir'], model.storage_dir)
        self.assertEqual(model_dict['internal_model_filepath'], model.internal_model_filepath)

        os.remove(expected_internal_model_filepath)
        os.remove(expected_model_filepath)

    def test_from_json(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)

        expected_model_name = 'fasttext_trained_model'
        expected_storage_dir = '.'
        expected_input_names = []
        expected_output_names = []
        expected_class_name = 'FasttextClassifierModel'
        expected_init_datetime = datetime.strptime('2017-11-26 18:43:50', FasttextClassifierModel.datetime_format)

        self.assertIsInstance(model, FasttextClassifierModel)
        self.assertEqual(expected_model_name, model.model_name)
        self.assertEqual(expected_storage_dir, model.storage_dir)
        self.assertEqual(expected_input_names, model.input_names)
        self.assertEqual(expected_output_names, model.output_names)
        self.assertEqual(expected_class_name, model.class_name)
        self.assertEqual(expected_init_datetime, model.init_datetime)
        self.assertIsNotNone(model.internal_model)

    def test_score_one(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)

        input = 'donkey'
        k = 1
        result = model.predict_one(input, k_best=k)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), k)
        self.assertIsInstance(result[0], str)

    def test_score_one_incl_probabilities(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)

        input = 'donkey\n'
        k = 1
        result = model.predict_one(input, k_best=k, incl_probabilities=True)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), k)
        self.assertIsInstance(result[0], tuple)
        self.assertIsInstance(result[0][0], str)
        self.assertGreaterEqual(result[0][1], 0)
        self.assertLessEqual(result[0][1], 1)

    def test_score_one_untrained(self):
        model = FasttextClassifierModel('test_model')

        input = 'donkey'
        k = 1

        with self.assertRaises(ValueError):
            model.predict_one(input, k_best=k)

    def test_score_many_list(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)

        input = ['donkey', 'shed', 'sea']
        k = 1
        result = model.predict_many(input, k_best=k)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(input))
        self.assertEqual(len(result[0]), k)

    def test_score_many_dataframe(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)
        model.input_names = ['input']
        model.output_names = ['output']

        input = pd.DataFrame({'input': ['donkey', 'shed', 'sea']})
        k = 1

        result = model.predict_many(input, k_best=k)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(input))

    def test_score_many_series(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)

        input = pd.Series(['donkey', 'shed', 'sea'], name='input')
        k = 1

        result = model.predict_many(input, k_best=k)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(input))

    def test_score_many_list_incl_probabilities(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)

        input = ['donkey', 'shed', 'sea']
        k = 3
        result = model.predict_many(input, k_best=k, incl_probabilities=True)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(input))
        self.assertEqual(len(result[0]), k)
        self.assertIsInstance(result[0][0], tuple)
        self.assertIsInstance(result[0][0][0], str)
        self.assertGreaterEqual(result[0][0][1], 0)
        self.assertLessEqual(result[0][0][1], 1)

    def test_score_many_series_incl_probabilities(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)

        input = pd.Series(['donkey', 'shed', 'sea'])
        k = 3
        result = model.predict_many(input, k_best=k, incl_probabilities=True)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(input))
        self.assertEqual(len(result[0]), k)
        self.assertIsInstance(result[0][0], tuple)
        self.assertIsInstance(result[0][0][0], str)
        self.assertGreaterEqual(result[0][0][1], 0)
        self.assertLessEqual(result[0][0][1], 1)

    def test_score_best_list(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)

        input = ['donkey', 'shed', 'sea']
        result = model.predict_many_best(input)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(input))
        self.assertEqual(result[0], '1')

    def test_score_best_dataframe(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)
        model.input_names = ['input']
        model.output_names = ['output']

        input = pd.DataFrame({'input': ['donkey', 'shed', 'sea']})

        result = model.predict_many_best(input)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(input))
        self.assertEqual(result[0], '1')

    def test_score_best_series(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)

        input = pd.Series(['donkey', 'shed', 'sea'], name='input')

        result = model.predict_many_best(input)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(input))
        self.assertEqual(result[0], '1')

    def test_predict_proba_matrix(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)

        input = pd.Series(['donkey', 'shed', 'sea'], name='input')
        expected_probs_array = np.array([
            [0.33398440736152024, 0.3320312896119305, 0.3320312896119305],
            [0.33398440736152024, 0.3320312896119305, 0.3320312896119305],
            [0.33398440736152024, 0.3320312896119305, 0.3320312896119305]
        ])

        result = model.predict_proba(input)
        self.assertNumpyArrayEqual(result, expected_probs_array)

    def test_get_labels(self):
        model_path = 'fixtures/fasttext_trained_model.json'
        model = FasttextClassifierModel.from_json(model_path)
        unique_sorted_labels = model.get_labels()
        expected_labels = ['1', '2', '3']
        self.assertListEqual(unique_sorted_labels, expected_labels)

if __name__ == '__main__':
    unittest.main()
