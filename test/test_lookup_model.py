import unittest
from datetime import datetime
from capture_model.modelling.lookup_model import LookUpClassifierModel
import pandas as pd
import os
import json


class TestLookUpModel(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        model = LookUpClassifierModel('test_model')
        self.assertIsInstance(model, LookUpClassifierModel)
        self.assertEqual(model.model_name, 'test_model')
        self.assertIsInstance(model.init_datetime, datetime)

    def test_train(self):
        model = LookUpClassifierModel('test_model')
        csv_filepath = 'fixtures/lookup_training.csv'
        result = model.train(csv_filepath=csv_filepath,
                             input_names=['words'],
                             output_names=['label'])

        expected_lookup_table = {
            'sea': "3",
            'doggie': "1",
            'chick': "1",
            'apartment building': "2",
            'apartment house': "2",
            'duck': "1",
        }

        self.assertEqual(result, 1)
        self.assertDictEqual(model.look_up_table, expected_lookup_table)
        self.assertEqual(model.look_up_csv_filename, 'lookup_training.csv')

        os.remove('lookup_training.csv')

    def test_predict_one(self):
        model = LookUpClassifierModel('test_model')
        csv_filepath = 'fixtures/lookup_training.csv'
        model.train(csv_filepath=csv_filepath,
                    input_names=['words'],
                    output_names=['label'])

        # Without probabilities
        pred_category = model.predict_one('chick')
        self.assertIsInstance(pred_category, list)
        self.assertEqual(pred_category[0], "1")

        # Include probabilities
        pred_output = model.predict_one('chick', incl_probabilities=True)
        self.assertIsInstance(pred_output, list)
        self.assertIsInstance(pred_output[0], tuple)
        self.assertEqual(pred_output[0][0], '1')
        self.assertEqual(pred_output[0][1], 1.0)

        # Item does not exist
        pred_output = model.predict_one('something not in csv')
        self.assertIsInstance(pred_output, list)
        self.assertIsNone(pred_output[0])

        # Item does not exist with probabilities
        pred_output = model.predict_one('something not in csv', incl_probabilities=True)
        self.assertIsInstance(pred_output, list)
        self.assertIsNone(pred_output[0][0])
        self.assertEqual(pred_output[0][1], 0)

        os.remove('lookup_training.csv')

    def test_predict_many(self):
        model = LookUpClassifierModel('test_model')
        csv_filepath = 'fixtures/lookup_training.csv'
        model.train(csv_filepath=csv_filepath,
                    input_names=['words'],
                    output_names=['label'])
        items = ['chicks', 'duck', 'something not in csv']

        # Without probabilities
        pred_out = model.predict_many(items)
        self.assertIsInstance(pred_out, list)
        self.assertIsInstance(pred_out[0], list)
        self.assertIsInstance(pred_out[1], list)
        self.assertIsNone(pred_out[2][0])

        # With probabilities
        pred_out = model.predict_many(items, incl_probabilities=True)
        self.assertIsInstance(pred_out, list)
        self.assertIsInstance(pred_out[0][0], tuple)
        self.assertIsInstance(pred_out[1][0], tuple)
        self.assertIsInstance(pred_out[2][0], tuple)
        self.assertIsNone(pred_out[2][0][0])
        self.assertEqual(pred_out[2][0][1], 0)

        # pd.Series
        pred_out = model.predict_many(pd.Series(items))
        self.assertIsInstance(pred_out, pd.Series)
        self.assertEqual(len(pred_out), len(items))
        self.assertIsInstance(pred_out[0], list)

        # pd.DataFrame
        pred_out = model.predict_many(pd.DataFrame({'words': items}))
        self.assertIsInstance(pred_out, pd.Series)
        self.assertEqual(len(pred_out), len(items))
        self.assertIsInstance(pred_out[0], list)

        os.remove('lookup_training.csv')

    def test_predict_many_best(self):
        model = LookUpClassifierModel('test_model')
        csv_filepath = 'fixtures/lookup_training.csv'
        model.train(csv_filepath=csv_filepath,
                    input_names=['words'],
                    output_names=['label'])
        items = ['chicks', 'duck', 'something not in csv']

        # List
        pred_out = model.predict_many_best(items)
        self.assertIsInstance(pred_out, list)
        self.assertEqual(len(pred_out), len(items))
        self.assertTrue(isinstance(pred_out[0], (str, type(None))))

        # pd.Series
        pred_out = model.predict_many_best(pd.Series(items))
        self.assertIsInstance(pred_out, pd.Series)
        self.assertEqual(len(pred_out), len(items))
        self.assertTrue(isinstance(pred_out[0], (str, type(None))))

        # pd.DataFrame
        pred_out = model.predict_many_best(pd.DataFrame({'words': items}))
        self.assertIsInstance(pred_out, pd.Series)
        self.assertEqual(len(pred_out), len(items))
        self.assertTrue(isinstance(pred_out[0], (str, type(None))))

        os.remove('lookup_training.csv')

    def test_predict_missing(self):
        model = LookUpClassifierModel('test_model')
        csv_filepath = 'fixtures/lookup_training.csv'
        model.train(csv_filepath=csv_filepath,
                    input_names=['words'],
                    output_names=['label'],
                    missing_label='missing')
        items = ['something not in csv']

        result = model.predict_one(items[0])
        self.assertEqual(result[0], 'missing')

        result = model.predict_many(items)
        self.assertEqual(result[0][0], 'missing')

        result = model.predict_many(pd.Series(items))
        self.assertEqual(result[0][0], 'missing')

        result = model.predict_many_best(items)
        self.assertEqual(result[0], 'missing')

        result = model.predict_many_best(pd.Series(items))
        self.assertEqual(result[0], 'missing')


    def test_to_model_info_empty_model(self):
        model = LookUpClassifierModel('test_model')
        model_info = model._to_model_info()
        self.assertIsInstance(model_info, dict)
        self.assertEqual(model_info['storage_dir'], '')
        self.assertEqual(model_info['look_up_csv_filename'], None)
        self.assertEqual(model_info['input_names'], None)
        self.assertEqual(model_info['output_names'], None)
        self.assertIsInstance(model_info['init_datetime'], str)

    def test_to_model_info_trained_model(self):
        model = LookUpClassifierModel('test_model')
        csv_filepath = 'fixtures/lookup_training.csv'
        model.train(csv_filepath=csv_filepath,
                    input_names=['words'],
                    output_names=['label'])
        model_info = model._to_model_info()
        self.assertIsInstance(model_info, dict)
        self.assertEqual(model_info['storage_dir'], '')
        self.assertEqual(model_info['look_up_csv_filename'], 'lookup_training.csv')
        self.assertEqual(model_info['input_names'], ['words'])
        self.assertEqual(model_info['output_names'], ['label'])
        self.assertIsInstance(model_info['init_datetime'], str)

        os.remove('lookup_training.csv')

    def test_to_json(self):
        model = LookUpClassifierModel('test_model')
        csv_filepath = 'fixtures/lookup_training.csv'
        model.train(csv_filepath=csv_filepath,
                    input_names=['words'],
                    output_names=['label'])
        model.to_json()

        expected_json_filepath = model.model_name + '.json'

        self.assertTrue(os.path.exists(expected_json_filepath))

        with open(expected_json_filepath, 'r') as model_file:
            model_dict = json.load(model_file)

        self.assertEqual(model_dict['model_name'], model.model_name)
        self.assertEqual(model_dict['input_names'], model.input_names)
        self.assertEqual(model_dict['output_names'], model.output_names)
        self.assertEqual(model_dict['storage_dir'], model.storage_dir)
        self.assertEqual(model_dict['look_up_csv_filename'], model.look_up_csv_filename)

        os.remove(expected_json_filepath)
        os.remove('lookup_training.csv')

    def test_from_model_info(self):
        model_info_dict = {
            'model_name': 'test_model',
            'storage_dir': '',
            'look_up_csv_filename': 'lookup_training.csv',
            'input_names': ['words'],
            'output_names': ['label'],
            'missing_label': 'unknown'
        }
        model = LookUpClassifierModel._from_model_info(model_info_dict)

        self.assertIsInstance(model, LookUpClassifierModel)

        self.assertEqual(model_info_dict['model_name'], model.model_name)
        self.assertEqual(model_info_dict['input_names'], model.input_names)
        self.assertEqual(model_info_dict['output_names'], model.output_names)
        self.assertEqual(model_info_dict['storage_dir'], model.storage_dir)
        self.assertEqual(model_info_dict['look_up_csv_filename'], model.look_up_csv_filename)
        self.assertEqual(model_info_dict['missing_label'], model.missing_label)

    def test_from_json(self):
        model_path = 'fixtures/lookup_trained_model.json'
        model = LookUpClassifierModel.from_json(model_path)

        expected_model_name = 'test_model'
        expected_storage_dir = ''
        expected_class_name = 'LookUpClassifierModel'
        expected_lookup_csv_filename = 'lookup_training.csv'
        expected_input_names = ['words']
        expected_output_names = ['label']
        expected_missing_label = None

        expected_lookup_table = {
            'sea': "3",
            'doggie': "1",
            'chick': "1",
            'apartment building': "2",
            'apartment house': "2",
            'duck': "1",
        }
        self.assertIsInstance(model, LookUpClassifierModel)
        self.assertEqual(expected_model_name, model.model_name)
        self.assertEqual(expected_storage_dir, model.storage_dir)
        self.assertEqual(expected_input_names, model.input_names)
        self.assertEqual(expected_output_names, model.output_names)
        self.assertEqual(expected_class_name, model.class_name)
        self.assertEqual(expected_input_names, model.input_names)
        self.assertEqual(expected_output_names, model.output_names)
        self.assertEqual(expected_lookup_csv_filename, model.look_up_csv_filename)
        self.assertEqual(expected_missing_label, model.missing_label)
        self.assertDictEqual(expected_lookup_table, model.look_up_table)

    def test_train_save_load_other_location(self):
        model = LookUpClassifierModel('test_model', storage_dir='fixtures')
        csv_filepath = 'fixtures/lookup_training.csv'
        result = model.train(csv_filepath=csv_filepath,
                             input_names=['words'],
                             output_names=['label'],
                             missing_label='unknown')

        model.to_json()

        loaded_model = model.from_json('fixtures/test_model.json')

        expected_lookup_table = {
            'sea': "3",
            'doggie': "1",
            'chick': "1",
            'apartment building': "2",
            'apartment house': "2",
            'duck': "1",
        }

        self.assertEqual(result, 1)
        self.assertDictEqual(loaded_model.look_up_table, expected_lookup_table)
        self.assertEqual(loaded_model.look_up_csv_filename, 'lookup_training.csv')
        self.assertEqual(loaded_model.missing_label, 'unknown')

        os.remove('fixtures/test_model.json')

    def test_get_labels(self):
        model = LookUpClassifierModel.from_json('fixtures/lookup_trained_model.json')
        sorted_unique_lables = model.get_labels()
        exepectd_labels = ['1', '2', '3']
        self.assertListEqual(sorted_unique_lables, exepectd_labels)

if __name__ == '__main__':
    unittest.main()