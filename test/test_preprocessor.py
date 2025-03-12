import unittest
from capture_model.modelling.preprocessor import PreProcessor
from capture_model.modelling.transformations import StatelessTransformation
from capture_model.modelling import string_functions as sf
import pandas as pd
import os
from pandas.util.testing import assert_series_equal


class TestPreProcessor(unittest.TestCase):
    def setUp(self):
        self.preproc = PreProcessor('test_preprocessor')

    def test_init(self):
        preproc = PreProcessor('test_preprocessor')

        self.assertEqual(preproc.pp_name, 'test_preprocessor')
        self.assertEqual(preproc.transformations, [])
        self.assertEqual(preproc.pp_hash, '0')
        self.assertIsNotNone(preproc.logger)

    def test_add_transformation(self):
        transformation = StatelessTransformation(function=lambda x: x + 1)
        self.preproc.add_transformation(transformation)

        self.assertEqual(len(self.preproc.transformations), 1)

        first_transformation = self.preproc.transformations[0]
        self.assertEqual(first_transformation, transformation)

    def test_add_transformation_multiple(self):
        self.preproc \
            .add_transformation(StatelessTransformation(function=lambda x: x + 1)) \
            .add_transformation(StatelessTransformation(function=lambda x: x + 2))

        self.assertEqual(len(self.preproc.transformations), 2)

    def test_add_transformation_invalid_string(self):
        transformation = 'a string that is not a transformation'

        with self.assertRaises(AssertionError):
            self.preproc.add_transformation(transformation)

    def test_add_transformation_invalid_list(self):
        transformation = ['alist']

        with self.assertRaises(AssertionError):
            self.preproc.add_transformation(transformation)

    def test_preproc_one(self):
        transformation_one = StatelessTransformation(function=lambda astring: astring.strip())
        transformation_two = StatelessTransformation(function=lambda astring: astring.lower())
        self.preproc.add_transformation(transformation_one)
        self.preproc.add_transformation(transformation_two)

        input = '  Once UPON a time   '
        expected_result = 'once upon a time'
        result = self.preproc.preproc_one(input)
        self.assertEqual(result, expected_result)

    def test_preproc_many_list(self):
        transformation_one = StatelessTransformation(function=lambda astring: astring.strip())
        transformation_two = StatelessTransformation(function=lambda astring: astring.lower())
        self.preproc.add_transformation(transformation_one)
        self.preproc.add_transformation(transformation_two)

        input = ['  HOW ARE YOU???   ', '    PRETTY GOOD']
        expected_result = ['how are you???', 'pretty good']
        result = self.preproc.preproc_many(input)
        self.assertEqual(result, expected_result)

    def test_preproc_many_pandas_series(self):
        transformation_one = StatelessTransformation(function=lambda astring_series: astring_series.str.strip())
        transformation_two = StatelessTransformation(function=lambda astring_series: astring_series.str.lower())
        self.preproc.add_transformation(transformation_one)
        self.preproc.add_transformation(transformation_two)

        input = pd.Series(['  HOW ARE YOU???   ', '    PRETTY GOOD'])
        expected_result = pd.Series(['how are you???', 'pretty good'])
        result = self.preproc.preproc_many(input)
        assert_series_equal(result, expected_result)

    def test_to_pp_info(self):
        def transform_one(item, arg):
            print(item + arg)

        def transform_two(item, arg):
            print(item, arg)

        transformation_one = StatelessTransformation(function=transform_one, arg=1)
        transformation_two = StatelessTransformation(function=transform_two, arg='world')
        self.preproc.add_transformation(transformation_one)
        self.preproc.add_transformation(transformation_two)

        pp_info = self.preproc.to_info_dict()
        expected_pp_info = {
            'pp_hash': '5461929454f012f8fc46f609ea138966cf04a395',
            'pp_name': 'test_preprocessor',
            'transformations': [transformation_one, transformation_two]
        }

        self.assertDictEqual(pp_info, expected_pp_info)

    def test_to_json_no_file_extension(self):
        transformation_one = StatelessTransformation(function=sf.lower)
        transformation_two = StatelessTransformation(function=sf.replace, regex_pattern='old', replacement='new')
        self.preproc.add_transformation(transformation_one)
        self.preproc.add_transformation(transformation_two)

        self.preproc.to_json('test_preprocessor')
        expected_pp_filepath = 'test_preprocessor.json'

        self.assertTrue(os.path.exists(expected_pp_filepath))
        os.remove(expected_pp_filepath)

    def test_to_json_file_extension(self):
        transformation_one = StatelessTransformation(function=sf.lower)
        transformation_two = StatelessTransformation(function=sf.replace, regex_pattern='old', replacement='new')
        self.preproc.add_transformation(transformation_one)
        self.preproc.add_transformation(transformation_two)

        self.preproc.to_json('test_preprocessor.json')
        expected_pp_filepath = 'test_preprocessor.json'

        self.assertTrue(os.path.exists(expected_pp_filepath))
        os.remove(expected_pp_filepath)

    def test_from_json(self):
        filepath = 'fixtures/test_preprocessor.json'
        preproc = PreProcessor.from_json(filepath)

        expected_transformations = [StatelessTransformation(function=sf.lower),
                                    StatelessTransformation(function=sf.replace,
                                                            regex_pattern='old',
                                                            replacement='new')]

        self.assertEqual(preproc.pp_name, 'test_preprocessor')
        self.assertEqual(preproc.pp_hash, 'a50ad6e3d67447e140d97c471ca320ee47c5b18c')
        self.assertListEqual(preproc.transformations, expected_transformations)


if __name__ == '__main__':
    unittest.main()
