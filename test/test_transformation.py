import unittest
from capture_model.modelling.transformations import StatelessTransformation
import pandas as pd
from pandas.util.testing import assert_series_equal


class TestTransformation(unittest.TestCase):
    def test_init(self):
        def transform_function(x):
            return x

        transformation = StatelessTransformation(function=transform_function)
        self.assertEqual(transformation.function, transform_function)
        self.assertDictEqual(transformation.kwargs, {})

    def test_init_kwargs(self):
        def transform_function(x):
            return x

        transformation = StatelessTransformation(function=transform_function, x=1)
        self.assertEqual(transformation.function, transform_function)
        self.assertDictEqual(transformation.kwargs, {'x': 1})

    def test_apply_one_integer(self):
        def transform_function(x):
            return x + 1

        transformation = StatelessTransformation(function=transform_function)

        input = 5
        transformed = transformation.apply_one(input)
        self.assertEqual(transformed, 6)

    def test_apply_one_string(self):
        def transform_function(a_string):
            return a_string.split()

        transformation = StatelessTransformation(function=transform_function)

        input = 'Hello World'
        transformed = transformation.apply_one(input)
        self.assertListEqual(transformed, ['Hello', 'World'])

    def test_apply_one_no_side_effects(self):
        def mutation_function(a_dict):
            a_dict['new_key'] = 'new_value'
            return a_dict

        transformation = StatelessTransformation(function=mutation_function)

        input = {'old_key': 'old_value'}
        transformed = transformation.apply_one(input)
        self.assertNotEqual(input, transformed)

    def test_apply_one_kwargs(self):
        def append_to_word(word, appendix):
            return word + appendix

        transformation = StatelessTransformation(function=append_to_word, appendix='... NOT!')

        input = 'I think you look pretty'
        transformed = transformation.apply_one(input)
        self.assertEqual(transformed, 'I think you look pretty... NOT!')

    def test_apply_many_list_integer(self):
        def transform_function(x):
            return x + 1

        transformation = StatelessTransformation(function=transform_function)

        input = [1, 2, 3, 4, 5]
        expected_output = [2, 3, 4, 5, 6]
        transformed = transformation.apply_many(input)
        self.assertListEqual(transformed, expected_output)

    def test_apply_many_list_string(self):
        def transform_function(a_string):
            return a_string[::-1]

        transformation = StatelessTransformation(function=transform_function)

        input = ['Alice', 'Bob', 'Carol', 'Charlie']
        expected_output = ['ecilA', 'boB', 'loraC', 'eilrahC']

        transformed = transformation.apply_many(input)
        self.assertListEqual(transformed, expected_output)

    def test_apply_many_pandas_string_input(self):
        def transform_function(series):
            return series \
                .str.replace('A', 'B') \
                .str.replace('a', 'b')

        transformation = StatelessTransformation(function=transform_function)

        input = pd.Series(['Alice', 'Bob', 'Carol', 'Charlie'])
        expected_output = pd.Series(['Blice', 'Bob', 'Cbrol', 'Chbrlie'])

        transformed = transformation.apply_many(input)
        assert_series_equal(transformed, expected_output)

    def test_apply_many_no_side_effects(self):
        def mutation_function(a_dict):
            a_dict['new_key'] = 'new_value'
            return a_dict

        transformation = StatelessTransformation(function=mutation_function)

        input = [{'old_key': 'old_value'}, {'other_key': 'other_value'}]
        transformed = transformation.apply_many(input)
        self.assertNotEqual(input, transformed)

    def test_to_dict(self):
        def transform_function(string):
            return string.lower().replace('happy', 'sad')
        transformation = StatelessTransformation(function=transform_function)

        transformation_dict = transformation.to_dict()
        expected_dict = {'function': 'transform_function',
                         'kwargs': {}}

        self.assertDictEqual(transformation_dict, expected_dict)

    def test_to_dict_kwargs(self):
        def transform_function(string, old, new):
            return string.lower().replace(old, new)
        transformation = StatelessTransformation(function=transform_function, old='sad', new='happy')

        transformation_dict = transformation.to_dict()
        expected_dict = {'function': 'transform_function',
                         'kwargs': {'old': 'sad',
                                    'new': 'happy'}}

        self.assertDictEqual(transformation_dict, expected_dict)

    def test_repr(self):
        def transform(string, ):
            return string.lower()

        transformation = StatelessTransformation(function=transform)

        representation = transformation.__repr__()
        expected_representation = "StatelessTransformation(function=transform)"
        self.assertEqual(representation, expected_representation)

    def test_repr_kwargs(self):
        def transform_function(string, a, b):
            return string.lower().replace(a, b)

        transformation = StatelessTransformation(function=transform_function, a='sad', b='happy')

        representation = transformation.__repr__()
        expected_representation = "StatelessTransformation(function=transform_function, a='sad', b='happy')"
        self.assertEqual(representation, expected_representation)

    def test_repr_regex(self):
        def apply_regex(string, regex, rep):
            return string.replace(regex, rep)

        transformation = StatelessTransformation(function=apply_regex, regex='(\\s)+', rep=' ')

        representation = transformation.__repr__()
        expected_representation = "StatelessTransformation(function=apply_regex, regex='(\\s)+', rep=' ')"
        self.assertEqual(representation, expected_representation)


if __name__ == '__main__':
    unittest.main()
