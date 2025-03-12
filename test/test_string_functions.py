import unittest
import pandas as pd
import capture_model.modelling.string_functions as sf
from pandas.util.testing import assert_series_equal


class TestStringFunctions(unittest.TestCase):

    def test_lower(self):
        test_string = 'HELLO'
        test_series = pd.Series(['HELLO', 'WORLD'])

        result_string = sf.lower(test_string)
        self.assertEqual(result_string, 'hello')

        result_series = sf.lower(test_series)
        assert_series_equal(result_series, pd.Series(['hello', 'world']))

    def test_lower_nonstring(self):
        test_string = 1

        with self.assertRaises(Exception):
            sf.lower(test_string)

    def test_upper(self):
        test_string = 'hello'
        test_series = pd.Series(['hello', 'world'])

        result_string = sf.upper(test_string)
        self.assertEqual(result_string, 'HELLO')

        result_series = sf.upper(test_series)
        assert_series_equal(result_series, pd.Series(['HELLO', 'WORLD']))

    def test_upper_nonstring(self):
        test_string = 1

        with self.assertRaises(Exception):
            sf.upper(test_string)

    def test_replace(self):
        test_string = 'brown cow'
        test_series = pd.Series(['brown cow', 'brown chicken'])

        result_string = sf.replace(test_string, 'brown', 'orange')
        self.assertEqual(result_string, 'orange cow')

        result_series = sf.replace(test_series, 'brown', 'orange')
        assert_series_equal(result_series, pd.Series(['orange cow', 'orange chicken']))

    def test_replace_regex(self):
        test_string = 'brown     cow'
        test_series = pd.Series(['brown     cow', 'brown     chicken'])

        result_string = sf.replace(test_string, r'(\s)+', ' ')
        self.assertEqual(result_string, 'brown cow')

        result_series = sf.replace(test_series, r'(\s)+', ' ')
        assert_series_equal(result_series, pd.Series(['brown cow', 'brown chicken']))

    def test_replace_nonstring(self):
        test_string = 1

        with self.assertRaises(Exception):
            sf.replace(test_string, 'brown', 'orange')

    def test_ljust(self):
        test_string = 'brown cow'
        test_series = pd.Series(['brown cow', 'brown chicken'])

        result_string = sf.ljust(test_string, width=12, fillchar='0')
        self.assertEqual(result_string, 'brown cow000')

        result_series = sf.ljust(test_series, width=12, fillchar='0')
        assert_series_equal(result_series, pd.Series(['brown cow000', 'brown chicken']))

    def test_ljust_empty(self):
        test_string = ''
        test_series = pd.Series(['', 'brown chicken'])

        result_string = sf.ljust(test_string, width=1, fillchar=' ')
        self.assertEqual(result_string, ' ')

        result_series = sf.ljust(test_series, width=1, fillchar=' ')
        assert_series_equal(result_series, pd.Series([' ', 'brown chicken']))

    def test_lstrip(self):
        test_string = '  hello  '
        test_series = pd.Series(['  hello  ', '  world  '])

        result_string = sf.lstrip(test_string)
        self.assertEqual(result_string, 'hello  ')

        result_series = sf.lstrip(test_series)
        assert_series_equal(result_series, pd.Series(['hello  ', 'world  ']))

    def test_lstrip_nonstring(self):
        test_string = 1

        with self.assertRaises(Exception):
            sf.lstrip(test_string)

    def test_rstrip(self):
        test_string = '  hello  '
        test_series = pd.Series(['  hello  ', '  world  '])

        result_string = sf.rstrip(test_string)
        self.assertEqual(result_string, '  hello')

        result_series = sf.rstrip(test_series)
        assert_series_equal(result_series, pd.Series(['  hello', '  world']))

    def test_rstrip_nonstring(self):
        test_string = 1

        with self.assertRaises(Exception):
            sf.rstrip(test_string)

    def test_strip(self):
        test_string = '  hello  '
        test_series = pd.Series(['  hello  ', '  world  '])

        result_string = sf.strip(test_string)
        self.assertEqual(result_string, 'hello')

        result_series = sf.strip(test_series)
        assert_series_equal(result_series, pd.Series(['hello', 'world']))

    def test_strip_nonstring(self):
        test_string = 1

        with self.assertRaises(Exception):
            sf.strip(test_string)

    def test_split(self):
        test_string = 'one,two'
        test_series = pd.Series(['one,two', 'three,four'])

        result_list = sf.split(test_string, ',')
        self.assertListEqual(result_list, ['one', 'two'])

        result_series = sf.split(test_series, ',')
        assert_series_equal(result_series, pd.Series([['one', 'two'], ['three', 'four']]))

    def test_split_nonstring(self):
        test_string = 1

        with self.assertRaises(Exception):
            sf.split(test_string, '')


if __name__ == '__main__':
    unittest.main()