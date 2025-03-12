import unittest
import capture_model.scoring as sc


class TestScoring(unittest.TestCase):

    def setUp(self):
        pass

    def test_models_exists(self):
        self.assertIsNotNone(sc.preprocessor)
        self.assertIsNotNone(sc.line_classifier)

    def test_predict_line(self):
        test_string = 'KYLLINGEFILET 29,95 KR'

        predicted_line = sc.predict_line(test_string)

        predicted_category = predicted_line['category_id']
        self.assertIsInstance(predicted_category, int)

        predicted_score = predicted_line['category_score']
        self.assertTrue(0 <= predicted_score <= 1)

        cleaned_line = predicted_line['clean_line']
        self.assertIsInstance(cleaned_line, str)

        line = predicted_line['line']
        self.assertEqual(test_string, line)

    def test_predict_line_empty(self):
        test_string = ''

        predicted_line = sc.predict_line(test_string)

        predicted_category = predicted_line['category_id']
        self.assertEqual(predicted_category, 0)

        predicted_score = predicted_line['category_score']
        self.assertEqual(predicted_score, 0)

        line = predicted_line['line']
        self.assertEqual(test_string, line)

        clean_line = predicted_line['clean_line']
        self.assertEqual(' ', clean_line)

    def test_predict_line_whitespace(self):
        test_string = '   '

        predicted_line = sc.predict_line(test_string)

        predicted_category = predicted_line['category_id']
        self.assertEqual(predicted_category, 0)

        predicted_score = predicted_line['category_score']
        self.assertEqual(predicted_score, 0)

        line = predicted_line['line']
        self.assertEqual(test_string, line)

        clean_line = predicted_line['clean_line']
        self.assertEqual(' ', clean_line)

    def test_predict_line_wrong_input(self):
        test_input = ['a', 'b', 'c']

        with self.assertRaises(Exception):
            sc.predict_line(test_input)

        test_input = 123

        with self.assertRaises(Exception):
            sc.predict_line(test_input)

    def test_classify_line(self):
        test_string = 'KYLLINGEFILET 29,95 KR'

        classified_line = sc.classify_line(test_string)

        predicted_category = classified_line['category_id']
        self.assertIsInstance(predicted_category, int)

        predicted_score = classified_line['category_score']
        self.assertTrue(0 <= predicted_score <= 1)

        cleaned_line = classified_line['clean_line']
        self.assertIsInstance(cleaned_line, str)

        line = classified_line['line']
        self.assertEqual(test_string, line)

        text = classified_line['text']
        self.assertEqual(text, 'KYLLINGEFILET')

        line_class = classified_line['class']
        self.assertEqual(line_class, 'PRICE')

        line_value = classified_line['value']
        self.assertEqual(line_value, '29,95')

    def test_classify_line_empty(self):
        test_string = '   '

        classified_line = sc.classify_line(test_string)

        predicted_category = classified_line['category_id']
        self.assertEqual(predicted_category, 0)

        predicted_score = classified_line['category_score']
        self.assertTrue(0 <= predicted_score <= 1)

        cleaned_line = classified_line['clean_line']
        self.assertIsInstance(cleaned_line, str)

        line = classified_line['line']
        self.assertEqual(test_string, line)

        text = classified_line['text']
        self.assertEqual(text, '   ')

        line_class = classified_line['class']
        self.assertEqual(line_class, 'TEXT')

        line_value = classified_line['value']
        self.assertEqual(line_value, '')

    def test_classify_lines(self):
        test_strings = ['KYLLINGEFILET 29,95 KR', '', '   ']

        classified_lines = sc.classify_lines(test_strings)

        for i, classified_line in enumerate(classified_lines):
            with self.subTest(i=i, classified_line=classified_line):
                predicted_category = classified_line['category_id']
                self.assertIsInstance(predicted_category, int)

                predicted_score = classified_line['category_score']
                self.assertTrue(0 <= predicted_score <= 1)

                cleaned_line = classified_line['clean_line']
                self.assertIsInstance(cleaned_line, str)

                line = classified_line['line']
                self.assertIsInstance(line, str)

                line_class = classified_line['class']
                self.assertIsInstance(line_class, str)


if __name__ == '__main__':
    unittest.main()