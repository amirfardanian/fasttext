import unittest
import pandas as pd
from capture_model import scoring as sf


class TestRegressionScoring(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('fixtures/regression_data.csv', encoding='utf-8')
        df = df[df['gold_class']]
        df['line'].fillna('', inplace=True)

        self.regression_df = df

    def test_regression_one(self):
        for i, row in self.regression_df.iterrows():
            with self.subTest(i=i, row=row):
                classified_line = sf.classify_line(row['line'])
                self.assertEqual(classified_line['class'], row['line_class'])

    def test_regression_many(self):
        regression_lines = self.regression_df['line'].tolist()

        classified_lines = sf.classify_lines(regression_lines)

        for row, classified_line in zip(self.regression_df.iterrows(),
                                        classified_lines):
            i = row[0]
            row = row[1]
            with self.subTest(i=i, row=row):
                self.assertEqual(classified_line['class'], row['line_class'])


if __name__ == '__main__':
    unittest.main()
