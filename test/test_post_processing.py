import unittest
from capture_model import post_processing as postproc


class TestPostProcessing(unittest.TestCase):

    def test_valid_cvr_string(self):
        self.assertRaises(Exception, postproc.valid_cvr('37639826'))

    def test_valid_cvr_short(self):
        self.assertFalse(postproc.valid_cvr(3763982))

    def test_valid_cvr_phone_number(self):
        self.assertFalse(postproc.valid_cvr(29249089))

    def test_valid_cvr_with_valid_cvrs(self):
        self.assertTrue(postproc.valid_cvr(37639826))
        self.assertTrue(postproc.valid_cvr(66137112))
        self.assertTrue(postproc.valid_cvr(37639796))
        self.assertTrue(postproc.valid_cvr(37636746))
        self.assertTrue(postproc.valid_cvr(37887897))
        self.assertTrue(postproc.valid_cvr(38280589))
        self.assertTrue(postproc.valid_cvr(36967862))
        self.assertTrue(postproc.valid_cvr(37641251))
        self.assertTrue(postproc.valid_cvr(73316618))
        self.assertTrue(postproc.valid_cvr(38168703))


if __name__ == '__main__':
    unittest.main()