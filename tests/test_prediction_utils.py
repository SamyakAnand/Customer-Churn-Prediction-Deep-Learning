import math
import unittest

from prediction_utils import extract_single_probability, normalize_probability


class NormalizeProbabilityTests(unittest.TestCase):
    def test_accepts_valid_probability_values(self):
        for value, expected in ((0, 0.0), (0.5, 0.5), (1, 1.0), ("0.75", 0.75)):
            with self.subTest(value=value):
                self.assertEqual(normalize_probability(value), expected)

    def test_rejects_non_numeric_values(self):
        for value in ("not-a-number", None, object(), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_probability(value)

    def test_rejects_non_finite_values(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_probability(value)

    def test_rejects_values_outside_probability_range(self):
        for value in (-0.001, 1.001):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_probability(value)


class ExtractSingleProbabilityTests(unittest.TestCase):
    def test_extracts_one_probability(self):
        self.assertEqual(extract_single_probability([[0.625]]), 0.625)

    def test_rejects_empty_or_multiple_prediction_rows(self):
        for prediction in ([], [[0.2], [0.8]]):
            with self.subTest(prediction=prediction):
                with self.assertRaises(ValueError):
                    extract_single_probability(prediction)

    def test_rejects_wrong_output_shape(self):
        for prediction in ([0.5], [[0.2, 0.8]], [[0.2], [0.8]]):
            with self.subTest(prediction=prediction):
                with self.assertRaises(ValueError):
                    extract_single_probability(prediction)

    def test_rejects_invalid_probability_inside_valid_shape(self):
        for value in (math.nan, math.inf, -0.1, 1.1, "invalid"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    extract_single_probability([[value]])


if __name__ == "__main__":
    unittest.main()
