import unittest
import numpy as np
from src.metrics import calibration, from_probabilities, summarize_corruptions


class MetricsTest(unittest.TestCase):
    def test_perfect(self):
        for scheme in ("equal_mass", "equal_width"):
            self.assertEqual(calibration([1, 1], [1, 1], scheme=scheme)["ece"], 0)

    def test_confident_wrong(self):
        self.assertEqual(calibration([1, 1], [0, 0])["ece"], 1)

    def test_calibrated_single_bin(self):
        self.assertAlmostEqual(calibration([0.75] * 4, [1, 1, 1, 0], n_bins=1)["ece"], 0)

    def test_binning_changes_answer(self):
        c, y = [0.1, 0.2, 0.3, 0.9], [0, 0, 1, 1]
        self.assertAlmostEqual(calibration(c, y, 2, "equal_mass")["ece"], 0.275)
        self.assertAlmostEqual(calibration(c, y, 2, "equal_width")["ece"], 0.125)

    def test_right_closed_boundaries(self):
        r = calibration([0, 0.5, 1], [0, 0, 1], 2, "equal_width")
        self.assertEqual([x["count"] for x in r["bins"]], [2, 1])

    def test_ties_and_empty_bins(self):
        r = calibration([0.8] * 4, [1, 0, 1, 1], 10)
        self.assertEqual(sum(x["count"] for x in r["bins"]), 4)
        self.assertEqual(len(r["bins"]), 10)

    def test_probabilities(self):
        r = from_probabilities([[0.9, 0.1], [0.4, 0.6]], [0, 0])
        self.assertEqual(r["accuracy"], 0.5)
        self.assertAlmostEqual(r["equal_mass"]["ece"], 0.35)

    def test_invalid_input(self):
        for c in ([float("nan")], [1.1], [-0.1], []):
            with self.assertRaises(ValueError): calibration(c, [1] * len(c))
        with self.assertRaises(ValueError): from_probabilities([[0.1, 0.1]], [0])

    def test_corruption_summary(self):
        rows = [from_probabilities([[0.9, 0.1]], [0]),
                from_probabilities([[0.9, 0.1]], [1])]
        r = summarize_corruptions(rows)
        self.assertEqual(r["mpc"], 0.5)
        self.assertAlmostEqual(r["mean_corruption_ece_equal_mass"], 0.5)


if __name__ == "__main__": unittest.main()
