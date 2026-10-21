import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from summarizer import clean_summary_formatting


class SummaryFormattingTests(unittest.TestCase):
    def test_duplicate_measurement(self):
        self.assertEqual(
            clean_summary_formatting("Score: 33.96 dB33.96 dB."),
            "Score: 33.96 dB.",
        )

    def test_duplicate_complexity(self):
        self.assertEqual(
            clean_summary_formatting("Complexity: O(W2H2)O(W2H2)."),
            "Complexity: O(W²H²).",
        )

    def test_pointwise_convolution(self):
        self.assertEqual(
            clean_summary_formatting("Uses 1×11×1 convolution."),
            "Uses 1×1 convolution.",
        )

    def test_depthwise_convolution(self):
        self.assertEqual(
            clean_summary_formatting("Uses 3×33×3 convolution."),
            "Uses 3×3 convolution.",
        )

    def test_duplicate_dimensions(self):
        self.assertEqual(
            clean_summary_formatting("Patch: 128×128128×128."),
            "Patch: 128×128.",
        )


if __name__ == "__main__":
    unittest.main()
