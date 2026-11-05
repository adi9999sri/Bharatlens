
import unittest
from src.preprocess import preprocess_text


class TestTeluguPreprocessing(unittest.TestCase):

    def test_extra_spaces(self):
        text = "  తెలంగాణలో    భారీ వర్షాలు  "
        expected = "తెలంగాణలో భారీ వర్షాలు"
        self.assertEqual(preprocess_text(text), expected)

    def test_newlines(self):
        text = "రైతులకు\nకొత్త పథకం"
        expected = "రైతులకు కొత్త పథకం"
        self.assertEqual(preprocess_text(text), expected)

    def test_tabs(self):
        text = "ప్రభుత్వ\tపాఠశాలలు"
        expected = "ప్రభుత్వ పాఠశాలలు"
        self.assertEqual(preprocess_text(text), expected)

    def test_punctuation(self):
        text = "వర్షాలు!!!"
        self.assertEqual(preprocess_text(text), text)

    def test_empty_text(self):
        self.assertEqual(preprocess_text("   "), "")

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            preprocess_text(123)


if __name__ == "__main__":
    unittest.main()
