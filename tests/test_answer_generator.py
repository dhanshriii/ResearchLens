import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from answer_generator import answer_question, build_prompt


class FakeRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.received_question = None

    def retrieve(self, question):
        self.received_question = question
        return self.passages


class AnswerGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.passage = {
            "source": "restormer.pdf",
            "page": 8,
            "chunk_id": "restormer_p08_c006",
            "text": "MDTA computes attention across feature channels.",
            "reranker_score": 2.4,
        }

    def test_prompt_includes_evidence_and_metadata(self):
        prompt = build_prompt("What does MDTA do?", [self.passage])
        self.assertIn("[Source 1]", prompt)
        self.assertIn("Paper: restormer.pdf", prompt)
        self.assertIn("Page: 8", prompt)
        self.assertIn(self.passage["text"], prompt)
        self.assertIn("Never invent a citation", prompt)

    def test_empty_question_is_rejected(self):
        retriever = FakeRetriever([self.passage])
        with self.assertRaisesRegex(ValueError, "enter a question"):
            answer_question(retriever, "   ")

    def test_no_retrieved_passages_returns_no_sources(self):
        retriever = FakeRetriever([])
        result = answer_question(retriever, "What does MDTA do?")
        self.assertEqual(result["sources"], [])
        self.assertIn("couldn't retrieve", result["answer"])

    @patch(
        "answer_generator.call_gemini",
        return_value="MDTA uses channel attention [Source 1, p. 8].",
    )
    def test_answer_preserves_source_and_evidence(self, mocked_gemini):
        retriever = FakeRetriever([self.passage])
        result = answer_question(retriever, "What does MDTA do?")

        self.assertEqual(
            result["answer"],
            "MDTA uses channel attention [Source 1, p. 8].",
        )
        self.assertEqual(len(result["sources"]), 1)
        source = result["sources"][0]
        self.assertEqual(source["source"], "restormer.pdf")
        self.assertEqual(source["page"], 8)
        self.assertEqual(source["chunk_id"], "restormer_p08_c006")
        self.assertEqual(source["text"], self.passage["text"])
        self.assertEqual(source["reranker_score"], 2.4)
        mocked_gemini.assert_called_once()

    @patch("answer_generator.call_gemini", return_value="Supported answer.")
    def test_question_is_trimmed_before_retrieval(self, _mocked_gemini):
        retriever = FakeRetriever([self.passage])
        answer_question(retriever, "  What does MDTA do?  ")
        self.assertEqual(retriever.received_question, "What does MDTA do?")


if __name__ == "__main__":
    unittest.main()
