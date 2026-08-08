"""Unit tests for the Watson emotion detector client."""

import json
import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


def watson_response(scores, status_code=200):
    """Build a small mock matching the Watson response object."""
    response = Mock()
    response.status_code = status_code
    response.text = json.dumps({
        "emotionPredictions": [{"emotion": scores}],
    })
    response.json.return_value = {
        "emotionPredictions": [{"emotion": scores}],
    }
    return response


class EmotionDetectionTests(unittest.TestCase):
    """Verify each expected dominant emotion and invalid-input behavior."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy_dominates(self, post):
        """Return joy when joy has the highest score."""
        post.return_value = watson_response(
            {
                "anger": 0.01,
                "disgust": 0.01,
                "fear": 0.01,
                "joy": 0.95,
                "sadness": 0.02,
            }
        )
        result = emotion_detector("I am very happy today")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger_dominates(self, post):
        """Return anger when anger has the highest score."""
        post.return_value = watson_response(
            {
                "anger": 0.92,
                "disgust": 0.01,
                "fear": 0.02,
                "joy": 0.02,
                "sadness": 0.03,
            }
        )
        result = emotion_detector("I am furious")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_disgust_dominates(self, post):
        """Return disgust when disgust has the highest score."""
        post.return_value = watson_response(
            {
                "anger": 0.01,
                "disgust": 0.94,
                "fear": 0.01,
                "joy": 0.01,
                "sadness": 0.03,
            }
        )
        result = emotion_detector("That is disgusting")
        self.assertEqual(result["dominant_emotion"], "disgust")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_fear_dominates(self, post):
        """Return fear when fear has the highest score."""
        post.return_value = watson_response(
            {
                "anger": 0.01,
                "disgust": 0.01,
                "fear": 0.94,
                "joy": 0.01,
                "sadness": 0.03,
            }
        )
        result = emotion_detector("I am terrified")
        self.assertEqual(result["dominant_emotion"], "fear")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness_dominates(self, post):
        """Return sadness when sadness has the highest score."""
        post.return_value = watson_response(
            {
                "anger": 0.01,
                "disgust": 0.01,
                "fear": 0.03,
                "joy": 0.01,
                "sadness": 0.94,
            }
        )
        result = emotion_detector("I feel sad")
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_blank_input_returns_empty_result(self):
        """Return an empty result without making a remote request."""
        result = emotion_detector("   ")
        self.assertIsNone(result["dominant_emotion"])

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_bad_request_returns_empty_result(self, post):
        """Return an empty result when Watson responds with HTTP 400."""
        post.return_value = watson_response({}, status_code=400)
        result = emotion_detector("invalid remote request")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
