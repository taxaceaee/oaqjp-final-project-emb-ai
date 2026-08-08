"""Watson NLP emotion detection client."""

from typing import Any, Dict, Optional

import requests

WATSON_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_HEADER = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
}
EMOTION_NAMES = ("anger", "disgust", "fear", "joy", "sadness")


def _empty_result() -> Dict[str, Optional[float]]:
    """Return the standard empty result used for invalid input or API errors."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def emotion_detector(text_to_analyse: str) -> Dict[str, Any]:
    """Analyze text and return emotion scores plus the dominant emotion.

    The Watson service returns HTTP 400 for invalid or blank input. Returning a
    consistent empty structure lets the Flask layer present a friendly error
    without exposing a remote-service traceback.
    """
    if not isinstance(text_to_analyse, str) or not text_to_analyse.strip():
        return _empty_result()

    payload = {"raw_document": {"text": text_to_analyse}}

    try:
        response = requests.post(
            WATSON_EMOTION_URL,
            json=payload,
            headers=MODEL_HEADER,
            timeout=15,
        )
    except requests.RequestException:
        return _empty_result()

    if response.status_code == 400:
        return _empty_result()
    if response.status_code != 200:
        return _empty_result()

    try:
        emotion_scores = response.json()["emotionPredictions"][0]["emotion"]
        result = {name: emotion_scores[name] for name in EMOTION_NAMES}
        result["dominant_emotion"] = max(
            EMOTION_NAMES, key=lambda emotion: result[emotion]
        )
        return result
    except (KeyError, IndexError, TypeError, ValueError):
        return _empty_result()
