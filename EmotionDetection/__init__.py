"""EmotionDetection package."""

# The package name is fixed by the course assignment and intentionally uses
# the original mixed-case spelling.
# pylint: disable=invalid-name

from .emotion_detection import emotion_detector

__all__ = ["emotion_detector"]
