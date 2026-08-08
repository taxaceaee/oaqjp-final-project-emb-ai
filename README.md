# Final Project - Emotion Detector

This is the Final Project - Emotion Detector application.

Emotion Detector is a Flask web application that analyzes a sentence with the IBM Watson Natural Language Processing emotion model. It returns the five emotion scores—anger, disgust, fear, joy, and sadness—along with the dominant emotion.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:5000/` and enter a sentence. The JSON endpoint is available at `/emotionDetector?text=I%20love%20this`.

## Validation

```bash
python -m unittest discover -s tests -v
pylint server.py EmotionDetection/emotion_detection.py EmotionDetection/__init__.py tests/test_emotion_detection.py
```
