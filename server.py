"""Flask deployment for the Emotion Detector application."""

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


def analyze_text(text: str):
    """Return an emotion result or a user-facing error message."""
    if not text or not text.strip():
        return None, "Invalid input! Try again."

    result = emotion_detector(text)
    if result["dominant_emotion"] is None:
        return None, "The emotion service could not analyze that statement."
    return result, None


@app.route("/")
def home():
    """Render the web interface and optionally analyze its query string."""
    has_text_parameter = "text" in request.args
    text = request.args.get("text", "")
    result = None
    error = None
    if has_text_parameter:
        result, error = analyze_text(text)
    return render_template("index.html", text=text, result=result, error=error)


@app.route("/emotionDetector")
def emotion_detector_api():
    """Return the formatted emotion response expected by the lab client."""
    text_to_analyse = request.args.get("textToAnalyze")
    if text_to_analyse is None:
        text_to_analyse = request.args.get("text", "")
    if not text_to_analyse.strip():
        return "Invalid input! Try again."

    result, error = analyze_text(text_to_analyse)
    if error:
        return error

    return (
        "For the given statement, the system response is "
        f"'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, "
        f"'joy': {result['joy']}, "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
