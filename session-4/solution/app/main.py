import logging
import pathlib

import torch
from flask import Flask, render_template, request
from torchtext.data.utils import get_tokenizer, ngrams_iterator

from model import SentimentAnalysis


VOCAB = None
MODEL = None
NGRAMS = None
TOKENIZER = None

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

@app.before_first_request
def _load_model():
    # First load into memory the variables that we will need to predict
    checkpoint_path = pathlib.Path(__file__).parent.absolute() / "state_dict.pt"
    checkpoint = torch.load(checkpoint_path)

    global VOCAB, MODEL, NGRAMS, TOKENIZER
    VOCAB = checkpoint["vocab"]
    MODEL = SentimentAnalysis(len(VOCAB), checkpoint["embed_dim"], checkpoint["num_class"]).eval()
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    NGRAMS = checkpoint["ngrams"]
    TOKENIZER = get_tokenizer("basic_english")


# Disable gradients
@torch.no_grad()
def predict_review_sentiment(text):
    # Convert text to tensor
    text = torch.tensor(
        [VOCAB[token] for token in ngrams_iterator(TOKENIZER(text), NGRAMS)]
    )

    # Compute output
    output = MODEL(text, torch.tensor([0]))
    confidences = torch.softmax(output, dim=1)
    return confidences.squeeze()[
        1
    ].item()  # Class 1 corresponds to confidence of positive


@app.route("/predict", methods=["POST"])
def predict():
    """The input parameter is `review`"""
    review = request.form["review"]
    print(f"Prediction for review:\n {review}")

    result = predict_review_sentiment(review)
    return render_template("result.html", result=result)


@app.route("/", methods=["GET"])
def hello():
    """ Return an HTML. """
    return render_template("hello.html")


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == "__main__":
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host="localhost", port=8080, debug=True)
