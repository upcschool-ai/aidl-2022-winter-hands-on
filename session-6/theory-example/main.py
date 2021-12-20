from flask import Flask, request

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    """ Return a friendly greeting. """
    return "This is my app inside a Docker container!\n"


@app.route("/predict", methods=["POST"])
def post_example():
    """ Example of POST method. The input parameters are x and y"""
    x = float(request.form["x"])
    y = float(request.form["y"])
    result = x + y
    return f"The result is: {result}\n"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    