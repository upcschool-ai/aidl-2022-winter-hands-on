from flask import Flask, request

app = Flask(__name__)

@app.route("/hello", methods=["GET"])
def hello():
    """ Return a friendly HTTP greeting. """
    return "Hello AIDL!\n"


@app.route("/sum", methods=["POST"])
def post_example():
    """ Example of POST method. The input parameters are x and y"""
    x = float(request.form["x"])
    y = float(request.form["y"])
    result = x + y
    return f"The result is: {result}\n"


app.run(host="localhost", port=8080, debug=True)
