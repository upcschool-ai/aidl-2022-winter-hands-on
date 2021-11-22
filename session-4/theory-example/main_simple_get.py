from flask import Flask

app = Flask(__name__)

@app.route("/hello", methods=["GET"])
def hello():
    """ Return a friendly HTTP greeting. """
    return "Hello AIDL!\n"

app.run(host="localhost", port=8080, debug=True)
