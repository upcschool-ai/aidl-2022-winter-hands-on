from flask import Flask, request

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
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
    return f"The result is: {result}"


@app.route("/html-example", methods=["GET"])
def html_example():
    """ Return an HTML. """
    return """
<html>
    <head>
        <title>HTML example</title>
        <style>
            table, th, td {
                border: 1px solid black;
            }
        </style>
    </head>
    <body>
        <h1>Hello AIDL!</h1>
        This is an example of a table in an HTML:
        <table>
            <tr>
                <th>Firstname</th>
                <th>Lastname</th>
                <th>Age</th>
            </tr>
            <tr>
                <td>Jill</td>
                <td>Smith</td>
                <td>50</td>
            </tr>
            <tr>
                <td>Eve</td>
                <td>Jackson</td>
                <td>94</td>
            </tr>
        </table>
    </body>
</html>
"""


if __name__ == "__main__":
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host="localhost", port=8080, debug=True)
    