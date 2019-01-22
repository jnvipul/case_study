__author__ = "Vipul J"

__version__ = "0.0.1"
__maintainer__ = "Vipul J"
__email__ = "messagevipul@gmail.com"


# imports
from flask import Flask, request, abort
import pandas as pd
from flask_cors import CORS
from predictor import read_model, read_precomputes, predict

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Welcome to Renthop home"


@app.route("/predict_interest", methods=['POST'])
def predict_interest():
    if not request.json:
        abort(400)
    df = pd.DataFrame(request.json)
    result_df = predict(df)
    return result_df.to_json(orient='records')


def setup():
    read_model()
    read_precomputes()
    return None


if __name__ == '__main__':
    setup()
    app.run(host='0.0.0.0', port=4000, debug=True)
