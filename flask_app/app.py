__author__ = "Vipul J"

__version__ = "0.0.1"
__maintainer__ = "Vipul J"
__email__ = "messagevipul@gmail.com"


# imports
from flask import Flask
import os
import collections
import smart_open
import random
import pandas as pd
import numpy as np
from pprint import pprint
import string
from flask_cors import CORS
from predictor import read_model, read_precomputes, predict


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Welcome to Renthop home"


def setup():
    read_model()
    read_precomputes()
    print(predict(pd.read_csv('data/sample_input.csv')))


if __name__ == '__main__':
    setup()
    app.run(debug=True)
