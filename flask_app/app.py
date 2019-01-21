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
import json
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Welcome to Renthop home"


if __name__ == '__main__':
    app.run(debug=True)
