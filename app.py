import numpy as np
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


@app.route('/', methods=['GET', 'POST'])
def index():

    # 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.debug("model was loaded")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)