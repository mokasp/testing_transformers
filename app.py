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

    sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
    ]

    # 2. Calculate embeddings by calling model.encode()
    embeddings = model.encode(sentences)
    logging.debug(str(embeddings.shape))
    # [3, 384]

    # 3. Calculate the embedding similarities
    similarities = model.similarity(embeddings, embeddings)
    logging.debug(str(similarities))
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)