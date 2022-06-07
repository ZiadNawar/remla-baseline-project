"""
Flask API of the Multilabel classification on Stack Overflow tags .
"""
import joblib
from flasgger import Swagger
from flask import Flask, jsonify, request
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from text_preprocessing import text_prepare
from vectorization import my_bag_of_words, create_words_to_index

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Class of the input test_data.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: test_data
            properties:
                test_data:
                    type: string
                    example: This is an example of an input data.
    responses:
      200:
        description: "The result of the classification: One of the model classes ."
    """
    input_data = request.get_json()
    test_data = input_data.get('test_data')
    processed = text_prepare(test_data)
    words_counts = joblib.load("output" + "/words_counts.joblib")

    DICT_SIZE, WORDS_TO_INDEX = create_words_to_index(words_counts)
    vectorized_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in processed])
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       token_pattern='(\S+)')

    vectorized_tfidf = tfidf_vectorizer.fit_transform(processed)
    classifier_mybag, classifier_tfidf = joblib.load('output/classifiers.joblib')
    prediction = classifier_mybag.predict(vectorized_mybag)
    prediction_tfidf = classifier_tfidf.predict(vectorized_tfidf)
    print(prediction)
    print(prediction_tfidf)
    res = {
        "result_mybag": prediction,
        "result_tfidf": prediction_tfidf,
        "classifier": "Logisitc Regression",
        "input data": test_data
    }
    print(res)
    return jsonify(res)


if __name__ == '__main__':
    clf = joblib.load('output/classifiers.joblib')
    app.run(host="0.0.0.0", port=8081, debug=True)
