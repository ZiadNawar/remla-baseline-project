"""
Flask API of the Multilabel classification on Stack Overflow tags .
"""
import joblib
from flasgger import Swagger
from flask import Flask, jsonify, request
from scipy import sparse as sp_sparse

from src.text_preprocessing import text_prepare                       # pylint: disable=import-error
from src.vectorization import my_bag_of_words, create_words_to_index  # pylint: disable=import-error

app = Flask(__name__)
swagger = Swagger(app)

output_directory = "output"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Class of the input input_data.
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
            required: input_data
            properties:
                input_data:
                    type: string
                    example: This is an example of an input data.
    responses:
      200:
        description: "The result of the classification: One of the model classes ."
    """
    # Prepare the input data
    input_data = request.get_json()
    input_data = input_data.get('input_data')

    return jsonify(predict_instance(input_data))


def predict_instance(input_data):
    """
    Generates a multi-label prediction for the given instance.
    :param input_data: string; the instance to be labeled
    :return: a json representation of the result
    """
    # Load the classifiers
    classifier_mybag, classifier_tfidf = joblib.load(output_directory + '/classifiers.joblib')
    mlb = joblib.load(output_directory + "/mlb.joblib")

    # Predict using bag-of-words
    words_counts = joblib.load(output_directory + "/words_counts.joblib")

    # Predict using tf-idf
    tfidf_vectorizer = joblib.load(output_directory + '/tfidf_vectorizer.joblib')

    vectorized_mybag, vectorized_tfidf = vectorize_instance(input_data, words_counts, tfidf_vectorizer)

    encoded_result_mybag = classifier_mybag.predict(vectorized_mybag)
    prediction_mybag = list(mlb.inverse_transform(encoded_result_mybag)[0])

    encoded_result_tfidf = classifier_tfidf.predict(vectorized_tfidf)
    prediction_tfidf = list(mlb.inverse_transform(encoded_result_tfidf)[0])

    # Print predictions to console for debugging purposes
    print(f"mybag prediction: {prediction_mybag}")
    print(f"tf-idf prediction: {prediction_tfidf}")

    result = "Not Java"
    if "java" in prediction_tfidf or "java" in prediction_mybag:
        result = "Java"

    res = {
        "result": result,
        "result_mybag": prediction_mybag,
        "result_tfidf": prediction_tfidf,
        "classifier": "bag-of-words and tf-idf",
        "input_data": input_data
    }
    print(res)
    return res


def vectorize_instance(input_data, words_counts, tfidf_vectorizer):
    """
    Vectorizes the instance to bag-of-words and tf-idf.
    :param input_data: string that needs to be embedded
    :param words_counts: the word counts for bag-of-words
    :param tfidf_vectorizer: TfidfVectorizer object with a fixed vocabulary
    :return: vectorized representation of input_data in bag-of-words format,
                vectorized representation of input_data in tf-idf format
    """
    processed = text_prepare(input_data)

    DICT_SIZE, WORDS_TO_INDEX = create_words_to_index(words_counts)
    vectorized_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(processed, WORDS_TO_INDEX, DICT_SIZE))])

    vectorized_tfidf = tfidf_vectorizer.transform([processed])

    return vectorized_mybag, vectorized_tfidf


if __name__ == '__main__':
    clf = joblib.load('output/classifiers.joblib')
    app.run(host="0.0.0.0", port=8081, debug=True)
