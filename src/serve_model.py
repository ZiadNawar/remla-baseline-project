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

output_directory = "../output"


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
    print(f"input data: {input_data}")
    processed = text_prepare(input_data)
    print(f"processed data: {processed}")

    # Load the classifiers
    classifier_mybag, classifier_tfidf = joblib.load(output_directory + '/classifiers.joblib')
    mlb = joblib.load(output_directory + "/mlb.joblib")

    # Predict using bag-of-words
    words_counts = joblib.load(output_directory + "/words_counts.joblib")

    DICT_SIZE, WORDS_TO_INDEX = create_words_to_index(words_counts)
    vectorized_mybag = sp_sparse.vstack(
        [sp_sparse.csr_matrix(my_bag_of_words(processed, WORDS_TO_INDEX, DICT_SIZE))])

    encoded_result_mybag = classifier_mybag.predict(vectorized_mybag)
    prediction_mybag = list(mlb.inverse_transform(encoded_result_mybag)[0])


    # Predict using tf-idf
    tfidf_vocab = joblib.load(output_directory + '/tfidf_vocabulary.joblib')
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       token_pattern='(\S+)', vocabulary=tfidf_vocab)
    tf_idf_vectorized = tfidf_vectorizer.fit_transform([processed])
    encoded_result_tfidf = classifier_tfidf.predict(tf_idf_vectorized)
    prediction_tfidf = list(mlb.inverse_transform(encoded_result_tfidf)[0])

    # Print predictions to console for debugging purposes
    print(f"mybag prediction: {prediction_mybag}")
    print(f"tf-idf prediction: {prediction_tfidf}")
    
    res = {
        "result": "Java",
        "result_mybag": prediction_mybag,
        "result_tfidf": prediction_tfidf,
        "classifier": "bag-of-words and tf-idf",
        "input_data": input_data
    }
    print(res)
    return jsonify(res)


if __name__ == '__main__':
    clf = joblib.load('output/classifiers.joblib')
    app.run(host="0.0.0.0", port=8081, debug=True)
