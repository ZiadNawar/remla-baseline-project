from src.text_preprocessing import text_prepare
from src.vectorization import my_bag_of_words
from src.serve_model import predict_instance


def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function",
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


def test_instance_prediction():
    answers = [
        {'result': 'Not Java', 'result_mybag': ['ruby-on-rails'], 'result_tfidf': ['json', 'ruby-on-rails'], 'classifier': 'bag-of-words and tf-idf', 'input_data': 'Content-Type "application/json" not required in rails'},
        {'result': 'Not Java', 'result_mybag': ['ruby', 'session'], 'result_tfidf': ['ruby'], 'classifier': 'bag-of-words and tf-idf', 'input_data': 'Sessions in Sinatra: Used to Pass Variable'},
        {'result': 'Not Java', 'result_mybag': ['json', 'ruby-on-rails'], 'result_tfidf': ['json', 'ruby-on-rails'], 'classifier': 'bag-of-words and tf-idf', 'input_data': 'Getting error - type "json" does not exist - in Postgresql during rake db migrate'}
    ]
    for i, xi in enumerate(['Content-Type "application/json" not required in rails',
                            'Sessions in Sinatra: Used to Pass Variable',
                            'Getting error - type "json" does not exist - in Postgresql during rake db migrate']):
        assert predict_instance(xi) == answers[i]
