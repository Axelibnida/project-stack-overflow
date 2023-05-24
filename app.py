from flask import Flask, request, jsonify
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load the saved model and dictionary
lda_model = gensim.models.LdaModel.load('lda_model')
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
    return result

def predict_topic(text):
    # Preprocess the input text
    processed_text = preprocess(text)

    # Convert text to bag-of-words format
    bow_text = dictionary.doc2bow(processed_text)

    # Get the topic distribution of the text
    topic_distribution = lda_model.get_document_topics(bow_text)

    return topic_distribution

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    result = predict_topic(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run()
