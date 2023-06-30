from flask import Flask, request, jsonify
from gensim import corpora
from gensim.models import LdaModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
import nltk
import gensim
import numpy as np
import logging
import pickle

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce logging level to WARNING
logger = logging.getLogger(__name__)

# Load the pre-trained models, vectorizer, dictionary, label binarizer, and topic common words
lda_model = pickle.load(open('lda_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
dictionary = pickle.load(open('dictionary.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
mlb = pickle.load(open('mlb.pkl', 'rb'))
common_words_for_topics = pickle.load(open('common_words_for_topics.pkl', 'rb'))

def decode_tags(tags_pred, classes):
    decoded_tags = []
    for tags in tags_pred:
        decoded_tags.append([classes[i] for i, tag in enumerate(tags) if tag == 1])
    return decoded_tags

app = Flask(__name__)

# Download stopwords data
nltk.download('stopwords')
nltk.download('punkt')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    logger.info('Data received: ' + str(data))

    try:
        # Preprocess the data
        stop_words = set(stopwords.words('english'))

        def preprocess(text):
            result = []
            for token in gensim.utils.simple_preprocess(text):
                if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
                    result.append(token)
            return result

        processed_data = preprocess(data['text'])
        corpus = dictionary.doc2bow(processed_data)

        # Infer topics with the LDA model
        topics_pred = lda_model[corpus]
        topic_pred = max(topics_pred, key=lambda x: x[1])  # Get the most probable topic
        words_for_topic = lda_model.show_topic(topic_pred[0], topn=5)
        topic_pred = {'topic': [word for word, _ in words_for_topic], 'probability': float(topic_pred[1])}

        # Infer tags with the Random Forest model
        x = vectorizer.transform([data['text']])
        tags_pred = rf_model.predict(x)
        tags_pred = decode_tags(tags_pred, mlb.classes_)

        # Prepare the response
        response = {
            'topics': topic_pred,
            'rf_tags': tags_pred
        }
        logger.info('Prediction successful: ' + str(response))
        return jsonify(response)

    except Exception as e:
        logger.error('An error occurred during prediction: ' + str(response))
        return jsonify(response)

if __name__ == '__main__':
    app.run()
