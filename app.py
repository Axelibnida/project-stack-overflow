import pickle
from flask import Flask, request, jsonify
import logging
from gensim import corpora
from gensim.models import LdaModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained models, vectorizer, dictionary, label binarizer, and topic common words
lda_model = pickle.load(open('lda_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
dictionary = pickle.load(open('dictionary.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
mlb = pickle.load(open('mlb.pkl', 'rb'))
common_words_for_topics = pickle.load(open('common_words_for_topics.pkl', 'rb'))

app = Flask(__name__)

# Download stopwords data
nltk.download('stopwords')
nltk.download('punkt')

# Define the topic tags
topic_tags = {
    0: 'html',
    1: 'image',
    2: 'request',
    3: 'string',
    4: 'android',
    5: 'java',
    6: 'data',
    7: 'file',
    8: 'table',
    9: 'application'
}

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
        topics_pred = sorted(topics_pred, key=lambda x: x[1], reverse=True)  # sort by probability
        topics_pred = [{'topic': common_words_for_topics[i], 'probability': prob} for i, prob in topics_pred]

        # Infer tags with the Random Forest model
        x = vectorizer.transform([data['text']])
        tags_pred = rf_model.predict(x)
        tags_pred = mlb.inverse_transform(tags_pred)

        # Prepare the response
        response = {
            'topics': topics_pred,
            'rf_tags': tags_pred
        }
        logger.info('Prediction successful: ' + str(response))
        return jsonify(response)

    except Exception as e:
        logger.error('An error occurred during prediction: ' + str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
