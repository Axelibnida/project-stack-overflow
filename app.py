from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    try:
        logger.info('Loading the models and dictionary...')
        # Load the LDA model, RF model, dictionary, vectorizer and label binarizer
        with open('lda_model.pkl', 'rb') as f:
            lda_model = pickle.load(f)
        with open('rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('dictionary.pkl', 'rb') as f:
            dictionary = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('mlb.pkl', 'rb') as f:
            mlb = pickle.load(f)
        logger.info('Models and dictionary loaded successfully.')

        # Get the input text from the request
        logger.info('Processing the request data...')
        data = request.get_json()
        text = data['text']

        # Preprocess the input text
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in stopwords.words('english')]

        # Create a bag-of-words representation of the input text
        bow = dictionary.doc2bow(tokens)

        # Perform LDA topic inference
        lda_topics = lda_model[bow]

        # Perform RF tag prediction
        X = vectorizer.transform([' '.join(tokens)])
        y = rf_model.predict(X)
        tags = mlb.inverse_transform(y)

        # Convert topics to a JSON serializable format
        lda_topics_serializable = [
            {'topic': topic_tags[topic[0]], 'probability': float(topic[1])}
            for topic in lda_topics
        ]
        tags_serializable = list(tags[0])

        # Return the inferred topics and predicted tags
        result = {'topics': lda_topics_serializable, 'tags': tags_serializable}
        logger.info('Inference complete. Sending response...')
        return jsonify(result)

    except Exception as e:
        logger.error('An error occurred: ' + str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
