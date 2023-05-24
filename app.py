from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

app = Flask(__name__)

# Download stopwords data
nltk.download('stopwords')
nltk.download('punkt')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the LDA model and dictionary
    with open('lda_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)

    # Get the input text from the request
    data = request.get_json()
    text = data['text']

    # Preprocess the input text
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Create a bag-of-words representation of the input text
    bow = dictionary.doc2bow(tokens)

    # Perform topic inference
    topics = model[bow]

    # Convert topics to a JSON serializable format
    topics_serializable = [(str(topic[0]), float(topic[1])) for topic in topics]

    # Return the inferred topics
    result = {'topics': topics_serializable}
    return jsonify(result)


if __name__ == '__main__':
    app.run()
