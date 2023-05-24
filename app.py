from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import LdaModel
from gensim.corpora import Dictionary

app = Flask(__name__)

# Download stopwords data
nltk.download('stopwords')


@app.route('/', methods=['POST'])
def predict():
    # Load the LDA model and dictionary
    model = LdaModel.load('model')
    dictionary = Dictionary.load('dictionary')

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

    # Return the inferred topics
    result = {'topics': topics}
    return jsonify(result)


if __name__ == '__main__':
    app.run()

