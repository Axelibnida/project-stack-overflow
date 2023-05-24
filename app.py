from flask import Flask, request, jsonify
import pandas as pd
import gensim
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

app = Flask(__name__)

# Load the data and preprocess
data = pd.read_csv('your_data.csv')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [lemmatizer.lemmatize(t) for t in tokenized_text if t not in stop_words and t.isalpha()]
    return cleaned_text

data['tokens'] = data['text'].apply(preprocess)
data['tags'] = data['tags'].str.split(',')

# Load the LDA model and dictionary
with open('lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

with open('dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to my LDA model API. Use /predict to access the model."

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        text = data['text']
        processed_text = preprocess(text)
        bow_vector = dictionary.doc2bow(processed_text)
        lda_vector = lda_model[bow_vector]
        sorted_topics = sorted(lda_vector, key=lambda x: x[1], reverse=True)
        tags = [dictionary[id] for id, _ in sorted_topics[:2]]
        return jsonify(tags)

if __name__ == '__main__':
    app.run(port=5000)
