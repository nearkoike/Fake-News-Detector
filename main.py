import pickle
import re
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
port_stem = PorterStemmer()
vectorizer = TfidfVectorizer()
news_dataset = pd.read_csv('Cleaned_News_Dataset.csv')
model = pickle.load(open("FakeNewsmodel.pkl", 'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = vectorizer.transform(input_data)
    prediction = model.predict(vectorized_input_data)
    if (prediction[0]==1):
        pred_final = "REAL"
    elif (prediction[0]==0):
        pred_final = "FAKE"
    return pred_final

def get_proba(news):
    input_data = [news]
    vectorized_input_data = vectorizer.transform(input_data)
    probabilities = model.predict_proba(vectorized_input_data)
    return probabilities


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        stemmed_message = stemming(message)
        pred = fake_news_det(stemmed_message)
        prob = get_proba(stemmed_message)
        real_prob = prob[0][1]
        fake_prob = prob[0][0]
        real_prob = real_prob*100
        fake_prob = fake_prob*100
        real_prob = round(real_prob, 2)
        fake_prob = round(fake_prob, 2)
        return render_template('index.html', prediction=pred, prob_real=real_prob, prob_fake=fake_prob)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True, port=5001)