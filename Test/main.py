# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:47:07 2024

@author: Vogie
"""
import pandas as pd
import pickle
import re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
vectorizer = TfidfVectorizer()
le = LabelEncoder()
port_stem = PorterStemmer()
news_dataset = pd.read_csv('Cleaned_News_Dataset.csv')
model = pickle.load(open("FakeNewsmodel.pkl", 'rb'))
X = news_dataset['text']
Y = news_dataset['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
tfid_x_train = vectorizer.fit_transform(X_train.values.astype('U'))
tfid_x_test = vectorizer.transform(X_test.values.astype('U'))
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = vectorizer.transform(input_data)
    prediction = model.predict(vectorized_input_data)
    prediction = le.inverse_transform(prediction)
    return prediction

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
        return render_template('index.html', prediction=pred[0], prob_real=real_prob, prob_fake=fake_prob)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True, port=5001)