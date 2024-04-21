from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import os
app = Flask(__name__, template_folder=os.path.abspath('templates'))
# Load the trained model
with open('sent.pkl', 'rb') as file:
    naive_bayes_classifier = pickle.load(file, encoding='latin1')
# Load the TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file, encoding='latin1')
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_tr = transform_text(text)
    text_tra = cv.transform([text_tr])
    yy = naive_bayes_classifier.predict(text_tra)
    dd = {0: 'sad', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    for i in dd:
        if yy == i:
            emotion = dd[i]
            break
    return render_template('result.html', emotion=emotion, text=text)
if __name__ == '__main__':
    app.run(debug=True)
