from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer



app = Flask(__name__)
CORS(app)  

try:
    with open(r'C:\Users\Aryan\Desktop\New folder\backend\flaskServer\sent (2).pkl', 'rb') as file:
        model = pickle.load(file, encoding='latin1')
        print("Pickle file loaded successfully")
except FileNotFoundError:
    print("Error: 'sent (2).pkl' file not found")
except Exception as e:
    print("Error loading pickle file:", e)

# Load the TfidfVectorizer
try:
    with open(r'C:\Users\Aryan\Desktop\New folder\backend\flaskServer\tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
        print("TfidfVectorizer loaded successfully")
except FileNotFoundError:
    print("Error: 'tfidf_vectorizer.pkl' file not found")
except Exception as e:
    print("Error loading TfidfVectorizer:", e)

#text transform function
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
def hello():
    return 'Hello, Flask!'

@app.route("/members", methods=['POST'])
def members():
    data = request.get_json()  # extract JSON data from request

    if data is None or 'expression' not in data:
        return jsonify({"error": "Invalid JSON data or missing 'expression'"}), 400

    expression = data['expression']


    input_2 = transform_text(expression)
    # input_3 = [input_2]

#   numeric feature using TF-IDF vectorization
    expression_vector = vectorizer.transform([input_2])

#     # print(expression_vector)
#   (0, 1050)     0.7074948831565999
#   (0, 983)      0.7067184660861984    vectorized o/p for hello nice to meet you


    result = model.predict(expression_vector)

    element = result.tolist()[0]
    dic = {0: 'sad', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    emotion = ""
    for i in dic:
        if i == element:
            emotion = dic[i]
            break



    # Prepare response to request 
    response_data = {
        "expression": expression,
        "result": emotion  # Convert numpy array to a list
    }

    return jsonify(response_data), 200



if __name__ == "__main__":
    app.run(debug=True)