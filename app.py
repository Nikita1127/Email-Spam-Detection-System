import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_email = st.text_area("Enter the Email Message")

if st.button('Predict'):

 # 1. Preprocess
 transform_email = transform_text(input_email)
 # 2. Vectorize
 vector_input = tfidf.transform([transform_email])
 # 3. Predict
 result = model.predict(vector_input)[0]
 # 4. Display
 if result == 1:
     st.header("Spam")
 else:
     st.header("Not Spam")