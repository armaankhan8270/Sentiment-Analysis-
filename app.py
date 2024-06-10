import pickle
import re
import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


stopwords_list = stopwords.words('english')

# Function to preprocess the text


def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets, remove links, remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    PreProcessText = []

    for word in text.split():
        if word not in stopwords_list:
            PreProcessText.append(word)
    return " ".join(PreProcessText)


# Load the pickled models and TF-IDF vectorizer
with open('rfc1.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf1.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
with open('preprocess1.pkl', 'rb') as file:
    preprocess = pickle.load(file)

# Function to predict sentiment


def predict_sentiment(text):
    processed_text = [preprocess(text)]
    vectorized_text = tfidf_vectorizer.transform(processed_text)

    prediction = model.predict(vectorized_text)[0]
    print(prediction)
    return prediction


# Streamlit app
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter text for sentiment analysis:", "")
if st.button("Predict Sentiment"):

    sentiment = predict_sentiment(user_input)
    if sentiment == 1:
        st.write('postive')
    else:
        st.write('negitive')
