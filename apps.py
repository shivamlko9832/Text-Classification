#Step 1: Import Libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


##Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}


# Load the pre-trained model with the ReLU activation
model = load_model('simple_rnn_imdb.h5')

#Step 2: Helper functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

#function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)

    return padded_review


#Step 3:Prediction Function
def predict_sentiment(review):
    padded_review = preprocess_text(review)
    pred = model.predict(padded_review)
    sentiment = 'Positive' if pred[0][0] > 0.5 else 'Negative'
    return sentiment, pred[0][0]



#Streamlit app

import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it a positive or negative.')

#User Input
user_input = st.text_area('Movie Review')
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    
    #Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'
    
    #Display the result
    st.write(f"Sentiment:{sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write('Please enter a movie Review')
    
