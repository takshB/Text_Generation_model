import flask
import pickle
from flask import Flask, render_template, url_for, redirect, request
import tensorflow
import keras 
import pandas as pd
import numpy as np
import re
import pickle5 as pickle
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
model = keras.models.load_model('LSTM_Attention3_10.h5', custom_objects=SeqSelfAttention.get_custom_objects())
seq_length = 50

with open('LSTM_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_words(seed, no_words=15):
    for i in range(no_words):
        token_list = tokenizer.texts_to_sequences([seed])[0]
        token_list = pad_sequences([token_list], maxlen=9, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=1)

        new_word = ''

        for word, index in tokenizer.word_index.items():
            if predicted == index:
                new_word = word
                break
        seed += " " + new_word
    return seed

@app.route('/', methods = ["GET", "POST"])
def home():
    if request.method == "POST":
        model_input = request.form.get("model_a")
        model_input = model_input.split(' ')
        user = ' '.join(model_input)
        input_is = []
        for i in model_input:
            if i != '':
                input_is.append(i)
        model_input = ''.join(input_is)
        if model_input ==  '':
            return render_template('index.html', model_output = 'Enter Something')
        else:
            return render_template('index.html', model_output=predict_words(user, seq_length))
    return render_template('index.html', model_input=" ")

if __name__ == "__main__":
    app.run(debug=True)
