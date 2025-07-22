from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle


model = load_model('resume_classifier_model.keras')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

MAX_SEQUENCE_LENGTH = 300

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    if request.method == 'POST':
        uploaded_file = request.files['resume']
        if uploaded_file and uploaded_file.filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            if 'Resume_str' not in df.columns:
                return "Column 'Resume_str' not found."

            texts = df['Resume_str'].fillna('').astype(str).tolist()
            sequences = tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            preds = model.predict(padded)
            predicted_labels = np.argmax(preds, axis=1)
            job_roles = label_encoder.inverse_transform(predicted_labels)
            job_roles = [BeautifulSoup(str(role), "html.parser").get_text() for role in job_roles]

            predictions = list(zip(texts, job_roles))

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
