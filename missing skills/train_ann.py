import os
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Preprocessing
def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return ' '.join(lemmatized_words)

# Load and preprocess job data
def load_jobs_data():
    jobs_df = pd.read_csv('jobs.csv')
    jobs_df['job_description'] = jobs_df['job_description'].fillna('').apply(preprocess_text)
    return jobs_df

# Train ANN Model
def train_ann_model():
    jobs_data = load_jobs_data()

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(jobs_data['job_description']).toarray()

    # Encode job titles
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(jobs_data['job_title'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ANN Model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model, tfidf_vectorizer, label_encoder

# Train model at startup
ann_model, tfidf_vectorizer, label_encoder = train_ann_model()

# Flask routes
@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    """Upload and process resume"""
    if 'resume' not in request.files:
        return 'No resume uploaded!', 400

    file = request.files['resume']
    if file.filename == '':
        return 'No selected file!', 400

    if file:
        resume_text = file.read().decode('utf-8')  # Assuming resume is a text file
        processed_resume = preprocess_text(resume_text)

        # Vectorize resume text
        resume_vector = tfidf_vectorizer.transform([processed_resume]).toarray()

        # Predict job
        predictions = ann_model.predict(resume_vector)
        best_job_index = np.argmax(predictions)
        best_job = label_encoder.inverse_transform([best_job_index])[0]

        return render_template("results.html", best_job=best_job)

if __name__ == '__main__':
    app.run(debug=True)
