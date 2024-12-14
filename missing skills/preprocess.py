import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocess Resume: Tokenize, remove stopwords, and lemmatize
def preprocess_resume(resume_text):
    # Tokenization
    words = word_tokenize(resume_text.lower())

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return ' '.join(lemmatized_words)

# Load job data (job descriptions, skills, etc.)
def load_jobs_data():
    return pd.read_csv('jobs.csv')

# Load learning resources data (skills and associated resources)
def load_learning_resources():
    return pd.read_csv('learning_resources.csv')

# Feature Extraction: Using TF-IDF Vectorizer and calculating cosine similarity
def extract_features(job_descriptions, resume_text):
    tfidf_vectorizer = TfidfVectorizer()
    all_text = job_descriptions + [resume_text]
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_text)
    return cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

# Matching Jobs with Resume: Find the best match job and suggest missing skills
def match_job_with_resume(resume_text):
    # Load the job data and learning resources
    jobs_data = load_jobs_data()
    learning_resources = load_learning_resources()

    # Preprocess the resume text
    processed_resume = preprocess_resume(resume_text)

    # Fill NaN job descriptions with empty strings to avoid errors
    jobs_data['job_description'] = jobs_data['job_description'].fillna('')

    # Get job descriptions and titles
    job_descriptions = jobs_data['job_description'].apply(lambda x: x if isinstance(x, str) else "").tolist()
    job_titles = jobs_data['job_title'].tolist()

    # Get cosine similarity between job descriptions and resume
    cosine_similarities = extract_features(job_descriptions, processed_resume)

    # Find the index of the best match job
    best_match_index = np.argmax(cosine_similarities)
    best_job = job_titles[best_match_index]
    best_job_skills = jobs_data.iloc[best_match_index]['skills'].split(',')

    # Find missing skills by comparing the resume skills with job skills
    resume_skills = processed_resume.split()
    missing_skills = list(set(best_job_skills) - set(resume_skills))

    # Fetch learning resources for missing skills
    missing_skills_resources = {}
    for skill in missing_skills:
        resources = learning_resources[learning_resources['skill'] == skill]
        missing_skills_resources[skill] = resources[['resource_type', 'resource_link']].values.tolist()

    return best_job, missing_skills, missing_skills_resources

