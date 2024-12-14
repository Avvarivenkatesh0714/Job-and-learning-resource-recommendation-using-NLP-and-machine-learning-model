from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from preprocess import preprocess_resume, match_job_with_resume

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume_file' not in request.files:
        return redirect(request.url)

    file = request.files['resume_file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Read and preprocess resume
        with open(filepath, 'r', encoding='utf-8') as f:
            resume_text = f.read()

        best_job, missing_skills, resources = match_job_with_resume(resume_text)

        # Pass data to results page
        return render_template(
            'results.html',
            best_job=best_job,
            missing_skills=missing_skills,
            resources=resources
        )

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
