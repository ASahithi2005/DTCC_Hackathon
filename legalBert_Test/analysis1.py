import os
import json
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import PyPDF2
import docx
from legalBert_Test.legalbertmodule1 import analyze_policy_text  # Import our updated helper

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Load knowledge base (Make sure this file exists!)
with open('knowledgeBase.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(uploaded_file):
    ext = uploaded_file.filename.rsplit('.', 1)[1].lower() if '.' in uploaded_file.filename else ''

    if ext == 'pdf':
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif ext in ['txt', 'text']:
        return uploaded_file.read().decode('utf-8')
    elif ext == 'docx':
        document = docx.Document(uploaded_file)
        full_text = [para.text for para in document.paragraphs]
        return '\n'.join(full_text)
    else:
        return ""

@app.route('/')
def index():
    return '''
        <h1>Upload NGO Policy Document</h1>
        <form method="POST" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part."

    file = request.files['file']
    if file.filename == '':
        return "No selected file."

    if file and allowed_file(file.filename):
        text = extract_text(file)

        # Run LegalBERT analysis
        matched, partial, unmatched = analyze_policy_text(text, knowledge_base)

        all_matches = matched + partial + unmatched

        # Save results for dashboard
        with open("last_results.json", "w", encoding='utf-8') as out:
            json.dump({
                "matched": matched,
                "partial": partial,
                "unmatched": unmatched,
                "all": all_matches
            }, out, indent=2)

        return render_template('result.html', matches=all_matches, unmatched=unmatched)

    return "File type not allowed."

@app.route('/dashboard')
def dashboard():
    with open('last_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    return render_template('dashboard.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)