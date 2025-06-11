import os
import json
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import PyPDF2
import docx
from legalbertmodule import analyze_policy_text

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Load knowledge base
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
        return ''.join([page.extract_text() or '' for page in pdf_reader.pages])
    elif ext in ['txt', 'text']:
        return uploaded_file.read().decode('utf-8')
    elif ext == 'docx':
        document = docx.Document(uploaded_file)
        return '\n'.join([para.text for para in document.paragraphs])
    return ""

@app.route('/')
def index():
    return '''
        <h1>Upload NGO Policy Document</h1>
        <form method="POST" action="/upload" enctype="multipart/form-data">
            <label>Country Code (e.g., IN):</label><br>
            <input type="text" name="country" required><br><br>
            <label>Region/State (e.g., Maharashtra):</label><br>
            <input type="text" name="region" required><br><br>
            <label>NGO Type:</label><br>
            <select name="ngo_type" required>
                <option value="Trust">Trust</option>
                <option value="Society">Society</option>
                <option value="Section 8 Company">Section 8 Company</option>
            </select><br><br>
            <label>Select File:</label><br>
            <input type="file" name="file" required><br><br>
            <input type="submit" value="Upload">
        </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file or file.filename == '':
        return "No file selected."

    if not allowed_file(file.filename):
        return "Unsupported file type."

    country = request.form.get("country", "").strip()
    region = request.form.get("region", "").strip()
    ngo_type = request.form.get("ngo_type", "").strip()

    text = extract_text(file)
    matched, partial, unmatched = analyze_policy_text(
        text, knowledge_base, ngo_type=ngo_type, region=region, country=country
    )

    all_matches = matched + partial + unmatched

    filters = {
        "country": country,
        "region": region,
        "ngo_type": ngo_type
    }

    # Save results and filters for dashboard
    with open("last_results.json", "w", encoding='utf-8') as out:
        json.dump({
            "matched": matched,
            "partial": partial,
            "unmatched": unmatched,
            "all": all_matches
        }, out, indent=2)

    with open("last_filters.json", "w", encoding='utf-8') as f:
        json.dump(filters, f, indent=2)

    return render_template('result.html', matches=all_matches, unmatched=unmatched, filters=filters)

@app.route('/dashboard')
def dashboard():
    with open('last_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    filters = {}
    if os.path.exists("last_filters.json"):
        with open("last_filters.json", "r", encoding="utf-8") as f:
            filters = json.load(f)

    return render_template('dashboard.html', results=results, filters=filters)

if __name__ == '__main__':
    app.run(debug=True)
