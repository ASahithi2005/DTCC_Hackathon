import os
import json
from typing import List, Dict
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import docx2txt
import PyPDF2

from sentence_transformers import SentenceTransformer, util
import re

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Load knowledge base
with open('expanded_knowledgebase.json', 'r') as f:
    knowledge_base = json.load(f)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(uploaded_file):
    # Use uploaded_file.filename to get extension
    if '.' in uploaded_file.filename:
        ext = uploaded_file.filename.rsplit('.', 1)[1].lower()
    else:
        ext = ''

    if ext == 'pdf':
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif ext in ['txt', 'text']:
        return uploaded_file.read().decode('utf-8')
    elif ext in ['docx']:
        import docx
        document = docx.Document(uploaded_file)
        full_text = []
        for para in document.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    else:
        return ""



model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and good for semantic similarity

def match_policies_to_kb(policy_text, knowledge_base, threshold=0.75, partial_threshold=0.55):
    matched_results = []

    doc_sentences = [s.strip() for s in policy_text.split('.') if len(s.strip()) > 10]
    doc_embeddings = model.encode(doc_sentences, convert_to_tensor=True)

    for entry in knowledge_base:
        clause_text = entry['clause']['text']
        clause_embedding = model.encode(clause_text, convert_to_tensor=True)
        similarities = util.cos_sim(clause_embedding, doc_embeddings)[0]
        max_sim = similarities.max().item()

        if max_sim >= threshold:
            status = "matched"
            explanation = "Clause fully covered in internal policy."
        elif max_sim >= partial_threshold:
            status = "partial"
            explanation = "Clause partially covered. Check for missing legal nuances or specificity."
        else:
            status = "unmatched"
            explanation = "Clause not found in internal policy. May lead to compliance risk."

        matched_results.append({
            **entry,
            "match_score": round(max_sim, 2),
            "status": status,
            "explanation": explanation
        })

    return matched_results
def match_clauses(text, knowledge_base):
    matched = []
    partial = []
    unmatched = []

    for clause in knowledge_base['clauses']:
  # assuming knowledge_base is a list of clauses
        # Example scoring logic:
        score = 0
        for keyword in clause.get('keywords', []):
            if keyword.lower() in text.lower():
                score += 1
        score /= max(len(clause.get('keywords', [])), 1)

        if score >= 0.75:
            matched.append({'clause': clause, 'match_score': score, 'status': 'matched', 'explanation': 'Fully matched.'})
        elif score >= 0.55:
            partial.append({'clause': clause, 'match_score': score, 'status': 'partial', 'explanation': 'Partially matched.'})
        else:
            unmatched.append({'clause': clause, 'match_score': score, 'status': 'unmatched', 'explanation': 'Not matched.'})

    return matched, partial, unmatched



@app.route('/')
def index():
    return '''<h1>Upload NGO Policy Document</h1>
              <form method="POST" action="/upload" enctype="multipart/form-data">
              <input type="file" name="file">
              <input type="submit" value="Upload">
              </form>'''

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    text = extract_text(file)  # Your existing text extraction logic

    # Load knowledge base
    with open('knowledgebase.json') as kb_file:
        kb = json.load(kb_file)

    # Run matching
    matched, partial, unmatched = match_clauses(text, kb)

    # 👇 Add this block
    all_matches = matched + partial + unmatched
    with open("last_results.json", "w") as out:
        json.dump({
            "matched": matched,
            "partial": partial,
            "unmatched": unmatched,
            "all": all_matches
        }, out, indent=2)

    # Show result page
    return render_template('result.html',
                           matches=all_matches,
                           unmatched=unmatched)

@app.route('/dashboard')
def dashboard():
    with open('last_results.json') as f:
        results = json.load(f)  # Store matches from the last upload step
    return render_template('dashboard.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
