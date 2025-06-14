# legalbertmodule.py

import json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch
import re

# Load the Multilingual BERT model
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def chunk_text(text, max_tokens=200):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for sentence in sentences:
        tokenized_len = len(tokenizer.tokenize(sentence))
        if current_length + tokenized_len > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = tokenized_len
        else:
            current_chunk += " " + sentence
            current_length += tokenized_len
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def analyze_policy_text(policy_text, knowledge_base, threshold_matched=0.75, threshold_partial=0.55, ngo_type=None, region=None, country=None):
    chunks = chunk_text(policy_text)
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

    matched = []
    partial = []
    unmatched = []

    for entry in knowledge_base:
        clause = entry["clause"]

        # Filter by country
        if country and entry.get("country", "").lower() != country.lower():
            continue

        # Filter by NGO type
        applicable_types = clause.get("applicable_ngo_types", ["All"])
        if ngo_type and ngo_type not in applicable_types and "All" not in applicable_types:
            continue

        # Filter by region
        applicable_regions = clause.get("applicable_regions", ["All"])
        if region and region not in applicable_regions and "All" not in applicable_regions:
            continue

        clause_text = clause["text"]
        clause_embedding = get_embedding(clause_text)

        max_similarity = max(util.pytorch_cos_sim(clause_embedding, chunk_emb).item() for chunk_emb in chunk_embeddings)

        clause_data = {
            "clause": {
                "id": clause["id"],
                "text": clause_text,
                "penalties": clause.get("penalties", []),
                "evidence_required": clause.get("evidence_required", []),
                "applicable_ngo_types": applicable_types,
                "applicable_regions": applicable_regions
            },
            "country": entry["country"],
            "law": entry["law"],
            "similarity_score": round(max_similarity, 2),
            "match_score": round(max_similarity, 2),
            "explanation": "",
        }

        if max_similarity >= threshold_matched:
            clause_data["status"] = "matched"
            clause_data["explanation"] = "Clause fully covered in internal policy."
            matched.append(clause_data)
        elif max_similarity >= threshold_partial:
            clause_data["status"] = "partial"
            clause_data["explanation"] = "Clause partially covered. Check for missing legal nuances."
            partial.append(clause_data)
        else:
            clause_data["status"] = "unmatched"
            clause_data["explanation"] = "Clause not found in internal policy. May lead to compliance risk."
            unmatched.append(clause_data)

    return matched, partial, unmatched