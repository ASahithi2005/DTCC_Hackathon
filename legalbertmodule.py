# legalbertmodule.py

import json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch

# Load the Multilingual BERT model
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def analyze_policy_text(policy_text, knowledge_base, threshold_matched=0.75, threshold_partial=0.55):
    embedding_policy = get_embedding(policy_text)

    matched = []
    partial = []
    unmatched = []

    for entry in knowledge_base:
        clause_text = entry["clause"]["text"]
        embedding_clause = get_embedding(clause_text)
        similarity = util.pytorch_cos_sim(embedding_policy, embedding_clause)
        similarity_score = similarity.item()

        clause_data = {
            "clause": {
                "id": entry["clause"]["id"],
                "text": clause_text,
                "penalties": entry["clause"].get("penalties", []),
                "evidence_required": entry["clause"].get("evidence_required", [])
            },
            "country": entry["country"],
            "law": entry["law"],
            "similarity_score": round(similarity_score, 2),
            "match_score": round(similarity_score, 2),
            "explanation": "",
        }

        if similarity_score >= threshold_matched:
            clause_data["status"] = "matched"
            clause_data["explanation"] = "Clause fully covered in internal policy."
            matched.append(clause_data)
        elif similarity_score >= threshold_partial:
            clause_data["status"] = "partial"
            clause_data["explanation"] = "Clause partially covered. Check for missing legal nuances."
            partial.append(clause_data)
        else:
            clause_data["status"] = "unmatched"
            clause_data["explanation"] = "Clause not found in internal policy. May lead to compliance risk."
            unmatched.append(clause_data)

    return matched, partial, unmatched
