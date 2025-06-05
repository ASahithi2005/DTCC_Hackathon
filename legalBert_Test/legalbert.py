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

# Load regulations from JSON
json_path = '../knowledgeBase.json'
with open(json_path, 'r', encoding='utf-8') as f:
    regulations = json.load(f)

# Example: Suppose you extracted this text from the uploaded policy PDF
extracted_policy_text = """
All foreign contributions received shall be reported annually using form FC-3.
The NGO has an FCRA registration certificate and maintains a designated bank account.
"""

embedding_policy = get_embedding(extracted_policy_text)

# Store regulations with low similarity
missing_regulations = []

# Threshold below which we consider a regulation "missing"
SIMILARITY_THRESHOLD = 0.5

for entry in regulations:
    clause_text = entry["clause"]["text"]
    embedding_clause = get_embedding(clause_text)
    similarity = util.pytorch_cos_sim(embedding_policy, embedding_clause)
    similarity_score = similarity.item()
    if similarity_score < SIMILARITY_THRESHOLD:
        missing_regulations.append({
            "clause_id": entry["clause"]["id"],
            "country": entry["country"],
            "law": entry["law"],
            "clause_text": clause_text,
            "similarity_score": similarity_score,
            "penalties": entry["clause"].get("penalties", []),
            "evidence_required": entry["clause"].get("evidence_required", [])
        })

# Print the Compliance Gap Analysis Report
print("\n📑 Compliance Gap Analysis Report")
print("\n✅ Analyzed NGO Policy Document:\n'uploaded_policy.pdf'")
print("\n🔎 Summary:")
print("We detected that the policy document **does not explicitly cover** the following regulatory requirements. These gaps may expose the organization to compliance risks.\n")
print("─────────────────────────────")
print("🚨 Regulation Gaps Detected:")
print("─────────────────────────────")

for reg in missing_regulations:
    print(f"\n❌ Clause ID: {reg['clause_id']}")
    print(f"Country: {reg['country']} 🇮🇳" if reg['country'] == "IN" else f"Country: {reg['country']}")
    print(f"Law: {reg['law']}")
    print(f"Clause Text:\n\"{reg['clause_text']}\"")
    print("Penalty:")
    for penalty in reg['penalties']:
        print(f" - Type: {penalty['type'].capitalize()}")
        print(f" - Severity: {penalty['severity'].capitalize()}")
        print(f" - Details: {penalty['details']}")
    print("Evidence Required:")
    for evidence in reg['evidence_required']:
        print(f" - {evidence}")
    print("\n─────────────────────────────")

print("\n💡 Recommendations:")
print("- Update your policy to address these missing clauses.")
print("- Collect and maintain the required evidence for each clause.")
print("- Regularly review your policy as regulations evolve.")
