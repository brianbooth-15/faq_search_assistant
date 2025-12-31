#!/usr/bin/env python3

import json
import os
from sentence_transformers import SentenceTransformer, util

# Load FAQs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "src", "faq_search_assistant", "data", "faqs.json")

with open(FAQ_FILE, "r") as f:
    faqs = json.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings including answer content
faq_texts = []
for faq in faqs:
    answer_keywords = faq['answer'].lower()
    text = f"{faq['question']} {answer_keywords}"
    faq_texts.append(text)

faq_embeddings = model.encode(faq_texts, convert_to_tensor=True)

# Test query
test_query = "what is Exclusive Player Games"
user_embedding = model.encode(test_query, convert_to_tensor=True)
cosine_scores = util.cos_sim(user_embedding, faq_embeddings)[0]

print(f"Query: {test_query}")
print("\nTop 5 matches:")
top_5_scores, top_5_indices = cosine_scores.topk(5)

for i, (score, idx) in enumerate(zip(top_5_scores, top_5_indices)):
    print(f"\n{i+1}. Score: {float(score):.4f}")
    print(f"   Question: {faqs[idx]['question']}")
    print(f"   Answer preview: {faqs[idx]['answer'][:100]}...")
    
    # Check if this FAQ mentions "Exclusive Player Games"
    if "exclusive player games" in faqs[idx]['answer'].lower():
        print("   *** CONTAINS 'Exclusive Player Games' ***")