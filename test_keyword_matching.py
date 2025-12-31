#!/usr/bin/env python3

import json
import os

# Load FAQs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "src", "faq_search_assistant", "data", "faqs.json")

with open(FAQ_FILE, "r") as f:
    faqs = json.load(f)

# Test the keyword matching logic
user_question = "what is Exclusive Player Games"
query_words = user_question.lower().split()
keyword_matches = []

print(f"Query: {user_question}")
print(f"Query words: {query_words}")
print()

for i, faq in enumerate(faqs):
    faq_text = f"{faq['question']} {faq['answer']}".lower()
    match_count = sum(1 for word in query_words if word in faq_text)
    if match_count > 0:
        keyword_matches.append((i, match_count))
        print(f"FAQ #{i+1}: {match_count} matches - {faq['question'][:50]}...")

# Sort by match count and take top matches
keyword_matches.sort(key=lambda x: x[1], reverse=True)

print(f"\nTop 3 keyword matches:")
for i, (idx, count) in enumerate(keyword_matches[:3]):
    print(f"{i+1}. FAQ #{idx+1} ({count} matches): {faqs[idx]['question']}")
    if "exclusive player games" in faqs[idx]['answer'].lower():
        print("   *** Contains 'Exclusive Player Games' ***")