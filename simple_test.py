#!/usr/bin/env python3

import json
import os

# Load FAQs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "src", "faq_search_assistant", "data", "faqs.json")

with open(FAQ_FILE, "r") as f:
    faqs = json.load(f)

# Test query
test_query = "exclusive player games"

print(f"Searching for: '{test_query}'")
print("\nFAQs that contain 'exclusive player games':")

found = False
for i, faq in enumerate(faqs):
    if test_query.lower() in faq['answer'].lower() or test_query.lower() in faq['question'].lower():
        found = True
        print(f"\nFAQ #{i+1}:")
        print(f"Question: {faq['question']}")
        print(f"Answer: {faq['answer'][:200]}...")
        print(f"Category: {faq['category']}")

if not found:
    print("No FAQs found containing 'exclusive player games'")
else:
    print(f"\nThe semantic search should find this FAQ when asked about 'Exclusive Player Games'")