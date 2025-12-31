#!/usr/bin/env python3

import json
import os

# Load FAQs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "src", "faq_search_assistant", "data", "faqs.json")

with open(FAQ_FILE, "r") as f:
    faqs = json.load(f)

def simulate_ask(user_question):
    print(f"Question: {user_question}")
    
    # Simulate low similarity (threshold = 0.2, assume we get 0.1)
    top_score = 0.1
    SIMILARITY_THRESHOLD = 0.2
    
    if top_score < SIMILARITY_THRESHOLD:
        # Keyword matching logic
        query_words = user_question.lower().split()
        keyword_matches = []
        
        for i, faq in enumerate(faqs):
            faq_text = f"{faq['question']} {faq['answer']}".lower()
            match_count = sum(1 for word in query_words if word in faq_text)
            if match_count > 0:
                keyword_matches.append((i, match_count))
        
        keyword_matches.sort(key=lambda x: x[1], reverse=True)
        
        if keyword_matches:
            top_idxs = [idx for idx, _ in keyword_matches[:5]]
            print(f"Top FAQ found: {faqs[top_idxs[0]]['question']}")
            
            # Create context
            context = "\\n\\n".join([
                f"FAQ {i+1}:\\nQ: {faqs[idx]['question']}\\nA: {faqs[idx]['answer']}" 
                for i, idx in enumerate(top_idxs[:3])  # Show top 3
            ])
            
            print("\\nContext that would be sent to AI:")
            print("=" * 50)
            print(context[:500] + "..." if len(context) > 500 else context)
            print("=" * 50)
            
            # Expected answer
            if "exclusive player games" in faqs[top_idxs[0]]['answer'].lower():
                print("\\n✅ SUCCESS: Top FAQ contains 'Exclusive Player Games'")
                print("Expected AI response should mention:")
                print("- Exclusive Player Games are available in My Account")
                print("- Prizes include iPad, smart TV, PS5, etc.")
            else:
                print("\\n❌ FAILED: Top FAQ doesn't contain the answer")

# Test the problematic query
simulate_ask("what is Exclusive Player Games")