import os
import json
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware

# Fix tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="FAQ Assistant with Semantic Search")

# Allow requests from any origin (for local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request model
# -----------------------------
class QuestionRequest(BaseModel):
    question: str

# -----------------------------
# Load FAQs
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "data", "faqs.json")

with open(FAQ_FILE, "r") as f:
    faqs = json.load(f)

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute FAQ embeddings - include both questions and key terms from answers
faq_texts = []
for faq in faqs:
    # Extract key terms from answers for better matching
    answer_keywords = faq['answer'].lower()
    text = f"{faq['question']} {answer_keywords}"
    faq_texts.append(text)

faq_embeddings = model.encode(faq_texts, convert_to_tensor=True)

def clean_response(response: str) -> str:
    """
    Clean up AI response by removing thinking artifacts but keep markdown formatting.
    """
    import re
    
    # Find the final answer after "...done Answer:" or just "Answer:"
    answer_match = re.search(r'(?:.*?done\s+)?Answer:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        cleaned = answer_match.group(1).strip()
    else:
        # If no "Answer:" found, remove everything before the last substantial sentence
        lines = response.split('\n')
        substantial_lines = [line.strip() for line in lines if len(line.strip()) > 20 and not any(word in line.lower() for word in ['okay', 'let me', 'looking', 'based on', 'according to', 'i need'])]
        
        if substantial_lines:
            cleaned = substantial_lines[-1]
        else:
            return "I don't have specific information about that."
    
    # Keep markdown formatting - just clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
    
    return cleaned

# -----------------------------
# Helper function to call Ollama
# -----------------------------
def run_ollama(prompt: str) -> str:
    """
    Call Ollama CLI with the given prompt and return the response.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "qwen3"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        response = result.stdout.strip()
        if not response:
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        return response
    except subprocess.TimeoutExpired:
        return "The request timed out. Please try again."
    except subprocess.CalledProcessError as e:
        return f"I'm having trouble accessing the information right now. Please try again later."

# -----------------------------
# Debug endpoint to test similarity scores
# -----------------------------
@app.post("/debug")
def debug_similarity(request: QuestionRequest):
    user_question = request.question.strip()
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, faq_embeddings)[0]
    
    # Get top 5 matches with scores
    top_5_scores, top_5_indices = cosine_scores.topk(5)
    
    results = []
    for i, (score, idx) in enumerate(zip(top_5_scores, top_5_indices)):
        results.append({
            "rank": i + 1,
            "score": float(score),
            "question": faqs[idx]['question'],
            "answer_preview": faqs[idx]['answer'][:100] + "..."
        })
    
    return {"query": user_question, "matches": results}

# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy", "faqs_loaded": len(faqs)}

# -----------------------------
# Ask endpoint
# -----------------------------
@app.post("/ask")
def ask_question(request: QuestionRequest):
    user_question = request.question.strip()
    
    if not user_question:
        return {"answer": "Please provide a question."}

    # Check for general contact queries only (not specific questions)
    general_contact_patterns = [
        r'^how (can|do) i contact',
        r'^contact (us|customer|support)',
        r'^customer service$',
        r'^how to (reach|contact)',
        r'^speak to customer service'
    ]
    
    import re
    is_general_contact = any(re.search(pattern, user_question.lower()) for pattern in general_contact_patterns)
    
    if is_general_contact:
        contact_response = """You can contact People's Postcode Lottery customer service in these ways:

**Phone:** **0808 109 8765** (freephone)
**Online:** Use the Contact Us form at https://www.postcodelottery.co.uk/about-us/contact-us
**Post:** Write to:
**People's Postcode Lottery**
28 Charlotte Square
Edinburgh
EH2 4ET
United Kingdom"""
        return {"answer": contact_response, "format": "markdown"}

    # Compute embedding for user question
    user_embedding = model.encode(user_question, convert_to_tensor=True)

    # Compute cosine similarity with FAQs
    cosine_scores = util.cos_sim(user_embedding, faq_embeddings)[0]

    # Find best match
    top_idx = cosine_scores.argmax().item()
    top_score = cosine_scores[top_idx].item()

    SIMILARITY_THRESHOLD = 0.8  # Much higher threshold to force AI processing

    if top_score < SIMILARITY_THRESHOLD:
        # Use keyword matching for better results
        query_words = user_question.lower().split()
        keyword_matches = []
        
        for i, faq in enumerate(faqs):
            faq_text = f"{faq['question']} {faq['answer']}".lower()
            match_count = sum(1 for word in query_words if word in faq_text)
            if match_count > 0:
                keyword_matches.append((i, match_count))
        
        keyword_matches.sort(key=lambda x: x[1], reverse=True)
        
        if keyword_matches:
            top_idxs = [idx for idx, _ in keyword_matches[:3]]
        else:
            # Fallback response with contact info
            fallback_response = f"""I don't have specific information about that. For assistance, please contact customer service:

**Phone:** **0808 109 8765** (freephone)
**Online:** Contact Us form at https://www.postcodelottery.co.uk/about-us/contact-us"""
            return {"answer": fallback_response, "format": "markdown"}
        
        context = "\n\n".join([
            f"{faqs[idx]['question']}: {faqs[idx]['answer']}" 
            for idx in top_idxs
        ])
        
        prompt = f"""Answer this question: {user_question}

Extract only the relevant information from these FAQs to answer the question directly and concisely. Use **bold** for important details like phone numbers, addresses, and key information:

{context}

Answer:"""

        ai_response = run_ollama(prompt)
        
        # Clean up any thinking artifacts
        ai_response = clean_response(ai_response)
        
        return {"answer": ai_response, "format": "markdown"}

    # High similarity: also process through AI for better responses
    context = f"{faqs[top_idx]['question']}: {faqs[top_idx]['answer']}"
    
    prompt = f"""Answer this question: {user_question}

Extract only the relevant information from this FAQ to answer the question directly and concisely. Use **bold** for important details like phone numbers, addresses, and key information:

{context}

Answer:"""
    
    ai_response = run_ollama(prompt)
    ai_response = clean_response(ai_response)
    
    return {"answer": ai_response, "format": "markdown"}
