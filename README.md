# FAQ Search Assistant

AI-powered RAG (Retrieval-Augmented Generation) chatbot for customer service FAQs using semantic search and natural language generation.

## Features

- **Semantic Search**: Uses sentence transformers for intelligent FAQ matching
- **AI Generation**: Ollama integration for natural language responses
- **Category-based FAQs**: Organized FAQ files by topic
- **FastAPI Backend**: RESTful API with automatic documentation
- **Hybrid Matching**: Combines semantic similarity with keyword matching

## Dependencies

- Python 3.8+
- [Ollama](https://ollama.ai) with qwen3 model
- Poetry for dependency management

## Installation

1. **Install Ollama and pull model:**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull qwen3
   ```

2. **Install Python dependencies:**
   ```bash
   poetry install
   ```

## Running

1. **Start the API server:**
   ```bash
   poetry run uvicorn src.faq_search_assistant.app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Test the API:**
   ```bash
   curl -X POST "http://localhost:8000/ask" \
        -H "Content-Type: application/json" \
        -d '{"question": "How much does a ticket cost?"}'
   ```

3. **View API docs:** http://localhost:8000/docs

## API Endpoints

- `POST /ask` - Ask a question
- `POST /debug` - Debug similarity scores
- `GET /health` - Health check

## FAQ Management

FAQs are stored in `src/faq_search_assistant/data/` as JSON files by category:
- `getting_in_touch.json`
- `draws_and_payments_faqs.json`
- `charity_faqs.json`
- etc.

Add new FAQs by creating/editing these files.