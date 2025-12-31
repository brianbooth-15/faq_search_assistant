# src/faq_search_assistant/test_ollama.py

import subprocess

# -----------------------------
# 1. Choose your model
# -----------------------------
MODEL_NAME = "qwen3"

# -----------------------------
# 2. Ensure model is downloaded
# -----------------------------
try:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True
    )
    installed_models = result.stdout.splitlines()
    if MODEL_NAME not in installed_models:
        print(f"Pulling model {MODEL_NAME}...")
        subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
except subprocess.CalledProcessError as e:
    print("Error checking installed models:", e)
    exit(1)

# -----------------------------
# 3. Interactive chat
# -----------------------------
print(f"\nStarting interactive chat with {MODEL_NAME}. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Exiting chat.")
        break

    try:
        # Use stdin to pass the prompt text
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME],
            input=user_input,
            capture_output=True,
            text=True,
            check=True
        )
        response = result.stdout.strip()
        print(f"{MODEL_NAME}: {response}\n")

    except subprocess.CalledProcessError as e:
        print("Error calling Ollama:", e)
        break
