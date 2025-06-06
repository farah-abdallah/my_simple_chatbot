# app.py

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
from rag_model import SimpleRAG

# 1. Load environment variables (.env)
load_dotenv()  # takes OPENAI_API_KEY from .env

# 2. Create Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret123")  
# – Flask needs a secret key for sessions/flash messaging. In production, set a real secret in .env.

# 3. Initialize (or lazily initialize) the RAG model
#    For simplicity, we initialize at startup. You can also delay until first request.
PDF_PATH = "data/Understanding_Climate_Change.pdf"  # or wherever your PDF lives
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
N_RETRIEVED = int(os.getenv("N_RETRIEVED", 2))

try:
    rag = SimpleRAG(
        pdf_path=PDF_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        n_retrieved=N_RETRIEVED
    )
except Exception as e:
    print("Error initializing RAG model:", e)
    rag = None


# 4. Define Routes

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Renders the chat page. On POST, processes the user’s question and returns the answer.
    """
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            flash("Please enter a question.", "error")
            return redirect(url_for("index"))

        if rag is None:
            flash("RAG model failed to initialize. Check server logs.", "error")
            return redirect(url_for("index"))

        # 4a. Retrieve contexts
        contexts = rag.get_retrieved_contexts(question)

        # 4b. Call OpenAI chat completion with contexts
        try:
            answer = rag.ask_chat_completion(question, contexts)
        except Exception as api_err:
            flash(f"OpenAI API error: {api_err}", "error")
            return redirect(url_for("index"))

        # 4c. Render template with question & answer
        return render_template("index.html", question=question, contexts=contexts, answer=answer)

    # GET: just render empty chat form
    return render_template("index.html", question=None, contexts=None, answer=None)


# 5. Run the app if executed directly
if __name__ == "__main__":
    # Flask’s built-in debug server. In CMD: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
