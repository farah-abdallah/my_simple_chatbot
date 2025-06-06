# rag_model.py

import os
import time
import requests
from helper_functions import encode_pdf, retrieve_context_per_question, encode_all_data_folder

class SimpleRAG:
    """
    A class to handle PDF chunking, embedding, and retrieval.
    """
    def __init__(self, folder_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, n_retrieved: int = 2, model: str="gpt-4"):
        """
        - pdf_path: path to local PDF to index.
        - chunk_size, chunk_overlap: control how large each chunk is and how much overlap.
        - n_retrieved: how many top chunks to return per query.
        """
        print("\n--- Initializing Simple RAG Retriever ---")
        start_time = time.time()

        # Step 1: Encode the PDF with embeddings + FAISS
        # New “load all supported file types in data/”
        self.vector_store = encode_all_data_folder(folder_path, chunk_size=1000, chunk_overlap=200)

        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Chunking Time: {self.time_records['Chunking']:.2f} seconds")

        # Step 2: Build a retriever that returns the top `k` most similar chunks
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def get_retrieved_contexts(self, query: str):
        """
        Returns a list of text chunks (strings) relevant to `query`.
        """
        retrieval_start = time.time()
        contexts = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - retrieval_start
        print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")
        return contexts

    def ask_chat_completion(self, query: str, contexts: list[str]) -> str:
        """
        Calls OpenAI Chat Completion endpoint with:
        - system prompt that instructs the model to answer based on the contexts.
        - user prompt that includes the original query.
        Uses the REST API via `requests`.
        Returns the answer text.
        """

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")

        # 1. Build a single “context” blob by concatenating the top chunks.
        #    You could also format each chunk separately. Here, we simply join them with separators.
        combined_context = "\n\n---\n\n".join(contexts)

        # 2. Build the messages payload for chat completion.
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer the user’s question using ONLY the provided context. "
                "If the answer is not in the context, say \"I do not know.\""
            )
        }
        user_message = {
            "role": "user",
            "content": f"Context:\n{combined_context}\n\nQuestion: {query}"
        }

        # 3. Prepare HTTP request to OpenAI REST endpoint
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        json_payload = {
            "model": "gpt-4o",         # or "gpt-4-turbo-preview" or "gpt-3.5-turbo"
            "messages": [system_message, user_message],
            "temperature": 0.0,
            "max_tokens": 512
        }

        response = requests.post(url, headers=headers, json=json_payload)
        response.raise_for_status()  # Raise an exception if we get an error (e.g. rate limit, bad request)
        data = response.json()

        # 4. Extract the assistant’s reply
        answer = data["choices"][0]["message"]["content"].strip()
        return answer
