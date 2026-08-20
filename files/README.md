# AI PDF Chatbot (RAG)

A Streamlit app that lets you upload PDF documents and ask questions about them in plain English, built using a manual Retrieval-Augmented Generation (RAG) pipeline — no LangChain.

## How it works

1. **Extract** — pulls text out of uploaded PDFs using PyMuPDF.
2. **Clean & split** — normalizes whitespace and breaks text into sentences with NLTK.
3. **Semantic chunking** — groups consecutive sentences into chunks based on embedding similarity (Sentence-Transformers `all-MiniLM-L6-v2`).
4. **Index** — embeds each chunk and stores it in a FAISS vector index.
5. **Retrieve** — embeds the user's question and finds the most relevant chunks.
6. **Generate** — sends the retrieved chunks + question to Google's `gemini-2.0-flash` to produce a grounded answer.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Get a key from [aistudio.google.com](https://aistudio.google.com/apikey).

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints, upload one or more PDFs, and ask a question.

## Notes

- Embeddings are normalized so cosine similarity (used for chunking) stays consistent with the FAISS index (used for retrieval).
- Index building is cached with `st.cache_resource`, so re-asking questions doesn't re-process the PDFs each time.
