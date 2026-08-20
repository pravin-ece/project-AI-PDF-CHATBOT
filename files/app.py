
import os
import streamlit as st
import fitz
import nltk
import faiss
 
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
 
load_dotenv()
 
st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 AI PDF Chatbot")
st.write("Ask questions about your uploaded PDF documents using RAG.")
 
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop()
 
client = genai.Client(api_key=api_key)
 
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
 
 
def extract_text_from_pdfs(uploaded_files):
    """Read each uploaded PDF with PyMuPDF and pull out its text."""
    documents = []
    for uploaded_file in uploaded_files:
        pdf_data = uploaded_file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
 
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
 
        documents.append({"name": uploaded_file.name, "text": text})
    return documents
 
 
def clean_and_split(documents):
    """Collapse whitespace and break each document into sentences."""
    all_sentences = []
    for document in documents:
        text = " ".join(document["text"].split())
 
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
 
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                all_sentences.append(sentence)
 
    return all_sentences
 
 
def semantic_chunk(sentences, model, similarity_threshold=0.55):
    """Group consecutive sentences into chunks when they stay on the same topic."""
    if not sentences:
        return []
 
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
 
    chunks = []
    current_chunk = [sentences[0]]
 
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(
            sentence_embeddings[i - 1].reshape(1, -1),
            sentence_embeddings[i].reshape(1, -1),
        )[0][0]
 
        if similarity >= similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
 
    if current_chunk:
        chunks.append(" ".join(current_chunk))
 
    return chunks
 
 
def create_chunk_embeddings(chunks, model):
    return model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
 
 
def build_faiss_index(embedded_chunks):
    dim = embedded_chunks.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedded_chunks.astype("float32"))
    return index
 
 
def retrieve_chunks(query, model, index, chunks, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
 
    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1:
            retrieved_chunks.append(chunks[idx])
 
    return retrieved_chunks
 
 
def generate_answer(query, retrieved_chunks, client):
    context = "\n\n".join(retrieved_chunks)
 
    prompt = f"""You are an AI assistant that answers questions from uploaded PDF documents.
Use only the information provided in the context. Do not invent information.
 
If the answer is not available in the context, say:
"Data not available in the uploaded documents."
 
Context:
{context}
 
Question:
{query}
 
Answer:
"""
 
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
 
    return response.text
 
 
def rag_pipeline(query, embed_model, index, chunks, client):
    retrieved_chunks = retrieve_chunks(query, embed_model, index, chunks, top_k=3)
    answer = generate_answer(query, retrieved_chunks, client)
    return answer, retrieved_chunks
 
 
@st.cache_resource(show_spinner="Processing PDFs and building index...")
def build_index(files_bytes):
    """files_bytes is a tuple of (filename, bytes) so Streamlit can cache on it."""
 
    class _UploadedFileLike:
        def __init__(self, name, data):
            self.name = name
            self._data = data
 
        def read(self):
            return self._data
 
    files = [_UploadedFileLike(name, data) for name, data in files_bytes]
 
    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = extract_text_from_pdfs(files)
    all_sentences = clean_and_split(documents)
    chunks = semantic_chunk(all_sentences, model)
    embedded_chunks = create_chunk_embeddings(chunks, model)
    index = build_faiss_index(embedded_chunks)
 
    return index, chunks, all_sentences, model
 
 
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
 
if uploaded_files:
    files_bytes = tuple((f.name, f.getvalue()) for f in uploaded_files)
    index, chunks, all_sentences, model = build_index(files_bytes)
 
    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} document(s).")
    st.write(f"Total sentences: {len(all_sentences)}")
    st.write(f"Total chunks: {len(chunks)}")
 
    query = st.text_input("Ask a question about your PDFs")
 
    if st.button("Ask") and query:
        with st.spinner("Thinking..."):
            answer, retrieved_chunks = rag_pipeline(query, model, index, chunks, client)
 
        st.subheader("Answer")
        st.write(answer)
 
        with st.expander("Retrieved Context"):
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)