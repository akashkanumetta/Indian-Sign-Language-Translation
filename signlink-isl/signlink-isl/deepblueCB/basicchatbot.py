import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from mistralai import Mistral

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = r"C:\Users\msdak\Desktop\vscode\Projects\deepblueCB\healthcare.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# -------------------------------
# LOAD MISTRAL AI (API KEY FROM STREAMLIT SECRETS)
# -------------------------------
def load_mistral():
    """
    Initialize Mistral AI client with API key from Streamlit secrets
    """
    try:
        api_key = st.secrets["MISTRAL_API_KEY"]
    except:
        st.error("MISTRAL_API_KEY not found in secrets. Please add it to .streamlit/secrets.toml")
        st.stop()
    
    # Initialize Mistral client
    client = Mistral(api_key=api_key)
    
    return client

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data():
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def create_documents(data):
    docs = []
    for item in data:
        docs.append(
            f"Disease: {item['name']}\n"
            f"Symptoms: {', '.join(item['symptoms'])}\n"
            f"Home Treatment: {', '.join(item['treatment'])}"
        )
    return docs

# -------------------------------
# VECTOR STORE
# -------------------------------
def build_vector_store(docs):
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(docs)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, embedder

# -------------------------------
# RETRIEVAL
# -------------------------------
def retrieve_context(query, index, embedder, docs, k=3):
    q_emb = embedder.encode([query])
    _, idx = index.search(np.array(q_emb), k)
    return "\n\n".join([docs[i] for i in idx[0]])

# -------------------------------
# CHATBOT (RAG) WITH MISTRAL AI
# -------------------------------
def chatbot_response(query, index, embedder, docs, model_name="mistral-large-latest"):
    client = load_mistral()
    context = retrieve_context(query, index, embedder, docs)

    messages = [
        {
            "role": "system",
            "content": "You are a healthcare assistant. Answer ONLY using the given context. If the issue seems serious, advise consulting a doctor."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]

    try:
        # Using Mistral AI chat completion
        response = client.chat.complete(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        error_msg = str(e)
        return f"Error calling Mistral API: {error_msg}"

# -------------------------------
# STREAMLIT UI (No Sidebar)
# -------------------------------
def main():
    st.title("🏥 Healthcare Assistant Chatbot")
    
    # Simple model selection in main area (compact)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Ask questions about symptoms and treatments based on our healthcare database.")
    with col2:
        MISTRAL_MODELS = {
            "mistral-large-latest": "Mistral Large",
            "mistral-medium-latest": "Mistral Medium",
            "mistral-small-latest": "Mistral Small",
            "open-mistral-nemo": "Mistral Nemo"
        }
        
        selected_model = st.selectbox(
            "Model",
            options=list(MISTRAL_MODELS.keys()),
            format_func=lambda x: MISTRAL_MODELS[x],
            index=0,
            label_visibility="collapsed"
        )
    
    st.divider()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        with st.spinner("📚 Loading healthcare database..."):
            try:
                data = load_data()
                docs = create_documents(data)
                index, embedder = build_vector_store(docs)
                
                st.session_state.vector_store = {
                    "index": index,
                    "embedder": embedder,
                    "docs": docs
                }
                st.success(f"✅ Loaded {len(data)} health conditions!")
            except Exception as e:
                st.error(f"Error building vector store: {str(e)}")
                return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about symptoms or treatments..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"🤔 Thinking..."):
                try:
                    response = chatbot_response(
                        prompt,
                        st.session_state.vector_store["index"],
                        st.session_state.vector_store["embedder"],
                        st.session_state.vector_store["docs"],
                        model_name=selected_model
                    )
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Disclaimer at the bottom (always visible)
    st.divider()
    st.caption("⚠️ **Disclaimer**: This assistant provides information only from our healthcare database. It is not a substitute for professional medical advice. Always consult with a healthcare professional for medical concerns.")

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    main()