import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

 

 
def load_system():
    try:
        # Load knowledge base
        with open("knowledge_base.pkl", "rb") as f:
            knowledge_base = pickle.load(f)

        # Load FAISS index
        index = faiss.read_index("ecommerce_index.faiss")

        # Load model
        with open("model_name.txt", "r") as f:
            model_name = f.read().strip()
        model = SentenceTransformer(model_name)

        return model, knowledge_base, index
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None, None, None

 
# Get best answer
 
def get_answer(query, model, knowledge_base, index, top_k=1):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(np.array(query_embedding).astype("float32"), top_k)

    best_match = knowledge_base[indices[0][0]]
    return best_match
 
# Streamlit Chat UI
 
def main():
    bot_name = "ShopEase Assistant"

    st.set_page_config(page_title=f"{bot_name} Chat", page_icon="🛍️")
    st.title(f"🛍️ {bot_name}")
    st.caption("Hi! I’m your friendly shopping support bot. Ask me anything 😊")

    # Load system
    model, knowledge_base, index = load_system()
    if not all([model, knowledge_base, index]):
        st.stop()

    # Sidebar branding
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/891/891462.png", width=120)  # replace with logo.png if you have
        st.markdown("### 🛒 ShopEase Support")
        st.write("Available 24/7 to help with orders, returns, and account questions.")
        st.markdown("---")
        st.write("Built with 💙 using Streamlit + HuggingFace")

    # Initialize chat session
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"👋 Hello! I’m {bot_name}. How can I help you today?"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**You:** {message['content']}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**{bot_name}:** {message['content']}")

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**You:** {prompt}")

        # Generate answer
        with st.chat_message("assistant", avatar="🤖"):
            response = get_answer(prompt, model, knowledge_base, index)
            st.markdown(f"**{bot_name}:** {response['answer']}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"]
        })

 
if __name__ == "__main__":
    main()
