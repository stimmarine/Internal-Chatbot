# Paste the entire Streamlit app code block below this line

import streamlit as st
import os
import faiss # For FaissVectorStore loading
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
#from llama_index.core.stores.docstore import SimpleDocumentStore #stores should be storage
#from llama_index.core.stores.index_store import SimpleIndexStore #stores should be storage
from llama_index.llms.google_genai import GoogleGenAI as GeminiLLM
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding as GeminiEmbedding
import google.generativeai as genai # For API key configuration

# --- Configuration ---
PROJECT_BASE_PATH = '/content/drive/MyDrive/Colab Notebooks/MR_Chatbot' 
VECTOR_STORE_DIR = os.path.join(PROJECT_BASE_PATH, "vectorstore")

# --- LlamaIndex Setup (Cached by Streamlit) ---
@st.cache_resource(show_spinner="Initializing AI Advisor and loading knowledge base...")
def load_and_setup_ai_advisor():
    # 1. Configure Google Gemini API Key
      #    Streamlit apps run in their own process. We rely on the GOOGLE_API_KEY
      #    environment variable being set in the environment where Streamlit runs.
      #    When running from Colab with `!streamlit run`, it *should* inherit.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔴 GOOGLE_API_KEY environment variable not found! Please ensure it's set in your Colab session before running Streamlit.")
        st.stop()
        return None, None
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"🔴 Error configuring Google GenAI with API key: {e}")
        st.stop()
        return None, None

    # 2. Configure LlamaIndex Global Settings (LLM and Embed Model)
    try:
        Settings.llm = GeminiLLM(model="models/gemini-1.5-flash-latest")
        Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
        st.sidebar.success(f"LLM: {Settings.llm.model}\nEmbed: {Settings.embed_model.model_name}")
    except Exception as e:
        st.error(f"🔴 Error configuring LlamaIndex Settings (LLM/Embeddings): {e}")
        st.stop()
        return None, None

    # 3. Load the Persisted Index    
    try:
        vector_store = FaissVectorStore.from_persist_dir(persist_dir=VECTOR_STORE_DIR)            
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=VECTOR_STORE_DIR
        )
        index = load_index_from_storage(storage_context = storage_context)
        # Get a chat engine
        # You can adjust similarity_top_k as needed. Higher means more context, potentially slower/costlier.
        chat_engine_obj = index.as_chat_engine(
            chat_mode="condense_question", 
            verbose=True,
            similarity_top_k=10,
            ) 
        st.success("💡 AI Advisor initialized and knowledge base loaded!")
        return index, chat_engine_obj
    except Exception as e:
        st.error(f"🔴 Error loading persisted index: {e}")
        st.exception(e) # Show full traceback in Streamlit app for debugging
        st.stop()
        return None, None


# --- Streamlit App UI ---
st.set_page_config(page_title="Market AI Advisor", page_icon="💡", layout="wide")
st.title("💡 Market AI Advisor")
st.caption(f"Powered by LlamaIndex and Google Gemini. Knowledge base last updated based on files in {VECTOR_STORE_DIR}")

# Load index and chat engine (cached)
# This function will only run once unless its code changes or cache is cleared.
loaded_index, chat_engine = load_and_setup_ai_advisor()

if loaded_index and chat_engine:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your market research insights today?"}]

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get new user input
    if prompt := st.chat_input("Ask your question..."):
        # Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                response_obj = chat_engine.chat(prompt)
                response_text = str(response_obj)
                
                # Optionally display source nodes
                source_nodes_md = "\n\n---\n**Retrieved Sources:**\n"
                if response_obj.source_nodes:
                    for i, node in enumerate(response_obj.source_nodes):
                        file_name = node.metadata.get('file_name', 'N/A') if node.metadata else 'N/A'
                        source_nodes_md += f"{i+1}. **File:** {file_name} (Score: {node.score:.2f})\n"
                        source_nodes_md += f"   *Snippet:* {node.text[:150].strip().replace(chr(10), ' ')}...\n"
                    response_text += source_nodes_md
                else:
                    response_text += "\n\n(No specific source text segments retrieved for this query)"

                message_placeholder.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                error_message = f"Sorry, an error occurred while processing your question: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("AI Advisor is not ready. Please check for error messages above.")
