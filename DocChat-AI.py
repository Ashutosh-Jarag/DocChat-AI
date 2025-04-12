import fitz
import docx
import streamlit as st
from langchain.llms import GooglePalm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def display_chat_history():
    # Add some spacing at the top
    st.write("")
    
    # Create a container for the chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Add some spacing at the bottom
        st.write("")

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# Initialize Hugging Face embeddings model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def extract_text_from_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()

def extract_text_from_word(file_path):
    loader = Docx2txtLoader(file_path)
    return loader.load()

def main():
    st.set_page_config(page_title="Document Chat Assistant", layout="wide")
    initialize_session_state()
    
    # Custom CSS for better chat styling
    st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .stChatMessage[data-type="user"] {
            background-color: #f0f2f6;
        }
        .stChatMessage[data-type="assistant"] {
            background-color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 Document Chat Assistant")

    # Sidebar configuration
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF or Word file", type=["pdf", "docx"])
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        if uploaded_file is not None:
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            
            if uploaded_file.name.endswith(".pdf"):
                documents = extract_text_from_pdf(file_path)
            else:
                documents = extract_text_from_word(file_path)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents([Document(page_content=doc.page_content) for doc in documents])
            
            vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state.retriever = vectorstore.as_retriever()
            st.success("✅ Document processed successfully!")
            
            # Clear chat history when new document is uploaded
            st.session_state.messages = []

    # Display chat interface
    display_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask a question about your document...", key="chat_input"):
        if st.session_state.retriever is None:
            st.error("⚠️ Please upload a document first!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    qa_chain = RetrievalQA.from_chain_type(
                        llm, 
                        retriever=st.session_state.retriever,
                        chain_type="stuff"
                    )
                    response = qa_chain.run(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Update chat history
                    st.session_state.chat_history.append({
                        "question": prompt,
                        "answer": response
                    })

if __name__ == "__main__":
    main()
