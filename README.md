# Document Chat Assistant

## Project Overview
In this project, I created a Streamlit web application that lets users upload PDF or Word documents and ask questions about their content. I worked on extracting text from these documents, splitting it into manageable chunks, and embedding it into a FAISS vector store for efficient retrieval. Using Google Gemini as the language model and Hugging Face embeddings, I built a retrieval-augmented generation (RAG) system to provide accurate answers based on the document. The app maintains a chat history and includes a clean, user-friendly interface for interactive querying.

## Tools and Libraries Used
- **Python 3.8+**: Powers the entire application and data processing
- **Streamlit**: Creates the interactive web interface for document uploads and chat
- **PyMuPDF (fitz)**: Extracts text from PDF files
- **python-docx & Docx2txt**: Handles text extraction from Word documents
- **LangChain**: Manages document loading, text splitting, embeddings, and retrieval
- **FAISS**: Stores document embeddings for fast similarity search
- **HuggingFaceEmbeddings**: Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- **Google Gemini (via ChatGoogleGenerativeAI)**: Provides the LLM for generating answers
- **Vscode**: Likely used for initial experimentation (optional)

## Key Features
- Upload and process PDF or Word documents to extract text
- Split documents into chunks and embed them in a FAISS vector store
- Answer user questions about the document using a RAG pipeline
- Display chat history in a styled, user-friendly Streamlit interface
- Clear chat history option to start fresh with new documents

## How to Set It Up
1. Clone this repo: `git clone https://github.com/Ashutosh-Jarag/DocChat-AI.git`
2. Install required packages: `pip install -r project-requirements.txt`
3. Set up your Google and Hugging Face API key for Gemini:
   - Create a `.env` file or set `GOOGLE_API_KEY` and 'HUGGINGFACE_API_KEY' as an environment variable
4. Run the Streamlit app: `streamlit run DocChat-AI.py`

## How to Use It
- Launch the app by running `streamlit run DocChat-AI.py`
- Open your browser at `http://localhost:8501`
- In the sidebar, upload a PDF or Word document
- Ask questions in the chat input box to get answers based on the document
- Use the "Clear Chat History" button to reset the conversation
- Check out the chat history displayed above the input box

## Want to Contribute?
Feel free to fork the repo and submit a Pull Request with your ideas or improvements!

## License
MIT License
