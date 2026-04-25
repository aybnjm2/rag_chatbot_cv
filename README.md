# 📄 Chatbot CV Interactif (RAG)

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain and Google Gemini that answers recruiter questions based exclusively on CV content. The system ensures accurate, contextual responses by retrieving relevant information from uploaded PDFs and generating answers only from that context.

## 🎯 Features

- **PDF CV Upload**: Seamlessly upload and process CV PDFs
- **Smart Document Processing**: Automatic text extraction, cleaning, and chunking
- **Vector Storage**: Uses ChromaDB to store and retrieve CV content via semantic similarity
- **Context-Aware Responses**: Generates answers using Google Gemini, strictly limited to CV information
- **Interactive Chat Interface**: User-friendly Streamlit web interface
- **Session Management**: Maintains conversation history and dynamically reloads vector stores when new CVs are uploaded
- **Security-First Approach**: Prevents hallucinations by refusing to answer questions outside CV scope

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- Google API key for Gemini access
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd devoir_rag_chatbot_st
   ```

2. **Install dependencies**
   
   Using UV (recommended):
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   
   Add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

### Running the Application

Start the Streamlit app:
```bash
streamlit run rag_cv_chatbot.py
```

The application will open in your browser at `http://localhost:8501`

### Usage Example

1. **Upload CV**: Click "Chargez votre CV" in the sidebar and select a PDF file
2. **Wait for Processing**: The system will analyze the CV and build the vector database (this takes a few seconds)
3. **Ask Questions**: Type your question in the chat input, e.g.:
   - "Quelles sont les compétences en Python ?"
   - "Décrivez votre expérience en développement web"
   - "Quels sont vos diplômes ?"
4. **View Answers**: The chatbot responds based exclusively on CV content

## 🏗️ Project Structure

```
devoir_rag_chatbot_st/
├── rag_cv_chatbot.py          # Main Streamlit application
├── pyproject.toml             # Project dependencies
├── .env.example               # Environment variables template
├── chroma_cv_store/           # ChromaDB vector storage (auto-generated)
├── testing/                   # Testing notebooks
└── README.md                  # This file
```

## 🔧 How It Works

### Architecture Overview

1. **PDF Loading**: Extracts text from uploaded PDF CVs using PyPDFLoader
2. **Text Preprocessing**: Cleans and normalizes text (removes extra whitespace, redundant newlines)
3. **Chunking**: Splits text into overlapping chunks (300 tokens, 20-token overlap) using RecursiveCharacterTextSplitter
4. **Embedding**: Generates vector embeddings using Google's Gemini embedding model
5. **Vector Storage**: Stores embeddings in ChromaDB for semantic search
6. **Retrieval**: Uses similarity search to find the 4 most relevant chunks for each question
7. **Answer Generation**: Passes retrieved context to Gemini to generate professional responses

### Key Components

- **embedding_model**: Google Gemini embedding model for vectorizing text
- **llm**: Google Gemini 2.5 Flash for answer generation
- **vectorstore**: ChromaDB collection for persistent vector storage
- **retriever**: Similarity-based search to find relevant CV sections

## 📋 Dependencies

- **streamlit**: Web UI framework
- **langchain**: LLM orchestration and document processing
- **langchain-google-genai**: Google Gemini integration
- **chromadb**: Vector database for semantic search
- **pypdf**: PDF text extraction
- **tiktoken**: Token counting for precise chunking
- **python-dotenv**: Environment variable management

See `pyproject.toml` for complete dependency list and versions.

## 🔐 Security Features

- **Context Constraint**: Responses are strictly limited to CV content
- **Hallucination Prevention**: Built-in safety rules prevent the chatbot from using general knowledge
- **Professional Output**: Ensures courteous and accurate responses
- **Input Validation**: Processes only authorized PDF files

## 🎓 Development

### Testing

Testing notebooks are available in the `testing/` directory for development and experimentation:
```bash
jupyter notebook testing/rag_chatbot\(testing\).ipynb
```

### Configuration

Modify these constants in `rag_cv_chatbot.py` to adjust behavior:
- **chunk_size**: Token count per document chunk (default: 300)
- **chunk_overlap**: Overlap between chunks (default: 20)
- **k**: Number of retrieved chunks for context (default: 4)

## 📝 Notes

- Vector embeddings are stored persistently in `./chroma_cv_store/`
- Chat history resets when a new CV is uploaded
- Supports French language queries and responses
- Requires active internet connection for Google Gemini API access

## 🤝 Contributing

Contributions are welcome! Please ensure your changes:
- Maintain the security-first approach (no hallucination)
- Follow Python best practices
- Include appropriate comments for complex logic
- Test with various CV formats

## 📄 License

See LICENSE file for details.

## 📞 Support

For issues, questions, or suggestions:
- Check the testing notebook for usage examples
- Review the code comments for implementation details
- Verify your Google API key has Gemini access

---

Built with ❤️ using LangChain, ChromaDB, and Google Gemini
