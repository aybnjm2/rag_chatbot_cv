# RAG Chatbot CV

A Python application that lets a user upload a CV in PDF format and ask questions in natural language. The app uses a retrieval-augmented generation (RAG) pipeline to find the most relevant passages in the CV and answer only from that source material.

This project is designed to be a privacy-conscious, context-grounded assistant for recruiter-style questions such as:

- "Quelles sont les compétences Python du candidat ?"
- "Décrivez son expérience en développement web."
- "Quels diplômes et certifications sont mentionnés ?"

The system is intentionally conservative: if the CV does not contain the answer, it refuses to answer instead of inventing information.

Quick links: [app.py](app.py), [rag.py](rag.py), [pyproject.toml](pyproject.toml), [.env.example](.env.example), [eval/run_eval.py](eval/run_eval.py)

## Why this project is useful

- Upload a PDF CV and index it in seconds
- Use semantic search to retrieve the most relevant sections
- Ask questions in plain French or English
- Keep answers grounded in the uploaded document only
- Use the same logic for demos, personal portfolio tools, or recruiter workflows

### Key features

- PDF text extraction with PyPDF
- Text cleaning and chunking for retrieval
- Vector storage with ChromaDB
- Embeddings and generation with Google Gemini
- Streamlit-based chat interface
- Evaluation scripts to benchmark retrieval and answer quality

## Project structure

```text
.
├── app.py                    # Main Streamlit app entrypoint
├── rag.py                    # Core RAG logic: loading, indexing, retrieval, answer generation
├── rag_cv_chatbot.py         # Earlier prototype / reference implementation
├── pyproject.toml            # Python project configuration and dependencies
├── requirements.txt          # Pip install dependencies
├── .env.example              # Sample environment file
├── README.md                 # Project documentation
├── eval/
│   ├── compare_configs.py    # Compare chunking/retrieval configurations
│   ├── run_eval.py           # Run evaluation against a question set
│   ├── questions.csv        # Evaluation questions
│   ├── questions.example.csv
│   ├── results.csv           # Evaluation results
│   └── cvs/
│       └── CV.pdf           # Example CV used for evaluation
└── .venv/                    # Local virtual environment (generated)
```

## Requirements

- Python 3.12+
- A Google API key with access to Gemini
- `uv` (recommended) or `pip`
- Internet access for Gemini API calls

## Getting started

### 1) Clone the repository

```bash
git clone <repository-url>
cd rag_chatbot_cv
```

### 2) Install dependencies

Using `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then add your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Run the app

Start the Streamlit app from the project root:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, typically:

```text
http://localhost:8501
```

### Typical usage

1. Upload a CV PDF in the sidebar.
2. Wait while the app processes the PDF and builds a vector store.
3. Ask a question about the CV in the chat box.
4. Read the answer, which is generated only from the uploaded CV context.

Example prompts:

```text
Quelles sont les compétences en Python ?
Quel est le parcours académique ?
Décrivez ses projets récents.
```

## How it works

The app follows a typical RAG pipeline:

1. The uploaded PDF is loaded with `PyPDFLoader`.
2. The raw text is cleaned and normalized.
3. The text is split into overlapping chunks.
4. Chunks are embedded with the Gemini embedding model.
5. Chunks are stored in a ChromaDB vector store.
6. A user question triggers a similarity search.
7. The most relevant passages are sent to Gemini with a strict prompt that forbids answers outside the CV context.

The most important logic lives in [rag.py](rag.py) and is used by the app entrypoint in [app.py](app.py).

## Configuration

The main retrieval settings are defined in [rag.py](rag.py):

- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`
- `MIN_SCORE`

These values control how much text is indexed and how strictly the system filters search results before generating an answer. For experimentation, the evaluation scripts in `eval/` compare several configurations.

## Evaluation

The repository includes evaluation tooling to test retrieval quality and answer correctness.

Run the default evaluation suite:

```bash
python eval/run_eval.py
```

Compare multiple chunking and retrieval settings:

```bash
python eval/compare_configs.py
```

These scripts read the question set in `eval/questions.csv` and generate result files under `eval/`.

## Contributing

Contributions are welcome. If you want to improve the project:

- keep the answer flow grounded in the uploaded CV only
- prefer small, focused pull requests
- test your changes with a representative PDF and a few natural-language questions
- keep the project easy to run locally with a consistent Python setup

For bug reports or improvements, open a GitHub issue or submit a pull request from the repository.

## Getting help

If you run into issues:

- check that `GOOGLE_API_KEY` is present in your `.env`
- make sure your Python environment matches the project requirements
- verify that the uploaded file is a valid PDF
- review the logic in `rag.py` and `app.py` for the exact runtime behavior

## Maintainer and ownership

This project is maintained through the repository's Git history and active contributors. If you want to take ownership of a feature or bug fix, please open a pull request and explain the change clearly.

## License

No license file is included in the repository currently, so the project should be used under the repository owner's default terms unless a license is added later.
