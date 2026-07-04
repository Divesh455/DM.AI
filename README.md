# DM.AI Backend

Production-ready FastAPI backend for **DM.AI**, a portfolio assistant that answers questions about Divesh Matkar using **RAG + Google Gemini**.

## What This Backend Does

- Loads portfolio knowledge from `data/knowledge/`
- Splits documents into chunks with `RecursiveCharacterTextSplitter`
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stores embeddings in FAISS
- Retrieves top matching chunks for each user question
- Sends only retrieved portfolio context to Gemini
- Returns concise answers without hallucinating beyond the knowledge base

## Tech Stack

- Python 3.12
- FastAPI
- LangChain
- Google Gemini
- FAISS
- Sentence Transformers
- Pydantic v2
- Loguru

## Project Structure

```text
DM.AI/
├── app/
│   ├── api/v1/          # API routes
│   ├── core/            # config, constants, prompts, logger, exceptions
│   ├── models/schemas/  # request and response schemas
│   ├── services/        # chat, llm, and rag business logic
│   ├── rag/             # loader, splitter, embeddings, vector store, retriever
│   ├── utils/           # middleware and helpers
│   └── main.py          # FastAPI entrypoint
├── data/
│   ├── knowledge/       # markdown, txt, pdf knowledge files
│   ├── vector_store/    # generated FAISS files
│   └── uploads/         # reserved for future use
├── scripts/
│   └── build_vector_store.py
├── tests/
├── Procfile
├── render.yaml
├── runtime.txt
└── requirements.txt
```

## API Endpoints

- `GET /`
- `GET /api/v1/health`
- `POST /api/v1/chat`
- `POST /api/v1/ingest`
- `GET /api/v1/resume`

## Local Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your Gemini key.

Important variables:

- `GEMINI_API_KEY`
- `LLM_PROVIDER=gemini`
- `KNOWLEDGE_DIR=data/knowledge`
- `VECTOR_STORE_DIR=data/vector_store`
- `INGEST_API_ENABLED=true` for local development

### 4. Build the vector store

Run this before testing chat locally for the first time:

```bash
python scripts/build_vector_store.py
```

If you only want to build when the index does not already exist:

```bash
python scripts/build_vector_store.py --skip-if-exists
```

### 5. Start the backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger docs are available at:

```text
http://localhost:8000/docs
```

## Chat Request Example

```json
{
  "message": "Who is Divesh?"
}
```

## Chat Response Example

```json
{
  "success": true,
  "message": "Response generated successfully.",
  "data": {
    "answer": "..."
  }
}
```

## Free Deployment on Render

This project is arranged to work well on **Render Free**.

### Important free-tier rule

Render free web services can sleep after inactivity, and local runtime file changes are not reliable for long-term persistence. Because of that:

- build the FAISS vector store locally
- commit `data/vector_store/index.faiss`
- commit `data/vector_store/index.pkl`
- deploy those files with the repo
- keep runtime ingestion disabled in production

### Deployment Steps

1. Make sure your knowledge files are final inside `data/knowledge/`
2. Build the vector store locally:

```bash
python scripts/build_vector_store.py
```

3. Confirm these files exist:

```text
data/vector_store/index.faiss
data/vector_store/index.pkl
```

4. Commit and push your repository
5. Create a new Render web service from the repo
6. Render will use:
   - `render.yaml`
   - `Procfile`
   - `runtime.txt`
7. Add environment variables in Render:
   - `GEMINI_API_KEY`
   - `CORS_ORIGINS`

### Production behavior

- `INGEST_API_ENABLED=false` in `render.yaml`
- `/api/v1/chat` uses the committed FAISS index
- `/api/v1/health` now shows whether the vector store is ready

## Recommended Render Environment Variables

- `APP_ENV=production`
- `LOG_LEVEL=INFO`
- `LLM_PROVIDER=gemini`
- `GEMINI_MODEL=gemini-2.5-flash`
- `KNOWLEDGE_DIR=data/knowledge`
- `VECTOR_STORE_DIR=data/vector_store`
- `INGEST_API_ENABLED=false`
- `CORS_ORIGINS=https://your-frontend-domain.com`
- `GEMINI_API_KEY=your_real_key`

## Testing

Run tests with:

```bash
pytest
```

If you only want a quick syntax check:

```bash
python -m compileall app tests scripts
```

## Notes for Portfolio Use

- The chatbot answers only from your portfolio documents
- If a fact is not present in the knowledge base, it should say it is not available
- To update knowledge, edit files in `data/knowledge/`, rebuild the vector store locally, and redeploy
- For free hosting, do not depend on runtime `/api/v1/ingest` as your long-term indexing workflow
