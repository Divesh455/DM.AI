"""
Global constants shared across the DM.AI backend.
Keeps static metadata and default values out of the service and API layers.
"""
from pathlib import Path


API_V1_STR: str = "/api/v1"
PROJECT_NAME: str = "DM.AI Backend"
APP_VERSION: str = "1.0.0"
BASE_DIR: Path = Path(__file__).resolve().parents[2]

# RAG Defaults
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
KNOWLEDGE_SUPPORTED_EXTENSIONS: set[str] = {".md", ".txt", ".pdf"}
VECTOR_STORE_INDEX_FILES: tuple[str, str] = ("index.faiss", "index.pkl")
RESUME_FILE_NAME: str = "resume.pdf"

# LLM Defaults
DEFAULT_LLM_TEMPERATURE: float = 0.0
DEFAULT_LLM_MAX_OUTPUT_TOKENS: int = 256
DEFAULT_LLM_TIMEOUT_SECONDS: int = 30
DEFAULT_LLM_MAX_RETRIES: int = 2

# System Limits
MAX_CHAT_HISTORY: int = 10
MAX_DOCUMENT_SIZE_MB: int = 10
