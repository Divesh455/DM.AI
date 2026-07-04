"""
Document loading layer.
Responsible for reading portfolio documents from disk and converting them into LangChain Documents.
"""
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger
import pypdf

from app.core.constants import KNOWLEDGE_SUPPORTED_EXTENSIONS
from app.core.exceptions import DocumentParsingException


class DocumentLoader:
    """
    Unified loader for supported knowledge-base file types.
    """

    def __init__(self, supported_extensions: set[str] | None = None):
        """
        Configures the loader with the allowed file extensions.
        """
        self._supported_extensions = supported_extensions or KNOWLEDGE_SUPPORTED_EXTENSIONS

    def load_file(self, file_path: str | Path) -> list[Document]:
        """
        Loads a single knowledge file into one or more LangChain documents.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            logger.error("Document loading failed because the file does not exist: {}", path)
            raise DocumentParsingException(f"File not found: {path}")

        extension = path.suffix.lower()
        if extension not in self._supported_extensions:
            raise DocumentParsingException(f"Unsupported file extension: {extension}")

        logger.info("Loading knowledge document: {}", path.name)

        if extension in {".md", ".txt"}:
            return self._load_text_file(path)
        if extension == ".pdf":
            return self._load_pdf_file(path)

        raise DocumentParsingException(f"Unsupported file extension: {extension}")

    def load_directory(self, directory_path: str | Path) -> list[Document]:
        """
        Loads every supported file from the configured knowledge-base directory.
        """
        directory = Path(directory_path).resolve()
        if not directory.is_dir():
            logger.error("Knowledge directory was not found: {}", directory)
            raise DocumentParsingException(f"Directory not found: {directory}")

        documents: list[Document] = []
        candidates = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in self._supported_extensions
        )

        logger.info(
            "Scanning knowledge directory {} and found {} supported files.",
            directory,
            len(candidates),
        )

        for path in candidates:
            documents.extend(self.load_file(path))

        logger.info("Loaded {} raw documents from the knowledge base.", len(documents))
        return documents

    def _load_text_file(self, file_path: Path) -> list[Document]:
        """
        Reads markdown and text files into a single document entry.
        """
        try:
            content = file_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.exception("Failed to read text file {}: {}", file_path, exc)
            raise DocumentParsingException(f"Failed to read text file: {file_path.name}") from exc

        if not content:
            logger.warning("Skipping empty text document: {}", file_path.name)
            return []

        return [
            Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.lstrip("."),
                },
            )
        ]

    def _load_pdf_file(self, file_path: Path) -> list[Document]:
        """
        Extracts text from each non-empty PDF page and stores page metadata.
        """
        try:
            reader = pypdf.PdfReader(str(file_path))
        except Exception as exc:
            logger.exception("Failed to open PDF file {}: {}", file_path, exc)
            raise DocumentParsingException(f"Failed to open PDF file: {file_path.name}") from exc

        documents: list[Document] = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                logger.debug("Skipping empty PDF page {} from {}", page_index, file_path.name)
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "file_type": "pdf",
                        "page": page_index,
                    },
                )
            )

        if not documents:
            raise DocumentParsingException(f"No extractable text found in PDF: {file_path.name}")

        return documents
