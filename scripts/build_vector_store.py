"""
Local utility for generating the FAISS vector store before deployment.
Use this on your machine, commit the generated files, then deploy the repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag_service import RAGService


def parse_args() -> argparse.Namespace:
    """
    Parses command-line flags for rebuilding or reusing the local index.
    """
    parser = argparse.ArgumentParser(
        description="Build the local FAISS vector store from data/knowledge."
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip rebuilding when a persisted vector store already exists.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Runs the ingestion workflow and prints a JSON summary for deployment logs.
    """
    args = parse_args()
    service = RAGService()

    try:
        summary = service.ingest_knowledge_base(rebuild_index=not args.skip_if_exists)
    except Exception as exc:  # pragma: no cover - CLI failure path
        print(f"Vector store build failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
