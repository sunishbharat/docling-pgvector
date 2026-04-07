# batch_processor_test
#
# Recursively scans a directory for PDF files and processes each with DocumentProcessor.
#
# To run directly:
# > uv run python -m test.batch_processor_test ./data/
#
# To run via pytest:
# > uv run pytest test/batch_processor_test.py -v -s
#
import sys
import logging
from os import PathLike
from pathlib import Path

# Ensure src/ is importable when run directly (pytest uses pyproject.toml pythonpath)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from collections.abc import Generator

import pytest

from document_processor import DocumentProcessor
from dconfig import EmbeddingsConfig
from pgvector_client import PGVectorClient
from test.docling_test import get_pgConfig_env, test_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
PAGE_CHUNKS = 50
DEFAULT_DATA_DIR = "./data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scan_pdfs(directory: str | PathLike) -> Generator[Path, None, None]:
    """Yield all PDF files found recursively under *directory*."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    for pdf in sorted(root.rglob("*.pdf")):
        logger.info(f"Found PDF: {pdf}")
        yield pdf


def _setup_table(processor: DocumentProcessor, dims: int) -> None:
    """Drop and recreate the items table before a batch run."""
    with PGVectorClient(get_pgConfig_env()) as client:
        with client.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS items")
            cur.execute(
                f"CREATE TABLE items "
                f"(id bigserial PRIMARY KEY, filename TEXT, text TEXT, embedding vector({dims}))"
            )
    logger.info("items table recreated (dims=%d)", dims)


def process_pdf(
    path: Path,
    processor: DocumentProcessor,
) -> int:
    """Process a single PDF: extract chunks, encode, insert into DB. Returns chunk count."""
    content_list, model = processor.embeddings_generate(path=path, page_chunks=PAGE_CHUNKS)
    if not content_list:
        logger.warning("No content extracted from %s", path)
        return 0

    embeddings = model.encode(content_list)

    with PGVectorClient(get_pgConfig_env()) as client:
        for text, embed in zip(content_list, embeddings):
            with client.cursor() as cur:
                cur.execute(
                    "INSERT INTO items (filename, text, embedding) VALUES (%s, %s, %s)",
                    (path.name, text, embed),
                )

    logger.info("Inserted %d chunks from %s", len(content_list), path.name)
    return len(content_list)


def batch_process_directory(directory: str | PathLike) -> dict[str, int]:
    """
    Process all PDFs found recursively in *directory*.

    Instantiates DocumentProcessor once (shared across all files) and returns
    a mapping of {filename: chunk_count}.
    """
    embedconfig = EmbeddingsConfig(model_name=MODEL_NAME)
    processor = DocumentProcessor(embedconfig=embedconfig)

    # Set up a clean table using the resolved embedding dimension
    dims = processor._embedconfig.dims
    _setup_table(processor, dims)

    results: dict[str, int] = {}
    for pdf in scan_pdfs(directory):
        try:
            count = process_pdf(pdf, processor)
            results[pdf.name] = count
        except Exception:
            logger.exception("Failed to process %s — skipping", pdf)
            results[pdf.name] = 0

    logger.info("Batch complete: %d file(s) processed", len(results))
    return results


# ---------------------------------------------------------------------------
# Pytest fixtures & tests
# ---------------------------------------------------------------------------

@pytest.fixture
def pdf_directory():
    path = Path(DEFAULT_DATA_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_pdfs_found(pdf_directory):
    pdfs = list(scan_pdfs(pdf_directory))
    assert len(pdfs) > 0, f"No PDFs found in {pdf_directory}. Add at least one PDF to run this test."


def test_batch_processing(pdf_directory):
    results = batch_process_directory(pdf_directory)

    assert len(results) > 0, "No files were processed"
    for filename, count in results.items():
        assert count > 0, f"Expected chunks from {filename}, got 0"


def test_similarity_search_after_batch(pdf_directory):
    """After batch ingestion, verify similarity search returns results."""
    batch_process_directory(pdf_directory)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    records = test_embeddings(query="introduction", model=model)
    assert len(records) > 0


# ---------------------------------------------------------------------------
# __main__ — direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    logger.info("Scanning directory: %s", directory)

    results = batch_process_directory(directory)

    print("\n--- Batch Processing Summary ---")
    for filename, count in results.items():
        print(f"  {filename}: {count} chunk(s)")
    print(f"Total files: {len(results)}")
    print(f"Total chunks: {sum(results.values())}")
