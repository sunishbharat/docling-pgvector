
# document_processor
#
# To run in local vscode environment.
# > uv run python -m test.document_processor_test
#
import logging
import pymupdf
from os import PathLike
from pathlib import Path
from itertools import chain
from contextlib import closing
from typing import Iterable
from dataclasses import dataclass
from dconfig import EmbeddingsConfig
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from docling_core.types.doc import DoclingDocument, TableItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvalidModelError(ValueError):
    """Raised when the specified HuggingFace model ID does not exist (HTTP 404)."""
    pass


class DocumentProcessor:
    """Orchestrates PDF parsing, chunking, and embedding generation using Docling and SentenceTransformers.

    Validates the model on HuggingFace Hub before loading it, then initialises a Docling
    DocumentConverter (GPU auto-detected), a HybridChunker, and a SentenceTransformer model.

    Args:
        embedconfig (EmbeddingsConfig | None): Embedding model settings. Defaults to
            EmbeddingsConfig() which uses 'BAAI/bge-base-en-v1.5' with dims=768, batch_size=32.

    Raises:
        InvalidModelError: If the model ID does not exist on HuggingFace Hub.
    """
    def __init__(self, embedconfig: EmbeddingsConfig | None = None):

        self._embedconfig: EmbeddingsConfig = embedconfig or EmbeddingsConfig()
        self._model_name = self._embedconfig.model_name

        # Fail fast: check Hub before downloading the model
        self.check_model_exists()

        # Configure pipeline to choose GPU if available
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        self._model = SentenceTransformer(self._model_name)
        self._initialize()

    def _initialize(self) -> None:
        """Set up the HuggingFace tokenizer, resolve embedding dimensions, and build the HybridChunker.

        Reads the true embedding dimension from the loaded SentenceTransformer and writes it back
        to ``self._embedconfig.dims``. Logs model name, max token count, and resolved dims.
        """
        self._tokenizer: BaseTokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self._model_name),
        )
        self._embedconfig.dims = self._model.get_sentence_embedding_dimension()
        self._chunker = HybridChunker(tokenizer=self._tokenizer)

        logger.info(
            f"model={self._model_name}: "
            f"max_tokens={self._tokenizer.get_max_tokens()} "
            f"embed_dims={self._embedconfig.dims}"
        )

    def check_model_exists(self) -> None:
        """Validate that the configured model ID exists on HuggingFace Hub.

        Queries the Hub API for model metadata. Raises ``InvalidModelError`` on HTTP 404.
        Other HTTP errors are not suppressed.

        Raises:
            InvalidModelError: If the model ID returns a 404 from HuggingFace Hub.
        """
        try:
            HfApi().model_info(self._model_name)
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise InvalidModelError(f"{self._model_name} does not exist on HuggingFace Hub") from e

    def embeddings_generate(self, path: str | PathLike, page_chunks: int = 50) -> tuple[list[str], SentenceTransformer]:
        """Extract text and table content from a PDF and return it alongside the embedding model.

        The caller is responsible for encoding the returned text list into vectors using the
        returned model (e.g. ``model.encode(content_list)``).

        Args:
            path (str | PathLike): Path to the source PDF file.
            page_chunks (int): Number of pages to convert per batch. Defaults to 50.

        Returns:
            tuple[list[str], SentenceTransformer]:
                - Deduplicated list of enriched text strings ready for encoding.
                - The SentenceTransformer model instance used by this processor.

        Raises:
            FileNotFoundError: If ``path`` does not point to an existing file.
            ValueError: If the file is not a valid PDF or cannot be opened.
        """
        logger.info(f"{path=} : {page_chunks=}")
        return self._extract_chunk_data(path=path, page_chunks=page_chunks), self._model

    def _convert_document_gen(self,
                              file: str | PathLike,
                              total_pages: int,
                              page_chunks: int) -> Iterable[DoclingDocument]:
        """Yield Docling document objects by converting the PDF in page-range batches.

        Streams conversion results to keep memory usage bounded for large PDFs.

        Args:
            file (str | PathLike): Path to the source PDF file.
            total_pages (int): Total page count + 1 (exclusive upper bound for range iteration).
            page_chunks (int): Number of pages per conversion batch.

        Yields:
            DoclingDocument: Parsed document object for each page batch.
        """
        for start in range(1, total_pages, page_chunks):
            end = min(start + page_chunks, total_pages)
            logger.info(f"Converting pages: {start}:{end}")
            result = self._converter.convert(source=file, page_range=(start, end))
            yield result.document

    def _extract_chunk_data(self, path: str | PathLike, page_chunks: int = 50) -> list[str]:
        """Parse a PDF and extract deduplicated enriched-text strings from text chunks and tables.

        Orchestrates page-batched conversion via ``_convert_document_gen``, then pipes each
        ``DoclingDocument`` through ``extract_text_gen`` and ``extract_table_gen``.
        Deduplication is scoped to this call via a local content set.

        Args:
            path (str | PathLike): Path to the PDF file.
            page_chunks (int): Pages per conversion batch. Defaults to 50.

        Returns:
            list[str]: Ordered, deduplicated list of enriched text strings (text chunks + table markdown).

        Raises:
            FileNotFoundError: If ``path`` does not point to an existing file.
            ValueError: If the file is not a valid PDF or page count cannot be determined.
        """
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"Invalid file: {file}")

        num_pages = self.get_total_pages(srcFile=file)
        logger.info(f"{num_pages=}")

        if num_pages == 0:
            raise ValueError(f"Could not determine page count for '{file}'. Ensure it is a valid PDF.")

        # Reset deduplication state per call to avoid cross-call contamination
        content_set: set[str] = set()
        embed_text_list: list[str] = []

        for docobj in self._convert_document_gen(file, total_pages=num_pages + 1, page_chunks=page_chunks):
            embed_text_list.extend(chain(
                self.extract_text_gen(docobj=docobj, content_set=content_set),
                self.extract_table_gen(docobj=docobj, content_set=content_set),
            ))

        return embed_text_list

    def get_total_pages(self, srcFile: str | PathLike) -> int:
        """Return the page count of a PDF file using PyMuPDF.

        Args:
            srcFile (str | PathLike): Path to the PDF file.

        Returns:
            int: Number of pages, or 0 if the file is not a PDF or cannot be opened.
        """
        if not Path(srcFile).suffix.lower() == '.pdf':
            return 0

        try:
            with closing(pymupdf.open(srcFile)) as doc:
                return len(doc)
        except Exception as e:
            logger.exception(f"Failed to read '{srcFile}': {e}")
            return 0

    def extract_text_gen(self, docobj: DoclingDocument, content_set: set[str]) -> Iterable[str]:
        """Yield context-enriched text chunks from a DoclingDocument.

        Uses ``HybridChunker`` to split the document into semantic chunks, then calls
        ``contextualize`` to prepend heading context. Skips empty strings and deduplicates
        against ``content_set``.

        Args:
            docobj (DoclingDocument): Parsed Docling document for a page batch.
            content_set (set[str]): Shared deduplication set for this extraction call.

        Yields:
            str: Non-empty, deduplicated enriched text strings.
        """
        for chunk in self._chunker.chunk(dl_doc=docobj):
            enriched_text = self._chunker.contextualize(chunk=chunk)
            text = (enriched_text or "").strip()
            if text and text not in content_set:
                content_set.add(text)
                yield text

    def extract_table_gen(self, docobj: DoclingDocument, content_set: set[str]) -> Iterable[str]:
        """Yield Markdown-formatted tables extracted from a DoclingDocument.

        Iterates all document items, filters for ``TableItem`` instances, exports each as
        Markdown, and deduplicates against ``content_set``.

        Args:
            docobj (DoclingDocument): Parsed Docling document for a page batch.
            content_set (set[str]): Shared deduplication set for this extraction call.

        Yields:
            str: Non-empty Markdown table strings not already present in ``content_set``.
        """
        for item, _ in docobj.iterate_items():
            if isinstance(item, TableItem):
                markdown = (item.export_to_markdown(doc=docobj) or "").strip()
                if markdown and markdown not in content_set:
                    content_set.add(markdown)
                    yield markdown
