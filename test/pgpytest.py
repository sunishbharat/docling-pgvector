# pgvector_test.py
import pytest
from test.document_processor_test import document_proc_test
from test.docling_test import docling_test

# ── define the file path once here ──────────────────────────
@pytest.fixture
def test_file():
    return "./data/test.pdf"

# ── pytest calls main() automatically ───────────────────────
def test_pgvector_processing(test_file):
    records = process_document(test_file)

    assert records is not None
    assert len(records)>0
    assert isinstance(records[0][1],str)
    assert records[0][1] != ""

def test_docling(test_file):
    records = process_document2(test_file)

    assert records is not None
    assert len(records)>0
    assert isinstance(records[0][1],str)
    assert records[0][1] != ""

def process_document(file_path: str) -> list:
    print(f"Processing: {file_path}")
    records = document_proc_test(file_path)
    return records

def process_document2(file_path: str) -> list:
    print(f"Processing: {file_path}")
    records = docling_test(file_path)
    return records