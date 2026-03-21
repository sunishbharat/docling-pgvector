# pgvector Pytest
#
# To run locally:
# > uv run python ./test/pgpytest.py -v -s
#
import os
import pytest
import urllib.request
from test.docling_test import docling_test
from test.document_processor_test import document_proc_test

url_paper=r"https://arxiv.org/pdf/1706.03762"

@pytest.fixture
def test_file():
    path=r"./data/test.pdf"
    os.makedirs("./data", exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url_paper, path)
    return path

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
    
def test_document_record_count(test_file):
    records = process_document(test_file)
    assert len(records) == 5

def process_document(file_path: str) -> list:
    print(f"Processing: {file_path}")
    records = document_proc_test(file_path)
    return records

def process_document2(file_path: str) -> list:
    print(f"Processing: {file_path}")
    records = docling_test(file_path)
    return records
