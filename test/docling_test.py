
# docling_test 
#
# To run in local vscode environment.
# > uv run python -m test.docling_test  
#
#
import os
import sys
import logging
import torch
import pymupdf
import urllib.request
from os import PathLike
from pathlib import Path
from docling.exceptions import ConversionError
from docling_core.types.doc import DoclingDocument
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from pgvector_client import PGVectorClient
from collections.abc import Iterable
from docling_core.types.doc import (
    DoclingDocument,
    TextItem,
    TableItem,
    PictureItem,
    SectionHeaderItem,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_ID = "BAAI/bge-base-en-v1.5"

srcFile = r"./data/test.pdf"
url_paper=r"https://arxiv.org/pdf/1706.03762"
page_chunks = 50



######################################
# get_total_pages
#
######################################
def get_total_pages(srcFile:str) -> int:
    """ 
    Count total number of pages in a pdf document 
    
    Args:
        srcFile: document file name with path (str)
        
    Return:
        int: Returns page count. Returns 0 if file not present or invalid type
        
    """
    
    if not Path(srcFile).suffix.lower() == '.pdf':
        return 0

    try:
        doc = pymupdf.open(srcFile)
        doclen = len(doc)
        doc.close()
        return doclen
    except Exception as e:
        logging.exception(f"Exception: {e}")
        return 0


######################################
# _convert_document_gen
#
# Using iterables - generator , memory efficient
######################################
def _convert_document_gen(converter:DocumentConverter, 
                          file:str|PathLike,
                          total_pages:int,
                          page_chunks:int) -> Iterable[DoclingDocument]:
    """ Convert document stream by page """
    
    file = Path(file)
    if not file.is_file():
        raise FileNotFoundError(f"Invalid file {file}") 
    
    for start in range(1,total_pages,page_chunks):
        end = min(start + page_chunks, total_pages)

        result = converter.convert(source=file,page_range=(start,end))
        yield result.document
        logging.info(f"Pages : {start}:{end}")
            
        


######################################
# model_init
#
######################################
def model_init(model_name:str) -> tuple[SentenceTransformer, HybridChunker]:

    model = SentenceTransformer(EMBED_MODEL_ID)
    tokenizer: BaseTokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    )
        
    chunker = HybridChunker(tokenizer=tokenizer)
    logging.info(f"{tokenizer.get_max_tokens()=}")
    
    return model, chunker
    
    
    

######################################
# generate_embeddings
#
######################################
def generate_embeddings(model:SentenceTransformer, 
                        chunker:HybridChunker,
                        doc_obj_list:list[DoclingDocument]) : 

    content_set=set()
    content_extract_list = []
    for docobj in doc_obj_list:
        """ Extract text """ 
        for chunk in chunker.chunk(dl_doc=docobj):
            enriched_text = chunker.contextualize(chunk=chunk)
            text = (enriched_text or "").strip()
            if(text):
                content_set.add(text)
                content_extract_list.append(text)

        """ Extract tables""" 
        for item, level in docobj.iterate_items():
            if isinstance(item, TableItem):
                markdown = (item.export_to_markdown(doc=docobj) or "").strip()
                if markdown not in content_set:
                    content_set.add(markdown)
                    content_extract_list.append(markdown)
                
    # Encode into vector embeddings.
    embeddings_list2 = model.encode(content_extract_list, batch_size=32)
    return embeddings_list2, content_extract_list




######################################
# pgVector_db_update
#
######################################
def pgVector_db_update(model:SentenceTransformer ,content_extract_list:list, embeddings_list2):

    embd_dim = model.get_sentence_embedding_dimension()
    with PGVectorClient() as pgclient:
        with pgclient.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS ITEMS")
            cur.execute(f"""CREATE TABLE IF NOT EXISTS items (id bigserial PRIMARY KEY, text TEXT, embedding vector({embd_dim}));""")

    with PGVectorClient() as pgclient:
        for chunk, embed in zip(content_extract_list, embeddings_list2):
            with pgclient.cursor() as cur:
                cur.execute("INSERT INTO items (text, embedding) VALUES (%s, %s);", (chunk, embed))

    logging.info("Embeddings Updated ...")

######################################
# start_main 
#
######################################
def start_main(path:str|PathLike)->SentenceTransformer:
    
    # Auto-detect GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using {device} accelerator")

    # Configure pipeline  
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    total_pages= get_total_pages(srcFile=path)+1
    #total_pages=20
    logging.info(f"{total_pages=}")
    
    doc_obj_list = []
    for docobj in _convert_document_gen(converter, file=path,total_pages=total_pages,page_chunks=page_chunks):
        doc_obj_list.append(docobj)
    
    logging.info(f"Total chunks = {len(doc_obj_list)}")
    
    model, chunker = model_init(EMBED_MODEL_ID)
    
    embeddings, content_extract_list = generate_embeddings(model, chunker, doc_obj_list)
    pgVector_db_update(model, content_extract_list=content_extract_list, embeddings_list2=embeddings)
    
    return model
    


##########################################
# Test the embeddings with a user query
##########################################
def test_embeddings(query:str, model:SentenceTransformer) -> list[tuple]:
    query_emb = model.encode(query, normalize_embeddings=True)
    
    sql = """
        SELECT id, text, embedding <-> %s AS distance
        FROM items
        ORDER BY embedding <-> %s
        LIMIT 5;
        """

    logging.info(f"SQL Query = {sql} ")

    with PGVectorClient() as pgclient:
        with pgclient.cursor() as cur:
            cur.execute(sql, (query_emb, query_emb))
            rows = cur.fetchall()

    for id_, text, dist in rows:
        logging.info("*"*40)
        logging.info(f"{id_=}, {dist=}, ->\n {text}")
        
    return rows

def docling_test(path:str|PathLike) -> tuple[str]:
    model = start_main(path=path)
    query = "The Transformer achieves better BLEU scores than previous state-of-the-art models"
    records = test_embeddings(query, model)
    return records

def load_file():
    path=srcFile
    if not os.path.exists(path):
        urllib.request.urlretrieve(url_paper, path)
        logging.info("Loading ..{path=}")
    return path
        
if __name__=="__main__":
    model = start_main(path=load_file())
    query = "The Transformer achieves better BLEU scores than previous state-of-the-art models"
    records = test_embeddings(query, model)
    assert len(records)>0

    assert isinstance(records[0][1],str)
    assert records[0][1] != ""



