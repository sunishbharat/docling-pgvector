
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError
import pymupdf


srcFile = "./data/Math1.pdf"
srcFile = "./data/VL_JEPA.pdf"
page_chunks = 50



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
        print(f"Exception: {e}")
        return 0



#########################
try:
    converter = DocumentConverter()
    
    total_pages= get_total_pages(srcFile=srcFile)
    total_pages=100
    print(f"{total_pages=}")
    
    doc_obj_list = []
    for start in range(1,total_pages,page_chunks):
        end = min(start + page_chunks, total_pages)
        result = converter.convert(source=srcFile,page_range=(start,end))
        doc_obj_list.append(result.document)
        print(f"Pages : {start}:{end}")
except ConversionError as e:
    print(f"Exception in DocumentionConverter : {e}")
     
print(f"Total chunks = {len(doc_obj_list)}")


####################

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer: BaseTokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
)
chunker = HybridChunker(tokenizer=tokenizer)

print(f"{tokenizer.get_max_tokens()=}")

chunk_list = []
for docobj in doc_obj_list:
    chunk_iter = chunker.chunk(dl_doc=docobj)
    chunk_list.append(list(chunk_iter))


for chunkitem in chunk_list:
    for i, chunk in enumerate(chunkitem):
        print(f"=== {i} ===")
        print(f"chunk.text:\n{f'{chunk.text[:1000]}…'!r}")

        enriched_text = chunker.contextualize(chunk=chunk)
        print(f"\nchunker.contextualize(chunk):\n{f'{enriched_text[:300]}…'!r}")

        print()
    
