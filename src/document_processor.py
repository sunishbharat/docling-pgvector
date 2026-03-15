
# document_processor test script
#
# To run in local vscode environment.
# > uv run python -m test.document_processor_test
#
#
import sys
import logging
import pymupdf
import numpy as np
from os import PathLike
from pathlib import Path
from itertools import chain
from contextlib import closing
from typing import Iterable, Sequence
from dataclasses import dataclass, field
from dconfig import Chunkerconfig, EmbeddingsConfig
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
from docling_core.types.doc import (
    DoclingDocument,
    TextItem,
    TableItem,
    PictureItem,
    SectionHeaderItem,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ChunkMetaData:
    """ Minimum immutable Metadata required """
    headings: list[str]
    page_no: int
    source_path: str
    chunk_index: int
    
@dataclass(frozen=True)
class ProcessedChunk:
    """ Immutable value object """
    raw_text: str
    enriched_text: str
    metadata: ChunkMetaData
    

class InvalidModelError(ValueError):
    pass

class DocumentProcessor:
    """ Process the documents using docling to get enriched text"""
    def __init__(self, embedconfig:EmbeddingsConfig |None=None):
        
        # Configure pipeline to choose GPU if available  
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        self._embedconfig:EmbeddingsConfig = embedconfig or EmbeddingsConfig()
        self._model = SentenceTransformer(self._embedconfig.model_name)
        self._model_name = self._embedconfig.model_name
        self._content_extract_list = []
        self._content_set = set()
        self.check_model_exists()
        self._initialize()
        

    def _initialize(self) -> None:
        self._tokenizer: BaseTokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self._model_name),
            )
        self._embedconfig.dims = self._model.get_sentence_embedding_dimension() 
        self._chunker = HybridChunker(tokenizer=self._tokenizer)
        
        logger.info(f"model={self._model_name}: "
                    f"max_tokens={self._tokenizer.get_max_tokens()} "
                    f"embed_dims={self._embedconfig.dims} "
                    )
        
    def embeddings_generate(self, path:str|PathLike, page_chunks:int|None=None) -> list[str]:
        logger.info(f"{path=} : {page_chunks=}")
        return self._extract_chunk_data( path=path, page_chunks=page_chunks), self._model
        

    def check_model_exists(self)->Exception:
        """ Sanity check if the model is from Huggingface , if not raise Exception"""
        try:
            HfApi().model_info(self._model_name)
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status == 404:
                raise InvalidModelError(f"{self._model} does not exist in Hugging face ") from e
            
        
    
    def _convert_document_gen(self,
                          file:str|PathLike,
                          total_pages:int,
                          page_chunks:int) -> Iterable[DoclingDocument]:
        """ Convert document stream by page """
        
        for start in range(1,total_pages,page_chunks):
            end = min(start + page_chunks, total_pages)

            result = self._converter.convert(source=file,page_range=(start,end))
            yield result.document
            logger.info(f"Pages : {start}:{end}")
        
    def _extract_chunk_data(self, path:str|PathLike, page_chunks:int | None =50) -> list[str]:
        """
        Extract the document based on item type text, table
        """
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"Invalid file {file}") 

        _numpages = self.get_total_pages(srcFile=file)
        logger.info(f"{_numpages=}")
        
        embed_text_list=[]
        for docobj in self._convert_document_gen(file,total_pages=_numpages+1,page_chunks=page_chunks):
            embed_text_list.extend(chain.from_iterable(
                [self.extract_text_gen(docobj=docobj),
                 self.extract_table_gen(docobj=docobj)]
            ))

    
        return embed_text_list 
                
                
    def _embeddings_vec_generate(self, embed_txt_list:list[str]) -> np.ndarray:
        return self._model.encode(embed_txt_list, batch_size=self._embedconfig.batch_size)
         
    def get_total_pages(self, srcFile:str) -> int:
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
            with closing(pymupdf.open(srcFile)) as doc:
                return len(doc)
        except Exception as e:
            logger.exception(f"Failed to Read:{srcFile} : {e}")
            return 0
        
        
    def extract_text_gen(self, docobj:DoclingDocument) -> Iterable[str]:
        
        """ Extract enriched text from document """
        for chunk in self._chunker.chunk(dl_doc=docobj):
            enriched_text = self._chunker.contextualize(chunk=chunk)
            text = (enriched_text or "").strip()
            if(text):
                self._content_set.add(text)
                yield text


    def extract_table_gen(self, docobj:DoclingDocument) -> Iterable[str]:
        
        """ Extract tables from document """
        for item, _ in docobj.iterate_items():
            if isinstance(item, TableItem):
                markdown = (item.export_to_markdown(doc=docobj) or "").strip()
                if markdown not in self._content_set:
                    self._content_set.add(markdown)
                    yield markdown
            
    