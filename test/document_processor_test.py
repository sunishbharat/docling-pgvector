import logging
from document_processor import DocumentProcessor
from dconfig import EmbeddingsConfig
from pgvector_client import PGVectorClient
from sentence_transformers import SentenceTransformer
from test.docling_test import test_embeddings
from os import PathLike

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_1 = "BAAI/bge-small-en-v1.5"
model_2= "sentence-transformers/all-MiniLM-L6-v2"
model_3= "llama-3.2b"

#srcFile = "./data/VL_JEPA.pdf"
#srcFile = "./data/Math1.pdf"
srcFile = "./data/python_compr.pdf"
srcFile = r"./data/3a.pdf"
page_chunks = 50

######################################
# pgVector_db_update
#
######################################
def pgVector_db_update(model:SentenceTransformer ,content_extract_list:list, embeddings_list):

    embd_dim = model.get_sentence_embedding_dimension()
    with PGVectorClient() as pgclient:
        with pgclient.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS ITEMS")
            cur.execute(f"""CREATE TABLE IF NOT EXISTS items (id bigserial PRIMARY KEY, text TEXT, embedding vector({embd_dim}));""")

    with PGVectorClient() as pgclient:
        for chunk, embed in zip(content_extract_list, embeddings_list):
            with pgclient.cursor() as cur:
                cur.execute("INSERT INTO items (text, embedding) VALUES (%s, %s);", (chunk, embed))

    logging.info("Embeddings Updated ...")
    
    
    
def document_proc_test(path:str|PathLike) -> tuple[str]:

    logger.info(f"{path=} : {page_chunks=}")
    embedconfig:EmbeddingsConfig = EmbeddingsConfig( model_name= model_1)

    doc = DocumentProcessor(embedconfig=embedconfig)
    content_list, model = doc.embeddings_generate(path=path, page_chunks=50)

    # Encode text documents into fixed-size vector embeddings using SentenceTransformer.
    embed_list = model.encode(content_list)

    # Commit it into postgres vector db
    pgVector_db_update(model=model, content_extract_list=content_list, embeddings_list=embed_list)


    # Test vector embeddings inference for similarity search.
    query = "Overview of Supervised Learning"
    records = test_embeddings(query=query, model=model)
    return records

if __name__=="__main__":
    document_proc_test(path=srcFile)
    logging.info("document_proc_test completed")