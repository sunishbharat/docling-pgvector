from document_processor import DocumentProcessor
from dconfig import EmbeddingsConfig

model_1 = "BAAI/bge-small-en-v1.5"
model_2= "sentence-transformers/all-MiniLM-L6-v2"
model_3= "llama-3.2b"

srcFile = "./data/Math1.pdf"
srcFile = "./data/VL_JEPA.pdf"
srcFile = r"./data/3a.pdf"
page_chunks = 50

embedconfig:EmbeddingsConfig = EmbeddingsConfig(
    model_name= model_1
)
doc = DocumentProcessor(embedconfig=embedconfig)

embeddings = doc.embeddings_generate(path=srcFile, page_chunks=50)

for chunk in embeddings:
    for text in chunk:
        print(text)

