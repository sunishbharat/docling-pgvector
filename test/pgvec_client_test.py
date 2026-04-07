from pgvector_client import PGVectorClient
from pgvector_client import PGVectorConfig
from tenacity import RetryError
import logging
import sys
import os
from urllib.parse import urlparse

url = urlparse(os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/vectordb"
))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pgConfig = PGVectorConfig(
    database=url.path.lstrip("/"),   # vectordb
    user=url.username,               # postgres
    password=url.password,           # postgres
    host=url.hostname,               # pgvector
    port=url.port,                   # 5432
)

try:
    with PGVectorClient(pgConfig) as client:
        with client.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("INSERT INTO items (embedding) VALUES ('[10,22,33]'), ('[44,55,66]');")
            cur.execute("SELECT * from items")
            
            for c in cur:
                print(c)
        print("Done")
        
except RetryError as e:
    logger.error("All retries exhausted. Root cause: %s", e.last_attempt.exception())
    sys.exit(1)
        
except Exception as e:
    logger.exception("%s",e)
    sys.exit(1)
    