
"""
 File to test pgvector with postgres database.
 Simple setup to store vector embedding in postgres database
 
 Steps to setup postgres and pgvector using docker.
 -------------------------------------------------
 ## Download image
 - docker pull pgvector/pgvector:pg17
 - docker images
 
 ## Create Volume
 - docker volume ls
 - docker volume create pgvector-data

 ## Create container
 - docker run --name pgvector-container -e POSTGRES_PASSWORD=postgres -p 5432:5432 -v pgvector-data:/var/lib/postgresql/data -d pgvector/pgvector:pg17
 - docker ps
 
 ## pgadmin4
 - docker pull dpage/pgadmin4
 - docker run --name pgadmin-container -p 5050:80 -e PGADMIN_DEFAULT_EMAIL=user@domain.com -e PGADMIN_DEFAULT_PASSWORD=password -d dpage/pgadmin4
"""

import os
import sys
import psycopg2
import numpy as np
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from psycopg2 import OperationalError

load_dotenv()


@retry(
    stop=stop_after_attempt(1),  # ~5 min max
    wait=wait_exponential_jitter(initial=1, max=2),  # 1-10s with jitter
    retry=retry_if_exception_type(OperationalError),
    before_sleep=print(f"Trying to connect..")
)


def pgvector_connect():
    return psycopg2.connect(
        database="pgvectordb",
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port=5432,
    )

try:
    conn = pgvector_connect()
except Exception as e:
    print(f"Connection Exception : {e}")
    sys.exit(1)

try:
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("DROP TABLE IF EXISTS ITEMS")
    cur.execute("CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));")
    cur.execute("INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');")

    conn.commit()

    cur.execute("SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 2;")
    records = cur.fetchall()
    for r in records:
        print(r)


finally:
    print("Closing db connection")
    cur.close()
