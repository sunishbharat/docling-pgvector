
"""
 PGVectorClient class:
 Prerequisites:
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
import logging
import numpy as np
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from psycopg2 import OperationalError
from psycopg2.extensions import connection, cursor
from dataclasses import dataclass, field
from collections.abc import Generator
from contextlib import contextmanager
from pgvector.psycopg2 import register_vector

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PGVectorConfig:
    database:str= field(default_factory=lambda: os.getenv("PG_DATABASE","vectordb"))
    host:str    = field(default_factory=lambda: os.getenv("PG_HOST","localhost"))
    user:str    = field(default_factory=lambda: os.getenv("PG_USER","postgres"))
    password:str= field(default_factory=lambda: os.getenv("PG_PASSWORD","postgres"))
    port:int    = field(default_factory=lambda: os.getenv("PG_PORT",5432))
    max_retries:int     = 5 
    retry_wait_initial:float =1.0
    retry_wait_max:float  =10.0

@dataclass
class ColumnDefinition:
    """ 
    Table columns with name and PostgreSQL datatype: 'TEXT','INT','vector(4)'
    """
    name:str
    dtype:str
    
    def to_sql(self) -> str:
        return f"{self.name} {self.dtype}"
    


class PGVectorClient:
    """
    pgvector client to interact with PostgreSQL+pgvector database.
    Manages vector embeddings and schemas.

    """
    def __init__(self, config: PGVectorConfig|None=None):
        self.config = config or PGVectorConfig()
        self._conn : connection | None = None
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1.0, max=10.0),
        retry=retry_if_exception_type(OperationalError),
    )
    def connect(self) -> None:
        """ Establish connection with pgvector database with retry mechanism"""
        
        logger.info(f"Connecting to PostgreSQL at \
            {self.config.host}:{self.config.port}:{self.config.database} ")
    
        self._conn = psycopg2.connect(
            database=self.config.database, 
            user    =self.config.user,
            password=self.config.password,
            host    =self.config.host,
            port    =self.config.port
        ) 
        register_vector(self._conn)
        logger.info(f"{self.config.database}:Connection established")
        
    def disconnect(self)->None:
        """
        Gracefully close the database connection
        """
        if self._conn and not self._conn.closed:
            self._conn.close()

        logger.info(f"{self.config.database}:Connection closed")
        
    @contextmanager
    def cursor(self) -> Generator[cursor, None,None]:
        """ 
        Yield a cursor , on success commit and rollback on error
        """
        if not self._conn  or self._conn.closed:
            raise RuntimeError("No database connected, call connect() first")
        cur = self._conn.cursor()

        try:
           yield cur
           self._conn.commit()
           
        except Exception as e:
            self._conn.rollback()
            logger.exception("Failed to commit : %s",e)
        finally:
            # release the cursor
            cur.close()
            
    def __enter__(self) -> "PGVectorClient":
        self.connect()
        return self

        
    def __exit__(self, *_) -> None:
        self.disconnect()
        

