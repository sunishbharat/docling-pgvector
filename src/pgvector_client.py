"""PGVectorClient — psycopg2 wrapper for PostgreSQL + pgvector."""

import os
import re
import psycopg2
import logging
from tenacity import Retrying, stop_after_attempt, wait_exponential_jitter, \
    retry_if_exception_type, before_sleep_log
from psycopg2 import OperationalError
from psycopg2.extensions import connection, cursor
from dataclasses import dataclass, field
from collections.abc import Generator
from contextlib import contextmanager
from pgvector.psycopg2 import register_vector

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Allowed pattern for column identifiers — prevents SQL injection in to_sql()
_SAFE_IDENTIFIER = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
# Allowed pattern for dtype: letters, digits, spaces, parentheses, underscores — no quotes or semicolons
_SAFE_DTYPE = re.compile(r'^[a-zA-Z][a-zA-Z0-9_()\s]*$')


@dataclass
class PGVectorConfig:
    database: str  = field(default_factory=lambda: os.getenv("PG_DATABASE", "vectordb"))
    host: str      = field(default_factory=lambda: os.getenv("PG_HOST", "localhost"))
    user: str      = field(default_factory=lambda: os.getenv("PG_USER", "postgres"))
    password: str  = field(default_factory=lambda: os.getenv("PG_PASSWORD", "postgres"))
    port: int      = field(default_factory=lambda: int(os.getenv("PG_PORT", 5432)))
    max_retries: int        = 5
    retry_wait_initial: float = 1.0
    retry_wait_max: float     = 10.0

    def __repr__(self) -> str:
        return (
            f"PGVectorConfig(host={self.host!r}, port={self.port}, "
            f"database={self.database!r}, user={self.user!r}, password='***')"
        )


@dataclass
class ColumnDefinition:
    """Table column with name and PostgreSQL datatype: 'TEXT', 'INT', 'vector(768)'."""
    name: str
    dtype: str

    def __post_init__(self) -> None:
        if not _SAFE_IDENTIFIER.match(self.name):
            raise ValueError(
                f"Invalid column name {self.name!r}. "
                "Must start with a letter or underscore and contain only alphanumerics and underscores."
            )
        if not _SAFE_DTYPE.match(self.dtype):
            raise ValueError(
                f"Invalid dtype {self.dtype!r}. "
                "Must contain only letters, digits, spaces, parentheses, and underscores."
            )

    def to_sql(self) -> str:
        return f"{self.name} {self.dtype}"


class PGVectorClient:
    """psycopg2 client for PostgreSQL + pgvector.

    Usage::

        with PGVectorClient(config) as client:
            with client.cursor() as cur:
                cur.execute("SELECT 1")
    """

    def __init__(self, config: PGVectorConfig | None = None):
        self.config = config or PGVectorConfig()
        self._conn: connection | None = None

    def connect(self) -> None:
        """Establish a connection with exponential-backoff retry on OperationalError.

        Raises:
            tenacity.RetryError: If all retry attempts are exhausted.
        """
        if self._conn and not self._conn.closed:
            logger.debug("connect() called on an already-open connection — skipping.")
            return

        logger.info(
            "Connecting to PostgreSQL at %s:%s/%s",
            self.config.host, self.config.port, self.config.database,
        )

        for attempt in Retrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(
                initial=self.config.retry_wait_initial,
                max=self.config.retry_wait_max,
            ),
            retry=retry_if_exception_type(OperationalError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                self._conn = psycopg2.connect(
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                    host=self.config.host,
                    port=self.config.port,
                )
                register_vector(self._conn)

        logger.info("Connection established to %s/%s", self.config.host, self.config.database)

    def disconnect(self) -> None:
        """Close the database connection if open."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Connection closed (%s/%s)", self.config.host, self.config.database)

    @contextmanager
    def cursor(self) -> Generator[cursor, None, None]:
        """Yield a cursor; commit on success, rollback and re-raise on error.

        Raises:
            RuntimeError: If called before connect().
            Exception: Re-raises any exception that occurs inside the block.
        """
        if not self._conn or self._conn.closed:
            raise RuntimeError("No active connection. Call connect() or use as a context manager.")

        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            logger.exception("Transaction rolled back")
            raise
        finally:
            cur.close()

    def __enter__(self) -> "PGVectorClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
