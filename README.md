# docling-pgvector

[![CI](https://github.com/sunishbharat/docling-pgvector/actions/workflows/python-app.yml/badge.svg)](https://github.com/sunishbharat/docling-pgvector/actions/workflows/python-app.yml)
[![Docker Image Test](https://github.com/sunishbharat/docling-pgvector/actions/workflows/docker-image-test.yml/badge.svg)](https://github.com/sunishbharat/docling-pgvector/actions/workflows/docker-image-test.yml)
[![License MIT](https://img.shields.io/github/license/sunishbharat/docling-pgvector)](https://github.com/sunishbharat/docling-pgvector/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue?logo=docker)](https://github.com/sunishbharat/docling-pgvector/pkgs/container/docling-pgvector)
[![pgvector](https://img.shields.io/badge/pgvector-pg17-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Docling](https://img.shields.io/badge/powered%20by-Docling-blue)](https://github.com/docling-project/docling)

Building a RAG pipeline requires wiring together a document parser, an embedding model, and a vector database yourself, investing time learning each tool along the way. **docling-pgvector** frees you from that overhead and gives you a simple interface to focus on what matters:

- Provide a PDF as input
- Run a query
- Get similarity-ranked text chunks back, ready to pass to any LLM of your choice

Support for additional input formats including CSV, web pages, and other document types is currently under development.

The default embedding model is `BAAI/bge-base-en-v1.5`, but any [HuggingFace SentenceTransformer model](https://huggingface.co/models?library=sentence-transformers) can be used by passing a different `model_name` to `EmbeddingsConfig`.

<br>

## How It Works

```
PDF File
   │
   ▼
Docling (PDF parser)          ← GPU auto-detected
   │  page-batched conversion
   ▼
HybridChunker + TableItem     ← semantic text chunks + Markdown tables
   │  unique content
   ▼
SentenceTransformer            ← BAAI/bge-base-en-v1.5 (768-dim)
   │  vector embeddings
   ▼
PostgreSQL + pgvector          ← similarity search (L2 distance)
```

<br>

## Requirements

> **Before you begin:** Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or an equivalent Docker runtime) is installed and **running** on your machine. All setup options below rely on Docker.

| Option | Tools needed |
|--------|-------------|
| **A — Pre-built Docker Image** (recommended) | Docker Desktop |
| **B — Dev Container** | VS Code + [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) |
| **C — Local** | Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/) |

<br>

## Option A — Pre-built Docker Image (Recommended)

> The fastest way to get started. Pull the pre-built image and run — no Python, no dependency installs, no setup scripts required.

> **Terminal:** Use **Git Bash** or **WSL** on Windows. If you prefer PowerShell, replace `$(pwd)` with `${PWD}` in step 5. Command Prompt is not recommended as some commands will not work correctly.

**1. Pull the image**
```bash
docker pull ghcr.io/sunishbharat/docling-pgvector:cpu-dev
```

**2. Clone the repository**
```bash
git clone https://github.com/sunishbharat/docling-pgvector.git
cd docling-pgvector
```

**3. Start PostgreSQL + pgvector**

Create a shared network so both containers can talk to each other, then start the database.
```bash
docker network create devnet || true

docker rm -f pgvector-container 2>/dev/null || true

docker run --name pgvector-container \
  --network devnet \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  -d pgvector/pgvector:pg17
```

**4. Create the database extension**

Wait until the database is ready, then enable the pgvector extension.
```bash
docker exec pgvector-container bash -c "until pg_isready -U postgres; do sleep 1; done"

docker exec pgvector-container psql -U postgres -d vectordb \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**5. Run the container**

> On Windows, replace `$(pwd)` with `${PWD}` in PowerShell or `%cd%` in Command Prompt.

```bash
docker run --rm -it \
  --network devnet \
  -e DATABASE_URL=postgresql://postgres:postgres@pgvector-container:5432/vectordb \
  -v $(pwd):/workspace/docling-pgvector \
  -w /workspace/docling-pgvector \
  ghcr.io/sunishbharat/docling-pgvector:cpu-dev \
  bash
```

**6. Inside the container — install the project and run tests**
```bash
/opt/venv/bin/pip install -e .
pytest test/pgpytest.py -v -s
python -m test.docling_test
python -m test.document_processor_test
```

<br>

## Option B — Dev Container

> Everything is pre-configured. All dependencies, the app container, and PostgreSQL+pgvector are set up automatically — no manual configuration needed.

**1. Clone the repository**
```bash
git clone https://github.com/sunishbharat/docling-pgvector.git
cd docling-pgvector
```

**2. Open in VS Code**
```bash
code .
```

**3. Reopen in Dev Container**

When VS Code prompts *"Reopen in Container"*, click it. Or open the Command Palette (`Ctrl+Shift+P`) and run:
```
Dev Containers: Reopen in Container
```

**4. Wait for setup to complete**

The first time takes a few minutes. The setup script automatically:
- Installs the project and all Python dependencies
- Creates the `vectordb` database and enables the `vector` extension
- Downloads `./data/test.pdf` (the "Attention Is All You Need" paper for testing)

**5. Run the tests**
```bash
uv run pytest test/pgpytest.py -v -s
uv run python -m test.docling_test
uv run python -m test.document_processor_test
```

<br>

## Option C — Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/sunishbharat/docling-pgvector.git
cd docling-pgvector
```

**2. Start PostgreSQL + pgvector**
```bash
docker run --name pgvector-container \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -d pgvector/pgvector:pg17
```

**3. Create the database and enable the extension**
```bash
PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres \
  -c "CREATE DATABASE vectordb;"

PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d vectordb \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**4. Install Python dependencies**
```bash
uv sync
uv pip install -e .
```

**5. Download the test PDF**
```bash
mkdir -p ./data
curl -L https://arxiv.org/pdf/1706.03762 -o ./data/test.pdf
```

**6. Set the database connection**
```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vectordb
```

> Alternatively, set individual env vars:
> ```bash
> export PG_HOST=localhost
> export PG_PORT=5432
> export PG_DATABASE=vectordb
> export PG_USER=postgres
> export PG_PASSWORD=postgres
> ```

**7. Run the tests**
```bash
uv run pytest test/pgpytest.py -v -s
uv run python -m test.docling_test
uv run python -m test.document_processor_test
```

<br>

## Usage

### 1. Parse PDF and generate embeddings

```python
from document_processor import DocumentProcessor
from dconfig import EmbeddingsConfig

config = EmbeddingsConfig(model_name="BAAI/bge-base-en-v1.5")
processor = DocumentProcessor(embedconfig=config)

content_list, model = processor.embeddings_generate(path="./data/test.pdf")
embeddings = model.encode(content_list)
```

### 2. Store embeddings in PostgreSQL

```python
from pgvector_client import PGVectorClient, PGVectorConfig

pg_config = PGVectorConfig(host="localhost", database="vectordb")
with PGVectorClient(pg_config) as client:
    with client.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"CREATE TABLE IF NOT EXISTS items (id bigserial PRIMARY KEY, text TEXT, embedding vector({model.get_sentence_embedding_dimension()}))")

    for chunk, embed in zip(content_list, embeddings):
        with client.cursor() as cur:
            cur.execute("INSERT INTO items (text, embedding) VALUES (%s, %s)", (chunk, embed))
```

### 3. Similarity search

```python
query_vec = model.encode("your search query", normalize_embeddings=True)
with PGVectorClient(pg_config) as client:
    with client.cursor() as cur:
        cur.execute("""
            SELECT id, text, embedding <-> %s AS distance
            FROM items ORDER BY distance LIMIT 2
        """, (query_vec, query_vec))
        results = cur.fetchall()

for id_, text, dist in results:
    print(f"[{dist:.4f}] {text[:100]}")
```

Docling detects tables in the PDF and exports them as Markdown, so they are stored and retrieved as structured text alongside regular chunks.

### Sample Output

Pass your query string directly to `model.encode()` — no other configuration needed:

```python
query = "Maximum path lengths, per-layer complexity and minimum number of sequential operations Table"
query_vec = model.encode(query, normalize_embeddings=True)
```

Results are ranked by distance — lower means more relevant. Real output from running `document_processor_test.py` against the "Attention Is All You Need" paper:

```
INFO:root:id_=43, dist=0.550322916862681, ->
 Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations
 for different layer types. n is the sequence length, d is the representation dimension,
 k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention.

 | Layer Type     | Complexity per Layer | Sequential Ops | Max Path Length |
 |----------------|----------------------|----------------|-----------------|
 | Self-Attention | O(n² · d)            | O(1)           | O(1)            |
 | Recurrent      | O(n · d²)            | O(n)           | O(n)            |
 | Convolutional  | O(k · n · d²)        | O(1)           | O(log_k(n))     |

INFO:root:id_=16, dist=0.672102512607041, ->
 4 Why Self-Attention
 In this section we compare various aspects of self-attention layers to the recurrent and convolutional
 layers commonly used for mapping one variable-length sequence of symbol representations (x1,...,xn)
 to another sequence of equal length (z1,...,zn).
```

The top result (`id=43`) is a table extracted directly from the PDF by Docling and stored as structured Markdown alongside regular text chunks.

<br>

## Configuration

### Embedding Model (`EmbeddingsConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `BAAI/bge-base-en-v1.5` | Any HuggingFace SentenceTransformer model |
| `dims` | `768` | Auto-resolved from the loaded model |
| `batch_size` | `32` | Encoding batch size |

> The model is validated against HuggingFace Hub before downloading. An `InvalidModelError` is raised if the model ID does not exist.

### Database (`PGVectorConfig`)

| Parameter | Env Var | Default |
|-----------|---------|---------|
| `host` | `PG_HOST` | `localhost` |
| `port` | `PG_PORT` | `5432` |
| `database` | `PG_DATABASE` | `vectordb` |
| `user` | `PG_USER` | `postgres` |
| `password` | `PG_PASSWORD` | `postgres` |

