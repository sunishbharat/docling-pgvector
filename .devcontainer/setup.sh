#!/bin/bash
set -e

echo "==> Installing postgresql-client..."
sudo apt-get update && sudo apt-get install -y postgresql-client

echo "==> Waiting for pgvector to be ready..."
until PGPASSWORD=postgres psql -h pgvector -p 5432 -U postgres -c '\q' 2>/dev/null; do
  echo "  pgvector not ready, retrying in 2s..."
  sleep 2
done

echo "==> Creating vectordb and enabling pgvector extension..."
PGPASSWORD=postgres psql -h pgvector -p 5432 -U postgres \
  -c "CREATE DATABASE vectordb;" || echo "DB already exists, continuing..."

PGPASSWORD=postgres psql -h pgvector -p 5432 -U postgres -d vectordb \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "==> Installing Python dependencies..."
uv sync
uv pip install -e .

echo "==> Downloading test PDF..."
mkdir -p ./data
curl -L https://arxiv.org/pdf/1706.03762 -o ./data/test.pdf

echo "==> Setup complete!"