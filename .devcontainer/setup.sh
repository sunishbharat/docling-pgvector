#!/bin/bash
set -e

echo "==> Installing local project..."
pip install -e .

echo "==> Creating vectordb and enabling pgvector extension..."
PGPASSWORD=postgres psql -h pgvector -p 5432 -U postgres \
  -c "CREATE DATABASE vectordb;" || echo "DB already exists, continuing..."

PGPASSWORD=postgres psql -h pgvector -p 5432 -U postgres -d vectordb \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "==> Installing Python dependencies..."
echo "==> Python dependencies Done!"

echo "==> Downloading test PDF..."
mkdir -p ./data
curl -L https://arxiv.org/pdf/1706.03762 -o ./data/test.pdf

echo "==> Setup complete!"
