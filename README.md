<p align="center">
  <img src="https://cdn.prod.website-files.com/68e09cef90d613c94c3671c0/697e805a9246c7e090054706_logo_horizontal_grey.png" alt="Yeti" width="200" />
</p>

---

# Yeti Vectors

[![Yeti](https://img.shields.io/badge/Yeti-Extension-blue)](https://yetirocks.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![fastembed](https://img.shields.io/badge/fastembed-ONNX-orange)](https://github.com/Anush008/fastembed-rs)

Automatic text and image embedding extension for Yeti. Uses local ONNX-based inference via `fastembed` to generate vector embeddings on writes and convert text queries to vectors on reads. Includes a persistent embedding cache shared across all consumer applications.

## Features

- **Automatic Embedding** - Generates vector embeddings on insert/update with zero client-side code
- **Text Search** - Convert natural language queries to vectors server-side
- **Image Embedding** - CLIP model support for base64-encoded image fields
- **Embedding Cache** - Persistent cache eliminates redundant embedding computations (~50-200ms savings per cache hit)
- **Cross-App Sharing** - Cache is shared across all apps using the same model
- **Backfill** - Automatically embeds existing records when added to an app
- **Multiple Models** - Support for 5 embedding models (text and image)
- **Per-App Config** - Each consumer app specifies its own field mappings and model choices

## Installation

```bash
# Clone into your Yeti applications folder
cd ~/yeti/applications
git clone https://github.com/yetirocks/yeti-vectors.git

# Restart Yeti to load the extension
# Models are downloaded automatically on first use (~80MB-1.3GB depending on model)
```

## Consumer App Setup

Add yeti-vectors to any application's `config.yaml`:

```yaml
extensions:
  - yeti-vectors:
      fields:
        - source: content        # Source text field
          target: embedding       # Target vector field (must have HNSW index)
          model: "BAAI/bge-small-en-v1.5"
          field_type: text
```

Your schema needs an HNSW-indexed vector field:

```graphql
type Document @table @export {
    id: ID! @primaryKey
    content: String!
    embedding: [Float!]! @indexed(type: "HNSW")
}
```

## How It Works

### On Write (Insert/Update)

1. Client sends a record with the `content` field populated
2. yeti-vectors intercepts the write, runs the text through the embedding model
3. The generated vector is stored in the `embedding` field
4. The record is saved with both fields

```bash
# Insert a document (embedding auto-generated from content)
curl -sk -X POST https://localhost:9996/my-app/Document \
  -H "Content-Type: application/json" \
  -d '{"id": "doc-1", "content": "Machine learning is a subset of AI..."}'

# The stored record includes the embedding:
curl -sk https://localhost:9996/my-app/Document/doc-1
# {"id":"doc-1","content":"Machine learning...","embedding":[0.12,-0.45,...]}
```

### On Read (Text Search)

1. Client sends a text query via `vector_text` parameter
2. Server checks the embedding cache for a matching (text, model) pair
3. On cache miss, the text is vectorized and the result cached
4. HNSW nearest-neighbor search runs against the index

```bash
# Search with natural language
curl -sk "https://localhost:9996/my-app/Document?\
vector_attr=embedding&\
vector_text=how+does+deep+learning+work&\
vector_model=BAAI/bge-small-en-v1.5&\
limit=5"
```

### Text Search Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `vector_text` | Yes | Natural language query text |
| `vector_model` | Yes | Embedding model name (must match field mapping) |
| `vector_attr` | No | Target attribute (default: `embedding`) |
| `limit` | No | Max results (default: 10) |
| `max_distance` | No | Maximum cosine distance threshold |

## Embedding Cache

The cache is stored in an `EmbeddingCache` table in the yeti-vectors RocksDB database. Cache keys are SHA-256 hashes of `model + "\0" + text`, so entries are deterministic and shared across all apps using the same model.

### Cache Management

```bash
# List cached embeddings
curl -sk https://localhost:9996/yeti-vectors/EmbeddingCache

# View a specific cache entry
curl -sk https://localhost:9996/yeti-vectors/EmbeddingCache/{id}

# Delete a cache entry
curl -sk -X DELETE https://localhost:9996/yeti-vectors/EmbeddingCache/{id}
```

### Disable Cache Per-App

```yaml
extensions:
  - yeti-vectors:
      cache: false
      fields:
        - source: content
          target: embedding
          model: "BAAI/bge-small-en-v1.5"
          field_type: text
```

## Image Embedding

Use `field_type: image` with a CLIP model for base64-encoded image fields:

```yaml
extensions:
  - yeti-vectors:
      fields:
        - source: description
          target: textEmbedding
          model: "BAAI/bge-small-en-v1.5"
          field_type: text
        - source: thumbnail
          target: imageEmbedding
          model: "clip-ViT-B-32"
          field_type: image
```

```graphql
type Product @table @export {
    id: ID! @primaryKey
    description: String
    textEmbedding: [Float!]! @indexed(type: "HNSW")
    thumbnail: Bytes
    imageEmbedding: [Float!]! @indexed(type: "HNSW")
}
```

## Backfill

When you add yeti-vectors to an app with existing records, it automatically backfills embeddings on the next restart:

- **Idempotent** - Only records without embeddings are processed
- **Non-blocking** - Server starts immediately; backfill runs concurrently
- **Progress logged** - Watch for `Backfilling` messages in the server log

## Supported Models

| Model | Type | Dimensions | Size |
|-------|------|------------|------|
| `BAAI/bge-small-en-v1.5` | Text | 384 | ~130 MB |
| `BAAI/bge-base-en-v1.5` | Text | 768 | ~440 MB |
| `BAAI/bge-large-en-v1.5` | Text | 1024 | ~1.3 GB |
| `all-MiniLM-L6-v2` | Text | 384 | ~80 MB |
| `clip-ViT-B-32` | Image | 512 | ~300 MB |

Models are downloaded automatically on first use and cached in `~/yeti/cache/models/`.

## Schema

```graphql
type EmbeddingCache @table(database: "yeti-vectors") @export {
  id: String @primaryKey       # SHA-256 hash of model + text
  model: String                # Embedding model name
  embedding: [Float]           # Cached vector
  createdAt: Int               # Unix timestamp
}
```

## Project Structure

```
yeti-vectors/
├── config.yaml          # Extension configuration
├── schema.graphql       # EmbeddingCache table
└── resources/
    └── vectors.rs       # VectorsExtension, FastEmbedVectorHook,
                         # model loading, text/image embedding
```

## Learn More

- [Yeti Documentation](https://yetirocks.com/docs)
- [Vector Search Guide](https://yetirocks.com/docs/guides/vector-search)
- [HNSW Indexing](https://yetirocks.com/docs/guides/vector-search#defining-a-vector-field)
- [fastembed](https://github.com/Anush008/fastembed-rs)

---

Built with [Yeti](https://yetirocks.com) - The fast, declarative database platform.
