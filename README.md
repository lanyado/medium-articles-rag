# Medium Article RAG Assistant

Retrieval-Augmented Generation over a corpus of ~7,600 English Medium articles.

## Stack

- **Embeddings:** `text-embedding-3-small` (1536-dim)
- **LLM:** `gpt-5-mini`
- **Vector store:** Pinecone
- **API:** FastAPI on Vercel

## Setup

```bash
pip install -r requirements.txt
export LLMOD_API_KEY=...
export PINECONE_API_KEY=...
```

## Index the corpus

```bash
python embed_and_index.py
```

The script skips chunks already present in Pinecone (deterministic IDs of the form `article_id:chunk_id`), so reruns cost ~zero. Clear the namespace from the Pinecone UI to force a full re-embed.

## API

### `POST /api/prompt`

`Input format (JSON):`

```json
{
    "question": "Your natural language question here"
}
```

`Output format (JSON):`

```json
{
    "response": "Final natural language answer from the model.",
    "context": [
        {
            "article_id": "1234",
            "title": "Sample article title",
            "chunk": "article chunk retrieved",
            "score": 0.1234
        }
    ],
    "Augmented_prompt": {
        "System": "the system prompt used to query the chat model",
        "User": "the user prompt used to query the chat model"
    }
}
```

### `GET /api/stats`

```json
{ "chunk_size": 512, "overlap_ratio": 0.1, "top_k": 6 }
```

## Local dev

```bash
uvicorn api.index:app --reload
```
