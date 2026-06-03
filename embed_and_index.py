import os
import uuid
import ast

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from consts import (
    SUBSET_DATASET_FILENAME,
    EMBEDDING_BASE_MODEL,
    EMBEDDING_MODEL,
    LLMOD_BASE_URL,
    PINECONE_NAMESPACE,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    CHUNK_SIZE,
    OVERLAP_RATIO,
)

UPSERT_BATCH_SIZE = 100
FETCH_BATCH_SIZE = 1000

embeddings_model = OpenAIEmbeddings(
    api_key=os.environ["LLMOD_API_KEY"],
    base_url=LLMOD_BASE_URL,
    model=EMBEDDING_MODEL,
)


def _safe_list(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def _doc_id(doc):
    return f"{doc.metadata['article_id']}:{doc.metadata['chunk_id']}"


def build_documents(df):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=EMBEDDING_BASE_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=int(CHUNK_SIZE * OVERLAP_RATIO),
        separators=["\n\n", "\n", " ", ""],
    )

    documents = []
    for _, row in df.iterrows():
        article_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(row["url"])))
        article_text_chunks = splitter.split_text(str(row["text"]))
        for chunk_id, chunk_text in enumerate(article_text_chunks, start=1):
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "article_id": article_id,
                        "chunk_id": chunk_id,
                        "title": str(row["title"]),
                        "url": str(row["url"]),
                        "authors": _safe_list(row["authors"]),
                        "tags": _safe_list(row["tags"]),
                        "timestamp": str(row["timestamp"]),
                    },
                )
            )
    return documents


def ensure_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=len(embeddings_model.embed_query("check dimensions")),
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(PINECONE_INDEX_NAME)


def existing_ids(pinecone_index, ids):
    found = set()
    for i in range(0, len(ids), FETCH_BATCH_SIZE):
        batch = ids[i:i + FETCH_BATCH_SIZE]
        found.update(pinecone_index.fetch(ids=batch, namespace=PINECONE_NAMESPACE).vectors.keys())
    return found


def upsert(pinecone_index, documents, vectors):
    records = [
        {
            "id": _doc_id(doc),
            "values": vector,
            "metadata": {**doc.metadata, "text": doc.page_content},
        }
        for doc, vector in zip(documents, vectors)
    ]
    for i in range(0, len(records), UPSERT_BATCH_SIZE):
        pinecone_index.upsert(
            vectors=records[i:i + UPSERT_BATCH_SIZE],
            namespace=PINECONE_NAMESPACE,
        )


def main():
    print(f"Loading dataset: {SUBSET_DATASET_FILENAME}")
    df = pd.read_csv(SUBSET_DATASET_FILENAME)

    print("Splitting articles intelligently via structural boundaries...")
    documents = build_documents(df)
    print(f"Created {len(documents)} chunks")

    pinecone_index = ensure_index()
    upserted_doc_ids = existing_ids(pinecone_index, [_doc_id(d) for d in documents])
    docs_to_embed = [doc for doc in documents if _doc_id(doc) not in upserted_doc_ids]
    print(f"Skipping {len(upserted_doc_ids)} existing chunks; embedding {len(docs_to_embed)} new chunks")

    if docs_to_embed:
        vectors = embeddings_model.embed_documents([doc.page_content for doc in docs_to_embed])
        upsert(pinecone_index, docs_to_embed, vectors)

    print(pinecone_index.describe_index_stats())


if __name__ == "__main__":
    main()
