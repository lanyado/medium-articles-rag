import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from consts import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLMOD_BASE_URL,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    CHUNK_SIZE,
    OVERLAP_RATIO,
    TOP_K,
    MIN_ARTICLES,
    CHUNKS_PER_ARTICLE,
    LLM_SYSTEM_PROMPT,
)


app = FastAPI()


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

embeddings_model = OpenAIEmbeddings(
    api_key=os.environ["LLMOD_API_KEY"],
    base_url=LLMOD_BASE_URL,
    model=EMBEDDING_MODEL,
)

vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings_model,
    namespace=PINECONE_NAMESPACE,
    text_key="text",
)

llm = ChatOpenAI(
    api_key=os.environ["LLMOD_API_KEY"],
    base_url=LLMOD_BASE_URL,
    model=LLM_MODEL,
)

document_prompt = PromptTemplate.from_template(
    "Title: {title}\n"
    "Authors: {authors}\n"
    "URL: {url}\n"
    "Tags: {tags}\n"
    "Passage: {page_content}"
)


def retrieve_chunks(question, n_articles=MIN_ARTICLES,
                      chunks_per_article=CHUNKS_PER_ARTICLE, top_k=TOP_K):
    """
    The user only cares about ARTICLES, not chunks.
    The user's questions are: find one article, list 3 articles, summarize one article, recommend one article.
    So an article's score = its highest-scored chunk.
    One query usually returns chunks from enough distinct articles. 
    If not, re-query excluding articles we already got, up to MIN_ARTICLES times.
    Because, in each query, we only get chunks from at least one article.
    """
    query_vector = embeddings_model.embed_query(question)
    recieved_article_ids, selected_chunks = [], []
    for _ in range(n_articles):
        articles_filter = (
            {"article_id": {"$nin": recieved_article_ids}} if recieved_article_ids else None
        )
        scored_chunks = vectorstore.similarity_search_by_vector_with_score(
            query_vector, k=top_k, filter=articles_filter
        )
        if not scored_chunks:
            break

        chunks_by_article = {}
        for chunk, score in scored_chunks:
            chunks_by_article.setdefault(chunk.metadata["article_id"], []).append((chunk, score))

        for id_, chunks in chunks_by_article.items():
            chunks.sort(key=lambda chunk_score: chunk_score[1], reverse=True)
            selected_chunks.extend(chunks[:chunks_per_article])
            recieved_article_ids.append(id_)

        if len(recieved_article_ids) >= n_articles:
            return selected_chunks

    return selected_chunks


def answer(question):
    chunks = retrieve_chunks(question)

    context_str = "\n\n".join(
        document_prompt.format(**{**doc.metadata, "page_content": doc.page_content})
        for doc, _ in chunks
    )
    system_prompt = LLM_SYSTEM_PROMPT + "\n\nContext:\n" + context_str
    response = llm.invoke([("system", system_prompt), ("human", question)])

    return {
        "response": response.content,
        "context": [
            {
                "article_id": doc.metadata["article_id"],
                "title": doc.metadata["title"],
                "chunk": doc.page_content,
                "score": float(score),
            }
            for doc, score in chunks
        ],
        "Augmented_prompt": {"System": system_prompt, "User": question},
    }


class PromptRequest(BaseModel):
    question: str = Field(min_length=1)


@app.post("/api/prompt")
def prompt(req: PromptRequest):
    return answer(req.question)


@app.get("/api/stats")
def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
    }
