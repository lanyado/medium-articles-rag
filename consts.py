DATASET_FILENAME = "data/medium-english-50mb.csv"
SUBSET_DATASET_FILENAME = "data/medium-english-50mb_subset.csv"

EMBEDDING_BASE_MODEL = "text-embedding-3-small"
LLM_BASE_MODEL = "gpt-5-mini"

EMBEDDING_MODEL = "ZYRANGG-"+EMBEDDING_BASE_MODEL
LLM_MODEL = "ZYRANGG-"+LLM_BASE_MODEL

PINECONE_NAMESPACE = "Medium-Rag"
PINECONE_INDEX_NAME = "medium-rag-1"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
LLMOD_BASE_URL = "https://api.llmod.ai/v1"

CHUNK_SIZE = 512 # Tokens
OVERLAP_RATIO = 0.1 # 10%

PINECONE_BATCH_SIZE = 100 # For fetching and upserting during indexing

TOP_K = 15 # chunks fetched per query (llm recieves chunks_per_article * articles_in_query, usually 6–15)
MIN_ARTICLES = 3 # distinct articles returned
CHUNKS_PER_ARTICLE = 3 # best chunks kept per article

LLM_SYSTEM_PROMPT = """You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: “I don’t know based on the provided Medium articles data.” Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful."""

USER_QUERIES = [
    "Find an article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable. Provide the title and author.",
    "List exactly 3 articles about education. Return only the titles.",
    "Find an article that argues past pandemics (such as the bubonic plague) can spur innovation and recovery, and summarise its central argument.",
    "I want practical, beginner-friendly advice on building habits that actually stick. Which article would you recommend, and why?"
]