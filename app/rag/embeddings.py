"""Text embeddings, computed locally with fastembed (ONNX, no API key needed)."""

from functools import lru_cache

from fastembed import TextEmbedding

from app.config import settings

# Output size of BAAI/bge-small-en-v1.5. If you change EMBEDDING_MODEL, update
# this and re-create the faq_entries table, because the vector column has a fixed size.
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    # The first call downloads the model (~70 MB); later calls load it from the local cache.
    return TextEmbedding(settings.embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    # query_embed adds the query prefix that bge models expect for retrieval.
    return next(iter(_model().query_embed(text))).tolist()
