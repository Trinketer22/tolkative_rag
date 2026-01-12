"""
Vector store service (FAISS wrapper).
Handles all vector searhc operations
"""

import asyncio
import logging
from typing import List, Optional, Callable
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import settings
from utils.async_helpers import execute_async
from services.embedding import embedding_service

log = logging.getLogger(__name__)


class VectorStoreService:
    """Vector store service."""

    def __init__(
        self,
        storage_path: str = settings.INDEX_PATH,
        headers_path: str = settings.HEADERS_INDEX_PATH,
        search_type: str = settings.SEARCH_TYPE,
    ):
        self.storage: Optional[FAISS] = None
        self.headers_storage: Optional[FAISS] = None
        self.storage_path: str = storage_path
        self.headers_path: str = headers_path
        self.embeder: EmbeddingService = embedding_service.embedder
        self._search_type = search_type
        self._initialized = False

    def initialize(self):
        """Load indexes on startup."""
        if self._initialized:
            return

        log.info("Loading documents index...")
        self.storage = FAISS.load_local(
            self.storage_path,
            embedding_service.embedder,
            allow_dangerous_deserialization=True,
        )

        log.info("Loading headersindex...")
        self.headers_storage = FAISS.load_local(
            self.headers_path,
            embedding_service.embedder,
            allow_dangerous_deserialization=True,
        )

        self._initialized = True

    async def search(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 10,
        from_headers: bool = False,
        with_score: bool = False,
        filter_fn: Optional[Callable] = None,
    ) -> List[Document]:
        if not self._initialized:
            raise RuntimeError("VectorStore not initialized")

        search_fn: Callable
        search_storage = self.headers_storage if from_headers else self.storage
        if query_vector is None:
            query_vector = await embedding_service.embed_query(query)

        if with_score:
            if self._search_type == "MMR":
                search_fn = (
                    lambda: search_storage.max_marginal_relevance_search_with_score_by_vector(
                        query_vector, k=top_k
                    )
                )
            else:
                search_fn = (
                    lambda: search_storage.similarity_search_with_score_by_vector(
                        query_vector, k=top_k
                    )
                )
        else:
            if self._search_type == "MMR":
                search_fn = (
                    lambda: search_storage.max_marginal_relevance_search_by_vector(
                        query_vector, k=top_k
                    )
                )
            else:
                search_fn = lambda: search_storage.similarity_search_by_vector(
                    query_vector, k=top_k
                )
        docs = await execute_async(search_fn)

        # Apply filter
        if filter_fn:
            try:
                filtered_docs = list(filter(filter_fn, docs))
                # Don't update if something has failed
                docs = filtered_docs
            except Exception as e:
                log.error(f"Filter failed for docs {docs}")

        return docs

    # Wraps FAISS get_by_ids
    def get_by_ids(
        self,
        ids: List[str],
    ) -> List[Document]:
        if not self._initialized:
            raise RuntimeError("VectorStore not initialized")

        return self.storage.get_by_ids(ids)

    async def cleanup(self):
        """Cleanup resources."""
        pass  # FAISS doesn't need cleanup


# Global singleton
vector_store = VectorStoreService()
