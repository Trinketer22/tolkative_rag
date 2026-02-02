"""
Vector store service (FAISS wrapper).
Handles all vector searhc operations
"""

import logging
from typing import List, Optional, Callable
from aiorwlock import RWLock
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
        storage: Optional[FAISS] = None,
        header_storage: Optional[FAISS] = None,
    ):
        self.storage: Optional[FAISS] = storage
        self.headers_storage: Optional[FAISS] = header_storage
        self.storage_path: str = storage_path
        self.headers_path: str = headers_path
        self._lock: RWLock = RWLock()
        self._search_type = search_type
        self._initialized = False
        self._storage_updated = False

    def initialize(self):
        """Load indexes on startup."""
        if self._initialized:
            return

        if embedding_service.embedder is None:
            raise RuntimeError("Embedder service is not initialized!")
        if self.storage is None:
            log.info("Loading documents index...")
            self.storage = FAISS.load_local(
                self.storage_path,
                embedding_service.embedder,
                allow_dangerous_deserialization=True,
            )

        if self.headers_storage is None:
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
        if search_storage is None:
            raise RuntimeError("Storage is not initialized!")
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

        async with self._lock.reader_lock:
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

    async def add_documents(self, documents: List[Document]) -> List[str]:
        if (
            not self._initialized
            or self.storage is None
            or self.headers_storage is None
        ):
            raise RuntimeError("VectorStore not initialized")
        added_ids: List[str] = []

        if len(documents) == 0:
            return added_ids

        doc_crumbs = [
            Document(page_content=doc.metadata["crumbs"], id=doc.id)
            for doc in documents
            if "crumbs" in doc.metadata
        ]

        storage = self.storage
        existing_docs = await self.get_by_ids([doc.id for doc in documents if doc.id])
        if len(existing_docs) > 0:
            existing_ids = [doc.id for doc in existing_docs if doc.id]
            raise RuntimeError(
                f"Documens with ids {','.join(existing_ids)} already exist"
            )
        header_storage = self.headers_storage
        async with self._lock.writer_lock:
            added_ids = await execute_async(lambda: storage.add_documents(documents))
            # Once in a storage lifespan update the flag
            if not self._storage_updated:
                self._storage_updated = len(added_ids) > 0
            if len(doc_crumbs) > 0:
                await execute_async(lambda: header_storage.add_documents(doc_crumbs))

        return added_ids

    # Wraps FAISS get_by_ids
    async def get_by_ids(
        self,
        ids: List[str],
    ) -> List[Document]:
        if not self._initialized or self.storage is None:
            raise RuntimeError("VectorStore not initialized")

        docs: List[Document] = []
        async with self._lock.reader_lock:
            docs = self.storage.get_by_ids(ids)
        return docs

    async def cleanup(self):
        """Cleanup resources."""
        if (
            not self._initialized
            or self.storage is None
            or self.headers_storage is None
        ):
            return
        if self._storage_updated:
            self.storage.save_local(self.storage_path)
            self.headers_storage.save_local(self.headers_path)


# Global singleton
vector_store = VectorStoreService()
