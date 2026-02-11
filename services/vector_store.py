"""
Vector store service (FAISS wrapper).
Handles all vector searhc operations
"""

import logging
from typing import Dict, List, Optional, Callable, Awaitable
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
                log.error(f"Filter failed for docs {docs} {e}")

        return docs

    async def _add_docs_internal(self, documents: List[Document]):
        """
        Internal document addition handler
        Locking should happen before the call
        """
        # Redundant, but for type safety i'll keep it
        if (
            not self._initialized
            or self.storage is None
            or self.headers_storage is None
        ):
            raise RuntimeError("VectorStore not initialized")

        doc_crumbs = [
            Document(page_content=doc.metadata["crumbs"], id=doc.id)
            for doc in documents
            if doc.metadata.get("crumbs")
        ]

        storage = self.storage
        added_ids = await execute_async(lambda: storage.add_documents(documents))
        # Once in a storage lifespan update the flag
        if not self._storage_updated:
            self._storage_updated = len(added_ids) > 0
        if len(doc_crumbs) > 0:
            header_storage = self.headers_storage
            await execute_async(lambda: header_storage.add_documents(doc_crumbs))
        return added_ids

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

        existing_docs = await self.get_by_ids([doc.id for doc in documents if doc.id])
        if len(existing_docs) > 0:
            existing_ids = [doc.id for doc in existing_docs if doc.id]
            raise ValueError(
                f"Documens with ids {','.join(existing_ids)} already exist"
            )
        async with self._lock.writer_lock:
            added_ids = await self._add_docs_internal(documents)

        return added_ids

    # Implementation specifics
    async def _iterate_documents(
        self,
        from_headers: bool = False,
    ):
        if (
            not self._initialized
            or self.storage is None
            or self.headers_storage is None
        ):
            raise RuntimeError("VectorStore not initialized")

        storage = self.headers_storage if from_headers else self.storage
        async with self._lock.reader_lock:
            for doc_id in storage.index_to_docstore_id.values():
                docs = storage.get_by_ids([doc_id])
                if len(docs) > 0:
                    yield docs[0]

    async def _search_documents(self, callback: Callable[[Document], bool]):
        found: List[Document] = []
        async for doc in self._iterate_documents():
            if callback(doc):
                found.append(doc)
        return found

    async def get_document_references(self, ids: List[str]) -> List[Document]:
        references: List[Document] = []
        ids_set = set(ids)
        if len(ids) == 0:
            return references

        def _search_references(doc: Document) -> bool:
            nonlocal ids_set
            found = False
            if not doc.id:
                return found

            doc_refs = doc.metadata.get("children_nodes", []) + doc.metadata.get(
                "references", []
            )
            for ref in doc_refs:
                if ref in ids_set:
                    return True
            return found

        return await self._search_documents(_search_references)

    async def delete_docs(self, ids: List[str], update_refs: bool = True) -> List[str]:
        if (
            not self._initialized
            or self.storage is None
            or self.headers_storage is None
        ):
            raise RuntimeError("VectorStore is not initialized")
        removed_ids = ids
        if len(ids) == 0:
            return removed_ids
        main_storage = self.storage
        headers_storage = self.headers_storage

        await self.get_by_ids(ids, required=True)
        async with self._lock.writer_lock:
            # In theory parallelizeable, but seems overcomplicated
            main_storage.delete(
                ids,
            )
            headers_storage.delete(ids)
            self._storage_updated = True

        if update_refs:
            id_set = set(ids)
            doc_updates: Dict[str, Document] = {}
            filter_ids = lambda doc_id: doc_id not in id_set
            references = await self.get_document_references(ids)
            for ref in references:
                assert ref.id
                children = ref.metadata.get("children_nodes", [])
                links = ref.metadata.get("references", [])
                if children:
                    ref.metadata["children_nodes"] = list(filter(filter_ids, children))
                if links:
                    ref.metadata["references"] = list(filter(filter_ids, links))
                doc_updates[ref.id] = ref

            await self.update_documents(doc_updates)

        return removed_ids

    async def update_documents(
        self,
        docs: Dict[str, Document],
        callback: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        if (
            not self._initialized
            or self.storage is None
            or self.headers_storage is None
        ):
            raise RuntimeError("VectorStore is not initialized")

        new_ids: List[str] = []
        if len(docs) == 0:
            return new_ids
        update_ref_ids: Dict[str, str] = {}
        for doc_id, doc in docs.items():
            if not doc.id:
                continue
            if doc_id != doc.id:
                update_ref_ids[doc_id] = doc.id

        references = await self.get_document_references(list(update_ref_ids.keys()))
        id_update = lambda id: id if id not in update_ref_ids else update_ref_ids[id]

        update_docs = {**docs}
        for ref in references:
            assert ref.id
            children = ref.metadata.get("children_nodes", [])
            links = ref.metadata.get("references", [])
            if children:
                ref.metadata["children_nodes"] = list(map(id_update, children))
            if links:
                ref.metadata["references"] = list(map(id_update, links))

            update_docs[ref.id] = ref

        old_docs = list(update_docs.keys())
        new_docs = list(update_docs.values())

        original_docs = await self.get_by_ids(old_docs, required=True)

        async with self._lock.writer_lock:
            self.storage.delete(old_docs)
            self.headers_storage.delete(old_docs)
            new_ids = await self._add_docs_internal(new_docs)
            print(f"New ids: {new_ids}")
            try:
                if callback is not None:
                    await callback()
                self._storage_updated = True
            except Exception:
                self.storage.delete(new_ids)
                with_crumbs = [
                    doc.id for doc in new_docs if doc.metadata.get("crumbs") and doc.id
                ]
                if with_crumbs:
                    self.headers_storage.delete(with_crumbs)
                await self._add_docs_internal(original_docs)
                new_ids = []

        return new_ids

    # Wraps FAISS get_by_ids
    async def get_by_ids(
        self, ids: List[str], required: bool = False
    ) -> List[Document]:
        if not self._initialized or self.storage is None:
            raise RuntimeError("VectorStore not initialized")

        docs: List[Document] = self.storage.get_by_ids(ids)
        if len(docs) < len(ids) and required:
            missing_ids = ids
            if len(docs) > 0:
                doc_ids = set([doc.id for doc in docs])
                missing_ids = list(set(ids) - doc_ids)
            raise ValueError(f"Documents: {','.join(missing_ids)} not found!")
        return docs

    async def cleanup(self):
        """Cleanup resources."""
        if (
            not self._initialized
            or self.storage is None
            or self.headers_storage is None
        ):
            return
        async with self._lock.writer_lock:
            if self._storage_updated:
                self.storage.save_local(self.storage_path)
                self.headers_storage.save_local(self.headers_path)


# Global singleton
vector_store = VectorStoreService()
