"""
Provides interface for snippet storage
"""

import logging
from typing import List, Optional, Callable, Awaitable, Dict
from config import settings
from langchain_core.documents import Document
from aiorwlock import RWLock

# from langchain_community.retrievers import BM25Retriever
from utils.json import load_json_dump, save_json_dump


class MissingSnippetError(Exception):
    """Exception for missing snippets"""

    pass


class SnippetAlreadyExists(Exception):
    """Exception for adding already existing snippers"""

    pass


class NotInitialized(Exception):
    """Exception for not initialized service"""

    pass


log = logging.getLogger(__name__)


class SnippetCacheService:
    #    snippet_index: Optional[BM25Retriever] = None
    """Snippet storage"""

    def __init__(self, snippet_path: str = settings.SNIPPETS_PATH):
        self.snippet_path = snippet_path
        self._snippets: Dict[str, Document]
        self._lock = RWLock()
        self._initialized = False
        self._modified = False

    def initialize(self, initial_data: Optional[List[Document]] = None):
        """
        Loading snippets
        Currently from json file
        But in production probably from
        redis cache
        """
        if self._initialized:
            raise RuntimeError("SnippetCached is alread initialized!")

        self._snippets = {}
        snip_list = (
            load_json_dump(self.snippet_path, lambda doc: Document(**doc))
            if initial_data is None
            else initial_data
        )
        # self.snippet_index = BM25Retriever.from_documents(snip_list,k = 10)

        for snip in snip_list:
            assert snip.id
            self._snippets[snip.id] = snip

        self._initialized = True

    async def get_total_snippets(self):
        async with self._lock.reader_lock:
            return len(self._snippets)

    async def get(self, id: str) -> Optional[Document]:
        if not self._initialized:
            raise NotInitialized("Snippet storage is not initialized")

        async with self._lock.reader_lock:
            snip = self._snippets.get(id)

        return snip

    # This performs bm25 keyword search over snippets.
    # I don't know how that would scale if snippets
    # are stored elsewhere (Redis?), but that's
    # a problem for another day.
    # Maybe we shouldn't even use it.
    async def lookup_snippet_index(self, query: str):
        raise RuntimeError("Not implemented!")
        # return self.snippet_index.invoke(query)

    async def get_req(self, id: str) -> Document:
        snip = await self.get(id)
        if not snip:
            raise MissingSnippetError(f"Snippet {id} is missing!")
        return snip

    async def mget(
        self,
        ids: List[str],
        filter_cb: Optional[Callable] = None,
        required: bool = False,
    ) -> List[Document]:
        if not self._initialized:
            raise NotInitialized("Snippet storage is not initialized")
        # In case of redis this would just wrap mget
        res_list = []
        async with self._lock.reader_lock:
            for doc_id in ids:
                snip = self._snippets.get(doc_id)

                if required and snip is None:
                    raise MissingSnippetError(f"Snippet {id} is missing!")
                res_list.append(snip)

        return res_list if filter is None else list(filter(filter_cb, res_list))

    async def mget_req(
        self, ids: List[str], filter_cb: Optional[Callable] = None
    ) -> List[Document]:
        return await self.mget(ids, filter_cb=filter_cb, required=True)

    async def mget_dict(
        self,
        ids: List[str],
        filter_cb: Optional[Callable] = None,
        required: bool = False,
    ) -> Dict[str, Document]:
        docs_list = await self.mget(ids, filter_cb, required)
        docs_dict = {}

        for doc in docs_list:
            if doc is not None:
                docs_dict[doc.id] = doc

        return docs_dict

    async def update_snippets(
        self,
        snippets: Dict[str, Document],
        callback: Optional[Callable[[Dict[str, str]], Awaitable[None]]] = None,
    ) -> List[str]:
        if not self._initialized:
            raise NotInitialized("Snippet storage is not initialized")

        if len(snippets) == 0:
            return []

        updated_ids: List[str] = []
        not_found = []
        update_dict: Dict[str, Document] = {}
        replace_dict: Dict[str, str] = {}
        orig_dict: Dict[str, Document] = {}

        async with self._lock.writer_lock:
            for doc_id, doc in snippets.items():
                if not doc.id:
                    continue
                if doc_id in self._snippets:
                    updated_ids.append(doc_id)
                    orig_dict[doc_id] = self._snippets[doc_id]
                    # In that case, it's a replace operation
                    if doc_id != doc.id:
                        replace_dict[doc_id] = doc.id
                        updated_ids.append(doc.id)
                        del self._snippets[doc_id]
                    update_dict[doc.id] = doc
                else:
                    not_found.append(doc_id)

            if len(not_found) > 0:
                raise ValueError(f"Snippets {','.join(not_found)} not found!")

            self._snippets.update(update_dict)
            try:
                if callback is not None:
                    await callback(replace_dict)
                self._modified = True
            except Exception as e:
                # Roll back
                for rep_id in replace_dict.values():
                    # Clear replaced keys
                    del self._snippets[rep_id]

                # Update original keys
                self._snippets.update(orig_dict)

            return updated_ids

    async def delete_snippets(self, snippet_ids: List[str]):
        not_found: List[str] = []

        if not self._initialized:
            raise NotInitialized("Snippet storage is not initialized")

        if len(snippet_ids) == 0:
            return []

        async with self._lock.writer_lock:
            for doc_id in snippet_ids:
                if doc_id not in self._snippets:
                    not_found.append(doc_id)
            if len(not_found) > 0:
                raise ValueError(f"Snippets {','.join(not_found)} not found!")

            for doc_id in snippet_ids:
                del self._snippets[doc_id]

        self._modified = True
        return snippet_ids

    async def add_snippets(self, snippets: Dict[str, Document]) -> List[str]:
        if len(snippets) == 0:
            return []
        async with self._lock.writer_lock:
            intersections = list(self._snippets.keys() & snippets.keys())
            if len(intersections) > 0:
                raise SnippetAlreadyExists(
                    f"Snippets {','.join(intersections)} already exist!"
                )
            self._snippets.update(snippets)
            self._modified = True

        return list(snippets.keys())

    async def mget_dict_req(
        self, ids: List[str], filter_cb: Optional[Callable] = None
    ) -> Dict[str, Document]:
        return await self.mget_dict(ids, filter_cb, required=True)

    async def cleanup(self):
        if self._modified:
            save_json_dump(self.snippet_path, self._snippets)
        self._initialized = False


#
# Cache singleton
snippet_cache = SnippetCacheService()
