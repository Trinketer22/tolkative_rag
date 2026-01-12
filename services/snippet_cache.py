"""
Provides interface for snippet storage
"""

from typing import List, Optional, Callable, Dict
from config import settings
from langchain_core.documents import Document

# from langchain_community.retrievers import BM25Retriever
from utils.json import load_json_dump


class MissingSnippetError(Exception):
    """Exception for missing snippets"""

    pass


class NotInitialized(Exception):
    """Exception for not initialized service"""

    pass


class SnippetCacheService:
    #    snippet_index: Optional[BM25Retriever] = None
    """Snippet storage"""

    def __init__(self):
        self.snippet_path = settings.SNIPPETS_PATH
        self._snippets: Dict
        self._initialized = False

    def initialize(self):
        """
        Loading snippets
        Currently from json file
        But in production probably from
        redis cache
        """
        if self._initialized:
            raise RuntimeError("SnippetCached is alread initialized!")

        self._snippets = {}
        snip_list = load_json_dump(self.snippet_path, lambda doc: Document(**doc))
        # self.snippet_index = BM25Retriever.from_documents(snip_list,k = 10)

        for snip in snip_list:
            self._snippets[snip.id] = snip

        self._initialized = True

    async def get(self, id: str, required: bool = False) -> Optional[Document]:
        if not self._initialized:
            raise NotInitialized("Snippet storage is not initialized")

        snip = self._snippets.get(id)

        if required and snip is None:
            raise MissingSnippetError(f"Snippet {id} is missing!")

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
        return await self.get(id, required=True)

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
        for doc_id in ids:
            snip = self._snippets.get(doc_id)

            if required and snip is None:
                raise MissingSnippetError(f"Snippet {id} is missing!")
            res_list.append(snip)

        return res_list if filter is None else list(filter(filter_cb, res_list))

    async def mget_req(
        self, ids: List[str], filter_cb: Optional[Callable] = None
    ) -> List[Document]:
        return await self.mget(ids, required=True)

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

    async def mget_dict_req(
        self, ids: List[str], filter_cb: Optional[Callable] = None
    ) -> Dict[str, Document]:
        return await self.mget_dict(ids, filter_cb, required=True)


#
# Cache singleton
snippet_cache = SnippetCacheService()
