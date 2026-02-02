from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.documents import Document
from typing import List, Sequence, Optional
from concurrent.futures import ThreadPoolExecutor
from services.embedding import embedding_service
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from utils.async_helpers import get_thread_pool
from config import settings
import asyncio
import logging

log = logging.getLogger(__name__)


class CrossEncoderRerankerWithScores(CrossEncoderReranker):
    score_threshold: float = 0
    executor: Optional[ThreadPoolExecutor] = None
    _owns_executor: bool = False

    def __init__(self, *args, max_workers: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        # Dedicated thread pool for CPU-intensive reranking
        if self.executor is not None:
            self._owns_executor = True
        else:
            log.debug("Creating separate executor for re-ranker")
            self.executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="reranker"
            )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        if len(documents) == 0:
            return []

        scores = self.model.score([(query, doc.page_content) for doc in documents])

        docs_with_scores = list(zip(documents, scores))
        docs_with_scores.sort(key=lambda doc: doc[1], reverse=True)

        result_docs = []
        for doc, score in docs_with_scores:
            # Since array is sorted already, we continue till first miss.
            if score < self.score_threshold:
                break
            doc.metadata["rerank_score"] = float(score)
            result_docs.append(doc)

        # print(f"{ query } Top doc: {docs_with_scores[1]}")
        log.debug(f"Reranked {len(documents)} -> {len(result_docs)} documents")
        return result_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        if len(documents) == 0:
            return []

        # Run blocking scoring in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, lambda: self.compress_documents(documents, query)
        )

    def __del__(self):
        """Only shutdown if we own the executor."""
        if self._owns_executor and self.executor:
            self.executor.shutdown(wait=False)


class RerankingService:
    cross_encoder: Optional[CrossEncoderRerankerWithScores] = None

    def __init__(
        self,
        reranker_model: str = settings.RERANKER_MODEL,
        cache_dir: str = settings.MODEL_CACHE_DIR,
    ):
        self.model_name = reranker_model
        self.model_dir = cache_dir

        self._initilized = False

    def initialize(self):
        if self._initilized:
            raise RuntimeError("Reranking service already initialized!")

        reranker_model = HuggingFaceCrossEncoder(
            model_name=self.model_name, model_kwargs={"cache_folder": self.model_dir}
        )
        self.cross_encoder = CrossEncoderRerankerWithScores(
            model=reranker_model,
            score_threshold=settings.RERANK_THRESHOLD,
            executor=get_thread_pool(),
        )
        self._initilized = True

    async def rerank(self, documents: List[Document], query: str):
        if not self._initilized or self.cross_encoder is None:
            raise RuntimeError("Reranker is not initialized")
        return await self.cross_encoder.acompress_documents(documents, query)


reranker = RerankingService()
