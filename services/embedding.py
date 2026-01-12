"""
Embedding service with batching and caching
Wraps HuggingFace embeddings
"""

import asyncio
import hashlib
import logging
from typing import List, Optional, Dict
from collections import OrderedDict
import numpy as np
import unicodedata

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from utils.async_helpers import execute_async

log = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service with caching.

    Features:
    - Batch embedding for efficiency
    - LRU cache for repeated queries
    - Thread-safe operations
    """

    def __init__(
        self,
        model_name: str = "",
        cache_folder: str = "",
        max_cache_size: int = 10000,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.cache_folder = cache_folder or settings.MODEL_CACHE_DIR
        self.max_cache_size = max_cache_size

        # Embedder (loaded on initialize)
        self.embedder: Optional[HuggingFaceEmbeddings] = None

        # Cache: query_hash -> embedding vector
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # Initialization flag
        self._initialized = False

        # Stats
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "embeddings_computed": 0,
            "batch_embeddings": 0,
        }

    def initialize(self):
        """Initialize the embedding model (call on startup)."""
        if self._initialized:
            return

        log.info(f"Loading embedding model: {self.model_name}")

        # Load model
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.model_name,
            cache_folder=self.cache_folder,
        )

        self._initialized = True
        log.info(f"Embedding model loaded: {self.model_name}")

    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query with caching.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector (list of floats)
        """
        if not self._initialized:
            raise RuntimeError(
                "EmbeddingService not initialized. Call initialize() first."
            )

        # Check cache
        cache_key = self._get_cache_key(text)

        cached = self._get_from_cache(cache_key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            log.debug(f"Embedding cache HIT: {text[:50]}...")
            return cached.tolist()

        # Cache miss - compute embedding
        self.stats["cache_misses"] += 1
        log.debug(f"Embedding cache MISS: {text[:50]}...")

        embedding = await execute_async(lambda: self.embedder.embed_query(text))

        self.stats["embeddings_computed"] += 1

        # Store in cache
        self._add_to_cache(cache_key, np.array(embedding))

        return embedding

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents with batching and caching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            raise RuntimeError(
                "EmbeddingService not initialized. Call initialize() first."
            )

        if not texts:
            return []

        # Check which embeddings are cached
        embeddings = [None] * len(texts)
        texts_to_compute = []
        text_to_idx = {}

        for idx, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached = self._get_from_cache(cache_key)

            if cached is not None:
                embeddings[idx] = cached.tolist()
                self.stats["cache_hits"] += 1
            else:
                text_to_idx[len(texts_to_compute)] = idx
                texts_to_compute.append(text)
                self.stats["cache_misses"] += 1

        # Batch compute missing embeddings
        if len(texts_to_compute) > 0:
            log.debug(
                f"Computing {len(texts_to_compute)}/{len(texts)} embeddings (batch)"
            )

            computed_embeddings = await execute_async(
                lambda: self.embedder.embed_documents(texts_to_compute)
            )

            self.stats["embeddings_computed"] += len(texts_to_compute)
            self.stats["batch_embeddings"] += 1

            # Fill in computed embeddings and cache them
            for compute_idx, embedding in enumerate(computed_embeddings):
                original_idx = text_to_idx[compute_idx]
                embeddings[original_idx] = embedding

                # Cache the computed embedding
                cache_key = self._get_cache_key(texts_to_compute[compute_idx])
                self._add_to_cache(cache_key, np.array(embedding))

        log.info(f"Embedded {len(texts)} texts ({self.stats['cache_hits']} cached)")

        return embeddings

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        normalized = unicodedata.normalize("NFKC", text.strip().lower())

        # Hash for consistent key
        return hashlib.md5(normalized.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache (thread-safe)."""
        if key in self._embedding_cache:
            # Move to end (LRU)
            self._embedding_cache.move_to_end(key)
            return self._embedding_cache[key]
        return None

    def _add_to_cache(self, key: str, embedding: np.ndarray):
        """Add embedding to cache with size limit."""
        self._embedding_cache[key] = embedding

        # Enforce size limit (LRU eviction)
        while len(self._embedding_cache) > self.max_cache_size:
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
            log.debug(
                f"Evicted oldest embedding from cache (size: {len(self._embedding_cache)})"
            )

    def clear_cache(self):
        """Clear the embedding cache."""
        count = len(self._embedding_cache)
        self._embedding_cache.clear()
        log.info(f"Cleared embedding cache ({count} entries)")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (
            self.stats["cache_hits"] / total_requests if total_requests > 0 else 0.0
        )

        return {
            "cache_size": len(self._embedding_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hit_rate": hit_rate,
            "model_name": self.model_name,
            **self.stats,
        }

    async def cleanup(self):
        """Cleanup resources on shutdown."""
        log.info("Cleaning up embedding service...")
        self.clear_cache()
        # HuggingFace embeddings don't need explicit cleanup


# Global singleton instance
embedding_service = EmbeddingService()
