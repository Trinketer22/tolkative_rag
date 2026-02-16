from typing import Callable, Optional
import pytest
from langchain_huggingface import HuggingFaceEmbeddings
from services.embedding import embedding_service
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from typing import Tuple, List


# Global embedder fixture
@pytest.fixture(scope="module")
def embedder() -> HuggingFaceEmbeddings:
    embedding_service.initialize()
    if embedding_service.embedder is None:
        raise RuntimeError("Failed to initialize embedder")
    return embedding_service.embedder


def create_faiss_db(
    embedder: HuggingFaceEmbeddings,
    documents: List[Document],
    header_docs: Optional[List[Document]] = None,
) -> Tuple[FAISS, FAISS]:
    if header_docs is None:
        header_docs = [
            Document(id=doc.id, page_content=" ".join(doc.page_content.split()[:4]))
            for doc in documents
        ]

    # 3. Create and return the vector store
    return FAISS.from_documents(documents, embedder), FAISS.from_documents(
        header_docs, embedder
    )


@pytest.fixture(scope="module")
def faiss_factory() -> Callable[
    [HuggingFaceEmbeddings, List[Document], Optional[List[Document]]],
    Tuple[FAISS, FAISS],
]:
    return create_faiss_db
