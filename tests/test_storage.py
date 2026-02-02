from typing import Tuple
from numpy import exp
import pytest
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from services.embedding import embedding_service
from services.vector_store import VectorStoreService

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder() -> HuggingFaceEmbeddings:
    embedding_service.initialize()
    if embedding_service.embedder is None:
        raise RuntimeError("Failed to initialize embedder")
    return embedding_service.embedder


@pytest.fixture(scope="function")
def faiss_db(embedder: HuggingFaceEmbeddings) -> Tuple[FAISS, FAISS]:
    documents = [
        Document(
            id="0",
            page_content="The quick brown fox jumps over the lazy dog.",
            metadata={"topic": "animals"},
        ),
        Document(
            id="2",
            page_content="A delicious apple pie recipe with cinnamon.",
            metadata={"topic": "food"},
        ),
        Document(
            id="3",
            page_content="A delicious mars pie with a lot of strawbery.",
            metadata={"topic": "food"},
        ),
        Document(
            id="4",
            page_content="The planet Mars is known as the Red Planet.",
            metadata={"topic": "space"},
        ),
        Document(
            id="5",
            page_content="Python functions are defined using the def keyword.",
            metadata={"topic": "coding"},
        ),
    ]

    header_docs = [
        Document(id=doc.id, page_content=" ".join(doc.page_content.split()[:4]))
        for doc in documents
    ]

    # 3. Create and return the vector store
    return FAISS.from_documents(documents, embedder), FAISS.from_documents(
        header_docs, embedder
    )


@pytest.fixture(scope="function")
def vec_store(faiss_db: Tuple[FAISS, FAISS], tmp_path_factory) -> VectorStoreService:
    # tmp_models = tmp_path_factory.mktemp("models")
    tmp_index = tmp_path_factory.mktemp("main_index")
    tmp_headers = tmp_path_factory.mktemp("headers_index")

    # monkeypatch.setattr(settings, "MODEL_CACHE_DIR", tmp_models)
    storage = VectorStoreService(
        storage=faiss_db[0],
        header_storage=faiss_db[1],
        storage_path=tmp_index,
        headers_path=tmp_headers,
    )
    storage.initialize()
    return storage


@pytest.fixture(scope="function")
def mmr_vec_store(
    faiss_db: FAISS, tmp_path_factory, embedder: HuggingFaceEmbeddings
) -> VectorStoreService:
    tmp_index = tmp_path_factory.mktemp("main_index")
    tmp_headers = tmp_path_factory.mktemp("headers_index")

    # monkeypatch.setattr(settings, "MODEL_CACHE_DIR", tmp_models)
    storage = VectorStoreService(
        storage=faiss_db,
        header_storage=FAISS.from_documents(
            [Document(page_content="Test header")], embedder
        ),
        storage_path=tmp_index,
        headers_path=tmp_headers,
        search_type="MMR",
    )
    storage.initialize()
    return storage


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_retrieval_without_keyword_match(vec_store: VectorStoreService):
    """
    Proves we aren't just doing keyword search
    """
    animals_query = "movements of animals"

    desert_query = "baking a fruit dessert"

    for query in [animals_query, desert_query]:
        results = await vec_store.search(query, top_k=1)
        top_doc = results[0]

        if query == animals_query:
            expected_topic = "animals"
        else:
            expected_topic = "food"
            assert "apple pie" in top_doc.page_content

        assert top_doc.metadata["topic"] == expected_topic, (
            f"Expected '{expected_topic}', but got '{top_doc.metadata['topic']}'"
        )


@pytest.mark.asyncio
async def test_semantic_retrieval_from_headers(vec_store: VectorStoreService):
    """
    Check that headers search works as expected
    """
    query = "procedures definition"
    from_full = await vec_store.search(query, top_k=1)
    from_headers = await vec_store.search(query, top_k=1, from_headers=True)

    assert from_headers[0].id is not None
    # Id should match
    assert from_full[0].id == from_headers[0].id
    assert from_full[0].metadata["topic"] == "coding"
    # But not content (that would prove they come from different indexes)
    assert from_full[0].page_content != from_headers[0].page_content

    by_ids = await vec_store.get_by_ids([from_headers[0].id])
    assert by_ids[0].page_content == from_full[0].page_content


@pytest.mark.asyncio
async def test_similarity_score_distances(vec_store: VectorStoreService):
    """
    Verifies that the L2 distance score is working meaningfully.
    Lower score = Closer distance = Better match.
    """
    query = "solar system astronomy"

    # Returns List[Tuple[Document, float]]
    results = await vec_store.search(query, top_k=3, with_score=True)
    assert isinstance(results[0], tuple)

    top_doc, top_score = results[0]
    second_doc, second_score = results[1]

    assert top_doc.metadata["topic"] == "space"

    # Top result must be closer (lower score) than the following
    assert top_score < second_score, (
        f"Top score ({top_score}) should be lower than second score ({second_score})"
    )


@pytest.mark.asyncio
async def test_add_documents(vec_store: VectorStoreService):
    """
    Should be able to add documents
    """
    neptune = Document(
        id="neptune",
        page_content="The Neptune is known as the Blue Planet.",
        metadata={"topic": "space", "crumbs": "Space>>Planets>>Neptune"},
    )
    plutone = Document(
        id="plutone",
        page_content="Pluto is not actually considered planet, but a Kuiper belt object.",
        metadata={"topic": "space", "crumbs": "Space>>Planets>>Pluto"},
    )

    query = "Tell me about neptune and pluto"
    test_resp = await vec_store.search(query, top_k=2)
    relevant_docs = [
        doc
        for doc in test_resp
        if doc.page_content == neptune.page_content
        or doc.page_content == plutone.page_content
    ]

    assert len(relevant_docs) == 0

    await vec_store.add_documents([neptune, plutone])

    test_resp = await vec_store.search(query, top_k=2)

    relevant_docs = [
        doc
        for doc in test_resp
        if doc.page_content == neptune.page_content
        or doc.page_content == plutone.page_content
    ]

    assert len(relevant_docs) == 2
    assert relevant_docs[0].page_content != relevant_docs[1].page_content

    # Now check that headers index is also populated after addition
    header_res = await vec_store.search(query, from_headers=True, top_k=2)

    assert len(header_res) == 2
    # Id match
    assert header_res[0].id is not None and header_res[0].id in ["neptune", "plutone"]
    assert header_res[1].id is not None and header_res[1].id in ["neptune", "plutone"]
    assert header_res[0].id != header_res[1].id

    # Text don't match, since headers text is populated from crumbs
    assert not any(
        doc.page_content == header_res[0].page_content
        or doc.page_content == header_res[1].page_content
        for doc in relevant_docs
    )


@pytest.mark.asyncio
async def test_filter_lambda(vec_store: VectorStoreService):
    test_resp = await vec_store.search("Tell me about mars", top_k=3)
    assert any(doc for doc in test_resp if doc.metadata["topic"] == "food")
    test_resp = await vec_store.search(
        "Tell me about mars",
        top_k=3,
        filter_fn=lambda doc: doc.metadata["topic"] == "space",
    )

    assert not any(doc for doc in test_resp if doc.metadata["topic"] == "food")


'''
@pytest.mark.asyncio
async def test_mmr_diversity(mmr_vec_store: VectorStoreService):
    """
    Tests Maximal Marginal Relevance (MMR).
    """
    # Create a near-duplicate to test diversity algorithms
    extra_doc = [
        Document(
            page_content="Mars has two moons named Phobos and Deimos.",
            metadata={"topic": "space"},
        )
    ]
    await vec_store.add_documents(extra_doc)

    query = "tell me about Mars"

    # Fetch 2 docs using MMR, looking at a pool of 10
    results = await vec_store.search(query, top_k=2)

    assert len(results) == 2
    # Ensure both results are space-related
    assert all(d.metadata["topic"] == "space" for d in results)
'''
