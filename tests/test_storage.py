from pathlib import Path
from typing import List, Tuple, Dict
import pytest
import random
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from services.vector_store import VectorStoreService

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def hierarchial_documents() -> List[Document]:
    return [
        Document(
            id="Earth orbit",
            page_content="Top document about earth and it's satelites",
            metadata={
                "crumbs": "Space>>Earth orbit",
                "children_nodes": [
                    "Earth",
                    "Moon",
                ],
                "topic": "space",
            },
        ),
        Document(
            id="Moon",
            page_content="Something about moon",
            metadata={
                "references": ["Earth", "Solar system info"],
                "crumbs": "Space>>Earth orbit>>Moon",
                "topic": "space",
            },
        ),
        Document(
            id="Earth",
            page_content="Earth stuff",
            metadata={
                "references": ["Moon", "Solar system info"],
                "crumbs": "Space>>Earth orbit>>Earth",
                "topic": "space",
            },
        ),
    ]


@pytest.fixture()
def default_documents(hierarchial_documents) -> List[Document]:
    return [
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
    ] + hierarchial_documents


@pytest.fixture(scope="function")
def faiss_db(
    embedder: HuggingFaceEmbeddings, faiss_factory, default_documents
) -> Tuple[FAISS, FAISS]:
    return faiss_factory(
        embedder,
        default_documents,
    )


@pytest.fixture(scope="module")
def storage_paths(tmp_path_factory) -> Tuple[Path, Path]:
    return tmp_path_factory.mktemp("main_index"), tmp_path_factory.mktemp(
        "headers_index"
    )


@pytest.fixture(scope="function")
def vec_store(faiss_db: Tuple[FAISS, FAISS], storage_paths) -> VectorStoreService:
    tmp_index, tmp_headers = storage_paths

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
async def test_delete_document(
    vec_store: VectorStoreService, default_documents: List[Document]
):
    del_candidates = [doc.id for doc in random.sample(default_documents, k=2) if doc.id]
    assert len(del_candidates) == 2

    assert len(await vec_store.get_by_ids(del_candidates)) == len(del_candidates)

    await vec_store.delete_docs([del_candidates[0]])

    assert len(await vec_store.get_by_ids(del_candidates)) == len(del_candidates) - 1

    with pytest.raises(ValueError, match=f"Documents: {del_candidates[0]} not found"):
        await vec_store.delete_docs(del_candidates)

    await vec_store.delete_docs([del_candidates[1]])

    assert len(await vec_store.get_by_ids(del_candidates)) == 0


@pytest.mark.asyncio
async def test_delete_hierarchial_document(
    vec_store: VectorStoreService, hierarchial_documents: List[Document]
):
    top_doc = hierarchial_documents[0]
    assert top_doc.id == "Earth orbit"
    children_ids = top_doc.metadata["children_nodes"]
    children_nodes = [
        doc for doc in hierarchial_documents if doc.id and doc.id in children_ids
    ]
    assert len(children_nodes) == 2
    assert all(len(doc.metadata["references"]) == 2 for doc in children_nodes)

    delete_idx = random.randint(0, 1)
    removed_doc = children_nodes[delete_idx]
    untouched_doc = children_nodes[(delete_idx + 1) % 2]
    assert removed_doc.id
    # Make sure reference was there in the first place
    assert removed_doc.id in untouched_doc.metadata["references"]

    await vec_store.delete_docs([removed_doc.id])
    assert untouched_doc.id

    # Let's get the updated versions
    after_update = await vec_store.get_by_ids(
        [top_doc.id, untouched_doc.id, removed_doc.id]
    )
    assert len(after_update) == 2
    top_after = after_update[0]
    assert top_after.id == top_doc.id

    untouched_after = after_update[1]
    assert untouched_after.id == untouched_doc.id

    updated_children = top_after.metadata["children_nodes"]

    # Removed expected id from children of the top doc
    # Other elements unchanged
    assert len(updated_children) == len(children_ids) - 1
    assert removed_doc.id not in updated_children
    assert all(doc_id in children_ids for doc_id in updated_children)

    # Same for references
    updated_references = untouched_after.metadata["references"]
    old_references = untouched_doc.metadata["references"]
    assert len(updated_references) == len(old_references) - 1
    assert removed_doc.id not in updated_references
    assert all(ref in old_references for ref in updated_references)


@pytest.mark.asyncio
async def test_update_documents(
    vec_store: VectorStoreService, default_documents: List[Document]
):
    picked_updates = random.sample(
        default_documents, k=random.randint(1, len(default_documents))
    )
    picked_ids: List[str] = []
    upd_dict: Dict[str, Document] = {}

    for update_doc in picked_updates:
        assert update_doc.id
        picked_ids.append(update_doc.id)
        new_meta = {**update_doc.metadata}
        assert "test_attribute" not in new_meta
        new_meta["test_attribute"] = (
            f"Test value {update_doc.id}:{random.randint(0, 1337)}"
        )
        new_doc = Document(
            id=update_doc.id, page_content=update_doc.page_content, metadata=new_meta
        )
        upd_dict[update_doc.id] = new_doc

    new_ids = await vec_store.update_documents(upd_dict)
    assert len(new_ids) == len(picked_updates)

    after_update = await vec_store.get_by_ids(picked_ids)
    for doc in after_update:
        assert doc.id
        assert (
            doc.metadata["test_attribute"]
            == upd_dict[doc.id].metadata["test_attribute"]
        )


@pytest.mark.asyncio
async def test_update_documents_new_id(
    vec_store: VectorStoreService, hierarchial_documents: List[Document]
):
    # Update everything except the top document
    target_docs = hierarchial_documents[1:]
    all_ids = [doc.id for doc in hierarchial_documents if doc.id]
    target_ids = all_ids[1:]
    updated_docs = [
        Document(
            id=f"{doc.id} updated", page_content=doc.page_content, metadata=doc.metadata
        )
        for doc in target_docs
        if doc.id
    ]
    assert len(target_docs) == len(updated_docs)
    upd_dict: Dict[str, Document] = {}

    for orig_doc, new_doc in zip(target_docs, updated_docs):
        assert orig_doc.id and new_doc.id
        upd_dict[orig_doc.id] = new_doc

    new_ids = await vec_store.update_documents(upd_dict)
    # Except that all documents will be updated, because
    # Child ids are impacted -> top document should be updated too
    assert len(new_ids) == len(hierarchial_documents)

    after_update = await vec_store.get_by_ids(all_ids)
    assert len(after_update) == len(hierarchial_documents)

    def check_id_update(orig_ids: List[str], updated_ids: List[str]):
        assert len(updated_ids) == len(orig_ids)
        for orig_id, upd_id in zip(orig_ids, updated_ids):
            if orig_id in target_ids:
                assert upd_id == f"{orig_id} updated"
            else:
                assert upd_id == orig_id

    for orig_doc, updated_doc in zip(hierarchial_documents, after_update):
        children = orig_doc.metadata.get("children_nodes", [])
        refs = orig_doc.metadata.get("references", [])
        if len(children) > 0:
            updated_children = updated_doc.metadata.get("children_nodes", [])
            check_id_update(children, updated_children)
        if len(refs) > 0:
            updated_refs = updated_doc.metadata.get("references", [])
            check_id_update(refs, updated_refs)


@pytest.mark.asyncio
async def test_update_callback(
    vec_store: VectorStoreService, default_documents: List[Document]
):
    pick_doc = random.choice(default_documents)
    assert pick_doc.id
    new_doc = Document(
        id=pick_doc.id, page_content="New page content", metadata=pick_doc.metadata
    )
    assert new_doc.page_content != pick_doc.page_content

    test_counter = 0

    async def update_counter():
        nonlocal test_counter
        test_counter += 1

    await vec_store.update_documents({pick_doc.id: new_doc}, callback=update_counter)
    after_update = await vec_store.get_by_ids([pick_doc.id], required=True)
    assert after_update[0].page_content == new_doc.page_content
    assert test_counter == 1


@pytest.mark.asyncio
async def test_update_revert_on_failed_callback(
    vec_store: VectorStoreService, default_documents: List[Document]
):
    pick_doc = random.choice(default_documents)
    assert pick_doc.id
    new_doc = Document(
        id=pick_doc.id, page_content="New page content", metadata=pick_doc.metadata
    )
    assert new_doc.page_content != pick_doc.page_content

    async def failed_cb():
        raise RuntimeError("Something went wrong!")

    new_ids = await vec_store.update_documents(
        {pick_doc.id: new_doc}, callback=failed_cb
    )
    assert len(new_ids) == 0

    after_rollback = await vec_store.get_by_ids([pick_doc.id])
    assert after_rollback[0].page_content == pick_doc.page_content


@pytest.mark.asyncio
async def test_clean_up(
    vec_store: VectorStoreService, storage_paths: Tuple[Path, Path]
):
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

    await vec_store.add_documents([neptune, plutone])

    # Storage directories should be empty
    assert not any(storage_paths[0].iterdir())
    assert not any(storage_paths[1].iterdir())

    await vec_store.cleanup()

    assert any(storage_paths[0].iterdir())
    assert any(storage_paths[1].iterdir())


@pytest.mark.asyncio
async def test_duplicate_id_document(vec_store: VectorStoreService, default_documents):
    test_doc = random.choice(default_documents)
    just_id = Document(id=test_doc.id, page_content="Totally different text")
    for doc in [test_doc, just_id]:
        with pytest.raises(ValueError) as exc_info:
            await vec_store.add_documents([doc])
            assert "duplicated" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_by_ids(
    vec_store: VectorStoreService, default_documents: List[Document]
):
    k_ids = 2
    test_ids = [doc.id for doc in random.sample(default_documents, k=k_ids) if doc.id]
    assert len(test_ids) == k_ids
    found_docs = {}
    for doc in default_documents:
        if doc.id in test_ids:
            found_docs[doc.id] = doc
    assert len(found_docs) == k_ids

    res_docs = await vec_store.get_by_ids(test_ids)
    for doc in res_docs:
        assert doc.id in found_docs
        assert found_docs[doc.id].page_content == doc.page_content


@pytest.mark.asyncio
async def test_get_by_ids_skip_elements(
    vec_store: VectorStoreService, default_documents: List[Document]
):
    k_ids = random.randint(1, len(default_documents))
    test_ids = [doc.id for doc in random.sample(default_documents, k=k_ids) if doc.id]
    assert len(test_ids) == k_ids

    test_ids.insert(random.randint(0, k_ids - 1), "nonexistent_doc")
    found_docs = {}
    for doc in default_documents:
        if doc.id in test_ids:
            found_docs[doc.id] = doc
    assert len(found_docs) == k_ids

    res_docs = await vec_store.get_by_ids(test_ids)
    assert len(res_docs) == k_ids
    for doc in res_docs:
        assert doc.id in found_docs
        assert found_docs[doc.id].page_content == doc.page_content


@pytest.mark.asyncio
async def test_get_by_ids_required(
    vec_store: VectorStoreService, default_documents: List[Document]
):
    random_ids = [doc.id for doc in random.sample(default_documents, k=2) if doc.id]
    for test_payload in [
        ["nonexistent_doc"],
        ["nonexistent_doc", *random_ids],
        [*random_ids, "nonexistent_doc"],
    ]:
        with pytest.raises(ValueError, match="Documents: nonexistent_doc not found"):
            await vec_store.get_by_ids(test_payload, required=True)


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
