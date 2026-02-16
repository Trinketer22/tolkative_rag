import pytest
import random
from typing import Dict, List
from langchain_core.documents import Document
from pathlib import Path

from services.snippet_cache import (
    SnippetCacheService,
    MissingSnippetError,
    SnippetAlreadyExists,
    NotInitialized,
)


@pytest.fixture
def snippet_service_not_init(tmp_path_factory):
    """Fixture to provide a clean SnippetCacheService instance for each test."""
    snippet_path = str(tmp_path_factory.mktemp("test_snippets"))
    return SnippetCacheService(snippet_path=f"{snippet_path}/test_snippets.json")


@pytest.fixture
def snippet_service(snippet_service_not_init, sample_documents):
    snippet_service_not_init.initialize(initial_data=sample_documents)
    return snippet_service_not_init


@pytest.fixture
def sample_documents():
    """Fixture to provide sample Document objects."""
    return [
        Document(
            page_content="Snippet content 1", metadata={"lang": "TOLK"}, id="doc1"
        ),
        Document(
            page_content="Snippet content 2", metadata={"lang": "FunC"}, id="doc2"
        ),
        Document(
            page_content="Snippet content 3", metadata={"lang": "TACT"}, id="doc3"
        ),
    ]


class TestSnippetCacheService:
    @pytest.mark.asyncio
    async def test_initialization_with_initial_data_success(
        self, snippet_service, sample_documents
    ):
        assert snippet_service._initialized is True
        assert (await snippet_service.get_total_snippets()) == len(sample_documents)
        assert snippet_service._snippets["doc1"] == sample_documents[0]

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, snippet_service):
        with pytest.raises(RuntimeError, match="SnippetCached is alread initialized!"):
            await snippet_service.initialize(initial_data=[])

    @pytest.mark.asyncio
    async def test_get_not_initialized(self, snippet_service_not_init):
        with pytest.raises(NotInitialized):
            await snippet_service_not_init.get("doc1")

    @pytest.mark.asyncio
    async def test_get_existing_snippet(self, snippet_service, sample_documents):
        snippet = await snippet_service.get("doc1")
        assert snippet == sample_documents[0]

    @pytest.mark.asyncio
    async def test_get_non_existing_snippet(self, snippet_service):
        snippet = await snippet_service.get("nonexistent_doc")
        assert snippet is None

    @pytest.mark.asyncio
    async def test_get_req_existing_snippet(self, snippet_service, sample_documents):
        snippet = await snippet_service.get_req("doc1")
        assert snippet == sample_documents[0]

    @pytest.mark.asyncio
    async def test_get_req_non_existing_snippet(self, snippet_service):
        with pytest.raises(
            MissingSnippetError, match="Snippet nonexistent_doc is missing!"
        ):
            await snippet_service.get_req("nonexistent_doc")

    @pytest.mark.asyncio
    async def test_mget_not_initialized(self, snippet_service_not_init):
        with pytest.raises(NotInitialized):
            await snippet_service_not_init.mget(["doc1"])

    @pytest.mark.asyncio
    async def test_mget_multiple_snippets(self, snippet_service, sample_documents):
        results = await snippet_service.mget(["doc1", "doc3"])
        assert len(results) == 2
        assert sample_documents[0] in results
        assert sample_documents[2] in results

    @pytest.mark.asyncio
    async def test_mget_some_missing_not_required(
        self, snippet_service, sample_documents
    ):
        results = await snippet_service.mget(["doc1", "nonexistent_doc"])
        assert len(results) == 1
        assert sample_documents[0] in results

    @pytest.mark.asyncio
    async def test_mget_some_missing_required(self, snippet_service):
        with pytest.raises(MissingSnippetError):
            await snippet_service.mget(["doc1", "nonexistent_doc"], required=True)

    @pytest.mark.asyncio
    async def test_mget_with_filter_cb(self, snippet_service, sample_documents):
        def filter_cb(doc):
            return doc.metadata["lang"] == "TOLK"

        results = await snippet_service.mget(
            ["doc1", "doc2", "doc3"], filter_cb=filter_cb
        )
        assert len(results) == 1
        assert results[0] == sample_documents[0]
        assert results[0].metadata["lang"] == "TOLK"

    @pytest.mark.asyncio
    async def test_mget_req_success(self, snippet_service, sample_documents):
        results = await snippet_service.mget_req(["doc1", "doc3"])
        assert len(results) == 2
        assert sample_documents[0] in results
        assert sample_documents[2] in results

    @pytest.mark.asyncio
    async def test_mget_req_some_missing(self, snippet_service):
        with pytest.raises(MissingSnippetError):
            await snippet_service.mget_req(["doc1", "nonexistent_doc"])

    @pytest.mark.asyncio
    async def test_mget_dict_success(self, snippet_service, sample_documents):
        results = await snippet_service.mget_dict(["doc1", "doc2"])
        assert len(results) == 2
        assert results["doc1"] == sample_documents[0]
        assert results["doc2"] == sample_documents[1]

    @pytest.mark.asyncio
    async def test_mget_dict_with_missing_not_required(
        self, snippet_service, sample_documents
    ):
        results = await snippet_service.mget_dict(["doc1", "nonexistent_doc"])
        assert len(results) == 1
        assert results["doc1"] == sample_documents[0]
        assert "nonexistent_doc" not in results

    @pytest.mark.asyncio
    async def test_mget_dict_with_filter_cb(self, snippet_service, sample_documents):
        def filter_cb(doc):
            return doc.metadata["lang"] == "FunC"

        results = await snippet_service.mget_dict(
            ["doc1", "doc2", "doc3"], filter_cb=filter_cb
        )
        assert len(results) == 1
        assert results["doc2"] == sample_documents[1]
        assert results["doc2"].metadata["lang"] == "FunC"

    @pytest.mark.asyncio
    async def test_mget_dict_req_success(self, snippet_service, sample_documents):
        results = await snippet_service.mget_dict_req(["doc1", "doc3"])
        assert len(results) == 2
        assert results["doc1"] == sample_documents[0]
        assert results["doc3"] == sample_documents[2]

    @pytest.mark.asyncio
    async def test_mget_dict_req_some_missing(self, snippet_service):
        with pytest.raises(MissingSnippetError):
            await snippet_service.mget_dict_req(["doc1", "nonexistent_doc"])

    @pytest.mark.asyncio
    async def test_add_snippets_success(self, snippet_service):
        count_before = await snippet_service.get_total_snippets()
        new_snippets = {
            "new_doc1": Document(page_content="New 1", id="new_doc1"),
            "new_doc2": Document(page_content="New 2", id="new_doc2"),
        }
        added_ids = await snippet_service.add_snippets(new_snippets)
        assert sorted(added_ids) == sorted(["new_doc1", "new_doc2"])
        assert (await snippet_service.get_total_snippets()) == count_before + 2
        assert (await snippet_service.get_req("new_doc1")) == new_snippets["new_doc1"]

    @pytest.mark.asyncio
    async def test_cleanup(self, snippet_service, sample_documents):
        import json

        count_before = await snippet_service.get_total_snippets()
        new_snippets = {
            "new_doc1": Document(page_content="New 1", id="new_doc1"),
            "new_doc2": Document(page_content="New 2", id="new_doc2"),
        }
        added_ids = await snippet_service.add_snippets(new_snippets)
        assert sorted(added_ids) == sorted(["new_doc1", "new_doc2"])
        assert (await snippet_service.get_total_snippets()) == count_before + 2
        assert (await snippet_service.get_req("new_doc1")) == new_snippets["new_doc1"]

        test_path = Path(snippet_service.snippet_path)
        assert not test_path.exists()
        await snippet_service.cleanup()

        assert test_path.is_file()

        content = test_path.read_text(encoding="utf8")
        line_count = 0
        for saved_line in content.split("\n"):
            if saved_line.strip():
                line_count += 1
                try:
                    Document(**json.loads(saved_line))
                except json.JSONDecodeError:
                    pytest.fail("Failed to parse json")
        assert line_count == len(new_snippets) + len(sample_documents)

    @pytest.mark.asyncio
    async def test_add_snippets_already_exists(self, snippet_service):
        count_before = await snippet_service.get_total_snippets()
        new_snippets = {
            "doc1": Document(page_content="Updated doc1", id="doc1"),
            "new_doc2": Document(page_content="New 2", id="new_doc2"),
        }
        with pytest.raises(
            SnippetAlreadyExists, match=r"Snippets doc1 already exist!"
        ):  # Use r"" for regex
            await snippet_service.add_snippets(new_snippets)
        # Ensure snippets are not modified
        assert (await snippet_service.get_total_snippets()) == count_before

    @pytest.mark.asyncio
    async def test_update_snippets_success(self, snippet_service, sample_documents):
        pick_snippets = random.sample(
            sample_documents, k=random.randint(2, len(sample_documents))
        )
        upd_docs = {
            doc.id: Document(
                id=doc.id, page_content=doc.page_content + f" Updated content"
            )
            for doc in pick_snippets
            if doc.id
        }
        await snippet_service.update_snippets(upd_docs)

        after_update = await snippet_service.mget_req(upd_docs.keys())
        for doc_orig, doc_retr in zip(upd_docs.values(), after_update):
            assert doc_orig.page_content == doc_retr.page_content

    @pytest.mark.asyncio
    async def test_update_snippets_replace(self, snippet_service, sample_documents):
        pick_snippets = random.sample(
            sample_documents, k=random.randint(2, len(sample_documents))
        )
        pick_replace = random.sample(pick_snippets, k=len(pick_snippets) // 2)
        replace_ids = set([doc.id for doc in pick_replace if doc.id])

        upd_docs = {}
        upd_ids: List[str] = []
        for doc in pick_snippets:
            if not doc.id:
                continue
            new_content = f"{doc.page_content} updated"
            upd_ids.append(doc.id)

            if doc.id in replace_ids:
                new_doc = Document(id=f"{doc.id} updated", page_content=new_content)
                assert new_doc.id
                upd_ids.append(new_doc.id)
            else:
                new_doc = Document(id=doc.id, page_content=new_content)
            upd_docs[doc.id] = new_doc

        await snippet_service.update_snippets(upd_docs)
        after_upd = await snippet_service.mget_dict(upd_ids)

        for doc_id, doc in upd_docs.items():
            assert doc.id
            if doc_id in replace_ids:
                # Make sure all replaced ids are removed
                assert doc_id not in after_upd

            # New id should be present
            assert doc.id in after_upd
            # And content should match expected
            assert doc.page_content == after_upd[doc.id].page_content

    @pytest.mark.asyncio
    async def test_update_snippets_callback(self, snippet_service, sample_documents):
        pick_doc = random.choice(sample_documents)
        new_content = "Updated content"

        assert pick_doc.id
        assert pick_doc.page_content != new_content

        counter = 0

        async def inc_callback(updated: Dict[str, Document]):
            nonlocal counter
            counter += 1

        await snippet_service.update_snippets(
            {pick_doc.id: Document(id=pick_doc.id, page_content=new_content)},
            callback=inc_callback,
        )

        after_upd = await snippet_service.get_req(pick_doc.id)
        assert after_upd.page_content == new_content
        assert counter == 1

    @pytest.mark.asyncio
    async def test_update_snippets_rollback(self, snippet_service, sample_documents):
        pick_doc = random.choice(sample_documents)
        new_content = "Updated content"

        assert pick_doc.id
        assert pick_doc.page_content != new_content

        async def failed_cb(replace: Dict[str, str]):
            raise RuntimeError("Something went wrong")

        await snippet_service.update_snippets(
            {pick_doc.id: Document(id=pick_doc.id, page_content=new_content)},
            callback=failed_cb,
        )

        after_upd = await snippet_service.get_req(pick_doc.id)
        assert after_upd.page_content == pick_doc.page_content

    @pytest.mark.asyncio
    async def test_update_snippets_should_exits(self, snippet_service):
        new_doc2 = Document(page_content="Brand new doc2", id="new_doc2")
        new_snippets = {
            "new_doc2": new_doc2,
        }
        with pytest.raises(ValueError, match="Snippets new_doc2 not found"):
            await snippet_service.update_snippets(new_snippets)

    @pytest.mark.asyncio
    async def test_add_snippets_empty_dict(self, snippet_service):
        count_before = await snippet_service.get_total_snippets()
        added_ids = await snippet_service.add_snippets({})
        assert added_ids == []
        assert (await snippet_service.get_total_snippets()) == count_before

    @pytest.mark.asyncio
    async def test_delete_snippets_success(self, snippet_service, sample_documents):
        pick_snippets = [
            doc.id
            for doc in random.sample(
                sample_documents, k=random.randint(1, len(sample_documents) - 1)
            )
            if doc.id
        ]

        await snippet_service.delete_snippets(pick_snippets)

        after_upd = await snippet_service.mget(pick_snippets)
        assert all(doc is None for doc in after_upd)

    @pytest.mark.asyncio
    async def test_delete_snippets_not_found(self, snippet_service, sample_documents):
        pick_snippets = random.sample(
            sample_documents, k=random.randint(1, len(sample_documents) - 1)
        )
        pick_ids = [doc.id for doc in pick_snippets if doc.id]
        non_existing_id = "Not existing id"
        assert (await snippet_service.get(non_existing_id)) is None

        pick_ids.insert(random.randint(0, len(pick_snippets) - 1), non_existing_id)
        assert non_existing_id in pick_ids

        with pytest.raises(ValueError, match=f"Snippets {non_existing_id} not found!"):
            await snippet_service.delete_snippets(pick_ids)

        after_del = await snippet_service.mget(pick_ids)
        assert len(after_del) == len(pick_snippets)

        for orig_doc, doc_after in zip(pick_snippets, after_del):
            assert orig_doc.id and orig_doc.id == doc_after.id
            assert orig_doc.page_content == doc_after.page_content
