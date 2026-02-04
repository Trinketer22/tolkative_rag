import pytest
from langchain_core.documents import Document

from services.snippet_cache import (
    SnippetCacheService,
    MissingSnippetError,
    SnippetAlreadyExists,
    NotInitialized,
)


@pytest.fixture
def snippet_service_not_init():
    """Fixture to provide a clean SnippetCacheService instance for each test."""
    return SnippetCacheService()


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
        assert snippet_service._snippets["new_doc1"] == new_snippets["new_doc1"]

    @pytest.mark.asyncio
    async def test_add_snippets_already_exists_no_merge(self, snippet_service):
        count_before = await snippet_service.get_total_snippets()
        new_snippets = {
            "doc1": Document(page_content="Updated doc1", id="doc1"),
            "new_doc2": Document(page_content="New 2", id="new_doc2"),
        }
        with pytest.raises(
            SnippetAlreadyExists, match=r"Snippets doc1 already exist!"
        ):  # Use r"" for regex
            await snippet_service.add_snippets(new_snippets, merge=False)
        # Ensure snippets are not modified
        assert (await snippet_service.get_total_snippets()) == count_before

    @pytest.mark.asyncio
    async def test_add_snippets_merge_existing_and_new(
        self, snippet_service, sample_documents
    ):
        count_before = await snippet_service.get_total_snippets()
        updated_doc1 = Document(page_content="Updated doc1 content", id="doc1")
        new_doc3 = Document(page_content="New doc 3 content", id="new_doc3")
        new_snippets = {
            "doc1": updated_doc1,
            "new_doc3": new_doc3,
        }
        added_ids = await snippet_service.add_snippets(new_snippets, merge=True)
        assert sorted(added_ids) == sorted(["doc1", "new_doc3"])
        [test_doc1, test_doc3, old_doc2] = await snippet_service.mget_req(
            ["doc1", "new_doc3", "doc2"]
        )
        assert test_doc1.page_content == updated_doc1.page_content
        assert test_doc3.page_content == new_doc3.page_content
        assert old_doc2.page_content == sample_documents[1].page_content
        assert (await snippet_service.get_total_snippets()) == count_before + 1

    @pytest.mark.asyncio
    async def test_add_snippets_merge_only_new(self, snippet_service):
        new_doc2 = Document(page_content="Brand new doc2", id="new_doc2")
        new_snippets = {
            "new_doc2": new_doc2,
        }
        added_ids = await snippet_service.add_snippets(new_snippets, merge=True)
        assert sorted(added_ids) == sorted(["new_doc2"])

    @pytest.mark.asyncio
    async def test_add_snippets_empty_dict(self, snippet_service):
        count_before = await snippet_service.get_total_snippets()
        added_ids = await snippet_service.add_snippets({})
        assert added_ids == []
        assert (await snippet_service.get_total_snippets()) == count_before

    @pytest.mark.asyncio
    async def test_add_snippets_merge_empty_dict(self, snippet_service):
        count_before = await snippet_service.get_total_snippets()
        added_ids = await snippet_service.add_snippets({}, merge=True)
        assert added_ids == []
        assert (await snippet_service.get_total_snippets()) == count_before
