from langchain_core.documents import Document
import pytest
import pytest_asyncio
from unittest.mock import patch
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from httpx import ASGITransport, AsyncClient
from models.response import AddMarkdownResponse
from services.vector_store import VectorStoreService
from services.snippet_cache import SnippetCacheService
from config import settings
from main import app

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def active_app():
    async with app.router.lifespan_context(app):
        yield app


@pytest_asyncio.fixture(scope="function")
async def _client_base(active_app):
    transport = ASGITransport(app=active_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(scope="function")
def client_unauth(_client_base: AsyncClient):
    yield _client_base


@pytest.fixture(scope="function")
def client(_client_base: AsyncClient):
    _client_base.headers.update(
        {"Authorization": f"Bearer {settings.ADMIN_AUTH_TOKEN}"}
    )
    yield _client_base


@pytest.fixture(scope="module")
def storage_paths(tmp_path_factory) -> Tuple[Path, Path]:
    return tmp_path_factory.mktemp("main_index"), tmp_path_factory.mktemp(
        "headers_index"
    )


@pytest.fixture(scope="module")
def default_documents() -> List[Document]:
    return [
        Document(
            id="TestDoc",
            page_content="Test content",
            metadata={"from": "test.md", "crumbs": "Docs>>Misc>>Test"},
        ),
        Document(
            id="42",
            page_content="42 bro",
            metadata={"from": "test.md", "crumbs": "Docs>>Misc>>Test"},
        ),
    ]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def faiss_db(
    embedder: HuggingFaceEmbeddings, faiss_factory, default_documents
) -> Tuple[FAISS, FAISS]:
    return faiss_factory(
        embedder,
        default_documents,
    )


@pytest.fixture(scope="module", autouse=True)
def mock_storage(faiss_db):
    def mock_initialize(self: VectorStoreService):
        self.storage = faiss_db[0]
        self.headers_storage = faiss_db[1]
        self._initialized = True

    async def mock_cleanup(self: VectorStoreService):
        pass

    with (
        patch.object(VectorStoreService, "initialize", new=mock_initialize),
        patch.object(VectorStoreService, "cleanup", new=mock_cleanup),
    ):
        yield


@pytest.fixture(scope="module", autouse=True)
def mock_snippets():
    def mock_initialize(self: SnippetCacheService, initial_data=None):
        self._initialized = True
        self._snippets = {}

    async def mock_cleanup(self):
        pass

    with (
        patch.object(SnippetCacheService, "initialize", new=mock_initialize),
        patch.object(SnippetCacheService, "cleanup", new=mock_cleanup),
    ):
        yield


# ──────────────────────────────────────────────────────────────
# Authorization tests
# ──────────────────────────────────────────────────────────────


async def test_read_doc_unauthorized(client_unauth: AsyncClient, default_documents):
    response = await client_unauth.get("/admin/documents/TestDoc")
    assert response.status_code == 401
    resp = response.json()
    assert not isinstance(resp, list)


async def test_get_documents_unauthorized(client_unauth: AsyncClient):
    response = await client_unauth.get(
        "/admin/documents/", params={"doc_id": ["TestDoc"]}
    )
    assert response.status_code == 401


async def test_create_document_unauthorized(client_unauth: AsyncClient):
    response = await client_unauth.post(
        "/admin/documents/",
        json={"page_content": "x", "metadata": {}, "id": "x", "type": "Document"},
    )
    assert response.status_code == 401


async def test_update_document_unauthorized(client_unauth: AsyncClient):
    response = await client_unauth.put(
        "/admin/documents/TestDoc",
        json={"page_content": "x", "metadata": {}, "id": "TestDoc", "type": "Document"},
    )
    assert response.status_code == 401


async def test_get_file_unauthorized(client_unauth: AsyncClient):
    response = await client_unauth.get("/admin/files/test.md")
    assert response.status_code == 401


async def test_delete_file_unauthorized(client_unauth: AsyncClient):
    response = await client_unauth.delete("/admin/files/test.md")
    assert response.status_code == 401


async def test_add_markdown_unauthorized(client_unauth: AsyncClient):
    response = await client_unauth.post(
        "/admin/files/add_markdown",
        files={"file": ("test.md", b"# Hello")},
    )
    assert response.status_code == 401


async def test_update_markdown_unauthorized(client_unauth: AsyncClient):
    response = await client_unauth.put(
        "/admin/files/add_markdown/test.md",
        files={"file": ("test.md", b"# Hello", "text/markdown")},
    )
    assert response.status_code == 401


async def test_wrong_token_unauthorized(_client_base: AsyncClient):
    _client_base.headers = {"Authorization": "Bearer wrong-token"}
    response = await _client_base.get("/admin/documents/TestDoc")
    assert response.status_code == 401


# ──────────────────────────────────────────────────────────────
# GET /documents/{doc_id}
# ──────────────────────────────────────────────────────────────


async def test_read_doc(client: AsyncClient, default_documents):
    response = await client.get("/admin/documents/TestDoc")
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 1
    parsed_doc = Document(**resp[0])
    assert parsed_doc.id == default_documents[0].id
    assert parsed_doc.page_content == default_documents[0].page_content


async def test_read_doc_not_found(client: AsyncClient):
    response = await client.get("/admin/documents/NonExistentDoc")
    assert response.status_code == 200
    resp = response.json()
    assert resp == []


# ──────────────────────────────────────────────────────────────
# GET /documents/?doc_id=...&doc_id=...
# ──────────────────────────────────────────────────────────────


async def test_get_documents_single(client: AsyncClient, default_documents):
    response = await client.get("/admin/documents/", params={"doc_id": ["TestDoc"]})
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 1
    parsed_doc = Document(**resp[0])
    assert parsed_doc.id == default_documents[0].id


async def test_get_documents_multiple(client: AsyncClient, default_documents):
    response = await client.get(
        "/admin/documents/", params={"doc_id": ["TestDoc", "42"]}
    )
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == len(default_documents)
    # Should return only the one that exists
    found_docs = {}
    for doc in default_documents:
        found_docs[doc.id] = doc

    for doc in resp:
        assert doc["id"] in found_docs
        assert found_docs[doc["id"]].page_content == doc["page_content"]


async def test_get_documents_missing_param(client: AsyncClient):
    response = await client.get("/admin/documents/")
    assert response.status_code == 422  # validation error, doc_id is required


# ──────────────────────────────────────────────────────────────
# POST /documents/  (create)
# ──────────────────────────────────────────────────────────────


async def test_create_document(client: AsyncClient):
    new_doc = Document(
        id="NewDoc",
        page_content="Brand new content",
        metadata={"from": "brand/new/doc.md", "crumbs": "Docs>>New"},
    ).model_dump(exclude_unset=True)
    response = await client.post("/admin/documents/", json=new_doc)
    assert response.status_code == 200
    # Verify the document was added
    get_response = await client.get("/admin/documents/NewDoc")
    assert get_response.status_code == 200
    resp = get_response.json()
    assert len(resp) == 1
    parsed = Document(**resp[0])
    assert parsed.id == "NewDoc"
    assert parsed.page_content == "Brand new content"
    assert parsed.metadata == new_doc["metadata"]


# ──────────────────────────────────────────────────────────────
# PUT /documents/{doc_id}  (update)
# ──────────────────────────────────────────────────────────────


async def test_update_document(client: AsyncClient):
    # First create a document to update
    create_doc = {
        "id": "UpdateMe",
        "page_content": "Original content",
        "metadata": {"from": "update.md", "crumbs": "Docs>>Update"},
        "type": "Document",
    }
    await client.post("/admin/documents/", json=create_doc)

    updated_doc = {
        "id": "UpdateMe",
        "page_content": "Updated content",
        "metadata": {"from": "update.md", "crumbs": "Docs>>Update"},
        "type": "Document",
    }
    response = await client.put("/admin/documents/UpdateMe", json=updated_doc)
    assert response.status_code == 200

    # Verify the update
    get_response = await client.get("/admin/documents/UpdateMe")
    assert get_response.status_code == 200
    resp = get_response.json()
    assert len(resp) == 1
    parsed = Document(**resp[0])
    assert parsed.page_content == "Updated content"


# ──────────────────────────────────────────────────────────────
# GET /files/{file_path}
# ──────────────────────────────────────────────────────────────


async def test_get_documents_by_file(client: AsyncClient, default_documents):
    response = await client.get("/admin/files/test.md")
    assert response.status_code == 200
    resp = response.json()
    assert isinstance(resp, list)
    assert len(resp) >= 1
    file_sources = [Document(**d).metadata.get("from") for d in resp]
    assert "test.md" in file_sources


async def test_get_documents_by_file_not_found(client: AsyncClient):
    response = await client.get("/admin/files/nonexistent.md")
    assert response.status_code == 200
    resp = response.json()
    assert resp == []


# ──────────────────────────────────────────────────────────────
# DELETE /files/{file_path}
# ──────────────────────────────────────────────────────────────


async def test_delete_documents_by_file(client: AsyncClient):
    # First add a document under a specific file
    doc = {
        "id": "DeleteFileDoc",
        "page_content": "To be deleted by file",
        "metadata": {
            "from": "/just/for/del/deleteme.md",
            "crumbs": "Docs>>Delete",
            "snippets": [],
        },
        "type": "Document",
    }
    await client.post("/admin/documents/", json=doc)

    response = await client.delete("/admin/files//just/for/del/deleteme.md")
    assert response.status_code == 200
    resp = response.json()
    assert resp["file_path"] == doc["metadata"]["from"]
    assert len(resp["removed_docs"]) == 1
    assert resp["removed_docs"][0] == doc["id"]

    # Verify the document is gone
    get_response = await client.get("/admin/documents/DeleteFileDoc")
    assert get_response.status_code == 200
    assert get_response.json() == []


async def test_delete_nonexistent_file(client: AsyncClient):
    response = await client.delete("/admin/files/nofile.md")
    assert response.status_code == 200
    resp = response.json()
    assert resp["file_path"] == "nofile.md"
    assert len(resp["removed_docs"]) == 0
    assert len(resp["removed_snippers"]) == 0


# ──────────────────────────────────────────────────────────────
# POST /files/add_markdown  (create from .md file)
# ──────────────────────────────────────────────────────────────


async def test_add_markdown_file(client: AsyncClient):
    md_content = """# Top level header
Hop

## Examples

Hey, here is some snippets
```
    Some snippet
```

```
    Some other snippet
```

## Conclusion

La la ley
        """
    response = await client.post(
        "/admin/files/add_markdown",
        files={
            "file": ("somepath/newfile.md", md_content.encode("utf8"), "text/markdown")
        },
    )
    assert response.status_code == 200
    resp = AddMarkdownResponse(**response.json())
    assert resp.file_path == "somepath/newfile.md"
    assert len(resp.new_documents) == 3
    assert len(resp.new_snippets) == 2

    rendered_docs = (
        await client.get(
            f"/admin/documents/", params={"doc_id": resp.new_documents, "render": True}
        )
    ).json()
    # print(f"Rendered docs {rendered_docs}")

    flat_list = [rendered for rendered in rendered_docs.values()]
    assert "Hop" in flat_list[0]
    assert "here is some snippets" in flat_list[1]
    assert "Some snippet" in flat_list[1]
    assert "Some other snippet" in flat_list[1]

    assert "La la ley" in flat_list[2]


async def test_add_markdown_mdx_file(client: AsyncClient):
    md_content = b"# MDX Title\n\nSome MDX content.\n"
    response = await client.post(
        "/admin/files/add_markdown",
        files={"file": ("newfile.mdx", md_content, "text/markdown")},
    )
    assert response.status_code == 200
    resp = response.json()
    assert resp["file_path"] == "newfile.mdx"


async def test_add_markdown_invalid_extension(client: AsyncClient):
    response = await client.post(
        "/admin/files/add_markdown",
        files={"file": ("file.txt", b"# Hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only .md and .mdx files allowed" in response.json()["detail"]


async def test_add_markdown_invalid_utf8(client: AsyncClient):
    invalid_bytes = b"\x80\x81\x82\x83"
    response = await client.post(
        "/admin/files/add_markdown",
        files={"file": ("bad.md", invalid_bytes, "text/markdown")},
    )
    assert response.status_code == 400
    assert "not valid utf8" in response.json()["detail"]


async def test_add_markdown_empty_content(client: AsyncClient):
    response = await client.post(
        "/admin/files/add_markdown",
        files={"file": ("empty.md", b"", "text/markdown")},
    )
    assert response.status_code == 200
    resp = response.json()
    assert resp["file_path"] == "empty.md"
    assert resp["new_documents"] == []
    assert resp["new_snippets"] == []


# ──────────────────────────────────────────────────────────────
# PUT /files/add_markdown/{update_path}  (update from .md file)
# ──────────────────────────────────────────────────────────────


async def test_update_markdown_file(client: AsyncClient):
    # First create the document via POST
    md_content = b"# Update Title\n\nOriginal paragraph.\n"
    file_path = "updatefile.md"
    post_response = await client.post(
        "/admin/files/add_markdown",
        files={"file": (file_path, md_content, "text/markdown")},
    )
    assert post_response.status_code == 200
    post_resp = post_response.json()
    assert len(post_resp["new_documents"]) > 0
    # print(f"Docs before {post_resp['new_documents']}")

    # Now update via PUT
    updated_content = b"# Update Title\n\nUpdated paragraph content.\n"
    put_response = await client.put(
        f"/admin/files/add_markdown/{file_path}",
        files={"file": (file_path, updated_content, "text/markdown")},
    )
    assert put_response.status_code == 201
    put_resp = put_response.json()
    assert put_resp["file_path"] == file_path

    updated_docs = (await client.get(f"/admin/files/{file_path}")).json()
    assert len(updated_docs) == 1
    # print(f"Docs after {updated_docs}")
    assert "Updated paragraph content" in updated_docs[0]["page_content"]
    assert "Original paragraph" not in updated_docs[0]["page_content"]


async def test_update_markdown_file_with_snippets(client: AsyncClient):
    # First create the document via POST
    def doc_with_snippet(snip_text: str):
        md_content = f"""
# Update Title
``` tolk

{snip_text}
```
Original paragraph
        """
        return md_content.encode("utf8")

    test_content = doc_with_snippet("Old snippet text")
    file_path = "update_snippets.md"
    post_response = await client.post(
        "/admin/files/add_markdown",
        files={"file": (file_path, test_content, "text/markdown")},
    )
    assert post_response.status_code == 200
    post_resp = post_response.json()
    assert len(post_resp["new_documents"]) > 0

    docs_before = (await client.get(f"/admin/files/{file_path}")).json()
    assert len(docs_before) == 1

    # Now update via PUT
    updated_content = doc_with_snippet("New snippet text dsadsasda")
    put_response = await client.put(
        f"/admin/files/add_markdown/{file_path}",
        files={"file": (file_path, updated_content, "text/markdown")},
    )
    assert put_response.status_code == 201
    put_resp = put_response.json()
    assert put_resp["file_path"] == file_path

    updated_docs = (await client.get(f"/admin/files/{file_path}")).json()
    assert len(updated_docs) == 1
    # print(f"Docs after {updated_docs}")
    assert updated_docs[0]["id"] == docs_before[0]["id"]
    assert updated_docs[0]["page_content"] == docs_before[0]["page_content"]
    assert updated_docs[0]["metadata"] != docs_before[0]["metadata"]["snippets"]

    rendered_docs = (
        await client.get(
            f"/admin/documents/{updated_docs[0]['id']}", params={"render": True}
        )
    ).json()
    assert len(rendered_docs) == 1

    assert updated_docs[0]["id"] in rendered_docs
    rendered_content = rendered_docs[updated_docs[0]["id"]]
    assert "New snippet text" in rendered_content
    assert "Old snippet text" not in rendered_content


async def test_update_markdown_path_mismatch(client: AsyncClient):
    md_content = b"# Mismatch\n\nContent.\n"
    response = await client.put(
        "/admin/files/add_markdown/other.md",
        files={"file": ("mismatch.md", md_content, "text/markdown")},
    )
    assert response.status_code == 400
    assert "update_path should match the file path" in response.json()["detail"]


async def test_update_markdown_nonexistent_file(client: AsyncClient):
    md_content = b"# No Existing\n\nContent.\n"
    response = await client.put(
        "/admin/files/add_markdown/neveradded.md",
        files={"file": ("neveradded.md", md_content, "text/markdown")},
    )
    assert response.status_code == 400
    assert "No documents found" in response.json()["detail"]


async def test_update_markdown_invalid_extension(client: AsyncClient):
    response = await client.put(
        "/admin/files/add_markdown/test.txt",
        files={"file": ("test.txt", b"# Hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only .md and .mdx files allowed" in response.json()["detail"]


async def test_update_markdown_invalid_utf8(client: AsyncClient):
    invalid_bytes = b"\x80\x81\x82\x83"
    response = await client.put(
        "/admin/files/add_markdown/bad.md",
        files={"file": ("bad.md", invalid_bytes, "text/markdown")},
    )
    assert response.status_code == 400
    assert "not valid utf8" in response.json()["detail"]


async def test_update_markdown_empty_becomes_noop(client: AsyncClient):
    # First create
    md_content = b"# Will Empty\n\nSome content.\n"
    await client.post(
        "/admin/files/add_markdown",
        files={"file": ("willempty.md", md_content, "text/markdown")},
    )
    # Update with empty (process_md returns no docs)
    response = await client.put(
        "/admin/files/add_markdown/willempty.md",
        files={"file": ("willempty.md", b"", "text/markdown")},
    )
    # Empty content produces 0 docs, so early return with 200
    assert response.status_code == 200
    resp = response.json()
    assert resp["new_documents"] == []
    assert resp["new_snippets"] == []
