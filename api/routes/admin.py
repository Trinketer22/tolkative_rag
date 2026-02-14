from typing import Dict, List, Optional, Tuple
from annotated_doc import Doc
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    Security,
    UploadFile,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.documents import Document
from config import settings
from core.documents import process_md
from models.response import AddMarkdownResponse, RemoveFileResponse
from services.snippet_cache import snippet_cache
from services.vector_store import vector_store
from core.rendering import render_docs_batch
from utils.async_helpers import execute_async


_sec_scheme = HTTPBearer()


async def authorize(credentials: HTTPAuthorizationCredentials = Security(_sec_scheme)):
    token = credentials.credentials
    if token != settings.ADMIN_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=settings.ADMIN_UNAUTHORIZED_BANNER,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


router = APIRouter(dependencies=[Depends(authorize)])


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, render: Optional[bool] = False):
    docs = await vector_store.get_by_ids([doc_id])
    if not render:
        return docs
    rendered = (await render_docs_batch(docs))[0]
    return {doc.id: rendered for doc, rendered in zip(docs, rendered)}


@router.get("/documents/")
async def get_douments(doc_id: List[str] = Query(...), render: Optional[bool] = False):
    docs = await vector_store.get_by_ids(doc_id)
    if not render:
        return docs
    rendered = (await render_docs_batch(docs))[0]
    return {doc.id: rendered for doc, rendered in zip(docs, rendered)}


@router.post("/documents/")
async def create_document(doc: Document):
    return await vector_store.add_documents([doc])


@router.put("/documents/{doc_id}")
async def update_document(doc_id: str, doc: Document):
    return await vector_store.update_documents({doc_id: doc})


@router.get("/files/{file_path:path}")
async def get_documents_by_file(file_path: str):
    return await vector_store.get_documents_from_file(file_path)


@router.delete("/files/{file_path:path}")
async def delete_documents_by_file(file_path: str) -> RemoveFileResponse:
    old_docs = await vector_store.get_documents_from_file(file_path)
    old_snippets = [
        snippet
        for doc in old_docs
        for snippet in doc.metadata.get("snippets", [])
        if snippet
    ]

    deleted_docs = await vector_store.delete_docs(
        [doc.id for doc in old_docs if doc.id]
    )
    deleted_snippets = await snippet_cache.delete_snippets(old_snippets)

    return RemoveFileResponse(
        file_path=file_path,
        removed_docs=deleted_docs,
        removed_snippers=deleted_snippets,
    )


@router.post("/files/add_markdown")
@router.put("/files/add_markdown/{update_path:path}")
async def add_markdown(
    response: Response, file: UploadFile = File(...), update_path: Optional[str] = None
):
    file_path = file.filename
    if file_path is None or not (
        file_path.endswith(".md") or file_path.endswith(".mdx")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .md and .mdx files allowed",
        )
    file_contents = await file.read()

    try:
        md_text = file_contents.decode("utf8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not valid utf8 text",
        )

    if update_path is None:
        old_docs = await vector_store.get_documents_from_file(file_path)
        if len(old_docs) > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {file_path} already exists, use PUT to update",
            )

    docs, snippets = await execute_async(lambda: process_md(file_path, md_text))
    if len(docs) == 0:
        return AddMarkdownResponse(
            file_path=file_path, new_documents=[], new_snippets=[]
        )

    added_snippets = []
    added_documents = []

    if update_path is None:
        added_snippets = await snippet_cache.add_snippets(snippets)
        added_documents = await vector_store.add_documents(docs)
        response.status_code = status.HTTP_200_OK
    elif update_path == file_path:
        old_docs = await vector_store.get_documents_from_file(update_path)
        if len(old_docs) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No documents found for path {update_path}. Use POST instead to create new document",
            )

        old_snippets = set(
            [
                snippet["id"]
                for doc in old_docs
                for snippet in doc.metadata.get("snippets", [])
                if snippet and "id" in snippet
            ]
        )

        new_snippets: Dict[str, Document] = {}
        update_snippets: Dict[str, Document] = {}
        update_documents: Dict[str, Document] = {}
        new_documents: List[Document] = []
        remove_documents: List[str] = []
        new_doc_ids = set([doc.id for doc in docs if doc.id])

        del_snippets: List[str] = list(old_snippets - snippets.keys())

        for k, v in snippets.items():
            if k not in old_snippets:
                new_snippets[k] = v
            else:
                update_snippets[k] = v

        crumbs_map: Dict[str, str] = {}

        for doc in old_docs:
            doc_crumbs = doc.metadata.get("crumbs")
            if (not doc.id) or (not doc_crumbs):
                continue
            crumbs_map[doc_crumbs] = doc.id

        for doc in docs:
            doc_crumbs = doc.metadata.get("crumbs")
            if (not doc.id) or (not doc_crumbs):
                continue

            old_doc_id = crumbs_map.get(doc_crumbs)
            if old_doc_id:
                update_documents[old_doc_id] = doc
            else:
                new_documents.append(doc)

        # Kinda lame, but such is life
        remove_documents = [
            doc.id
            for doc in old_docs
            if doc.id and doc.id not in new_doc_ids and doc.id not in update_documents
        ]

        added_snippets = await snippet_cache.add_snippets(new_snippets)
        added_documents = await vector_store.add_documents(new_documents)

        await snippet_cache.update_snippets(update_snippets)
        # Currently there is a race condition between those calls,
        # but from practical standpoint and local usage it is ok

        await vector_store.update_documents(update_documents)

        await vector_store.delete_docs(remove_documents)
        await snippet_cache.delete_snippets(del_snippets)

        # In complete honor of REST standard update should return 201
        response.status_code = status.HTTP_201_CREATED

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"On markdown update, update_path should match the file path expected: {file_path} got {update_path}",
        )

    return AddMarkdownResponse(
        file_path=file_path, new_documents=added_documents, new_snippets=added_snippets
    )
