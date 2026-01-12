import logging
from typing import List, Dict, Optional, Tuple, Any
from langchain_core.documents import Document
from services.snippet_cache import snippet_cache
from config import settings
from utils.tokens import calc_token_count

log = logging.getLogger(__name__)


def fence_code(snip: Document):
    lang = snip.metadata.get("lang", "")
    if len(snip.page_content.split("\n")) <= 1 and not lang:
        return f"`{snip.page_content}`"
    return f"""
```{lang}
{snip.page_content}
```
    """


async def render_docs_batch(
    docs: List[Document],
    token_limit: int = settings.CTX_TOKEN_LIMIT,
    language_filter: Optional[List[str]] = None,
) -> Tuple[List[str], int]:
    """Async version (though not strictly necessary for dict lookups)."""

    if not docs:
        return ([], 0)

    doc_snippet_mapping = {}
    all_snippet_ids = set()

    for doc in docs:
        snippet_refs = doc.metadata.get("snippets", [])
        doc_snippet_mapping[doc.id] = snippet_refs
        all_snippet_ids.update(snip_ref["id"] for snip_ref in snippet_refs)

    log.debug(
        f"Batch rendering {len(docs)} docs with {len(all_snippet_ids)} unique snippets"
    )

    # Run in thread pool (useful if snippets_index becomes I/O later)
    snippet_filter = None
    if language_filter is not None:
        snippet_filter = lambda snip: snip.metadata.get("lang") not in language_filter

    snippets_map = await snippet_cache.mget_dict_req(
        list(all_snippet_ids), filter_cb=snippet_filter
    )
    # Render synchronously (fast string operations)

    rendered_tokens = 0
    rendered_docs = []
    for doc in docs:
        if rendered_tokens + doc.metadata.get("token_count", 0) > token_limit:
            log.debug(f"Out of token limit, while processing {doc.id}")
            break

        rendered_doc = render_single_doc(doc, doc_snippet_mapping[doc.id], snippets_map)
        doc_size = calc_token_count(rendered_doc)

        if rendered_tokens + doc_size > token_limit:
            log.debug(f"Out of token limit, while processing {doc.id}")
            break
        rendered_tokens += doc_size
        rendered_docs.append(rendered_doc)

    return (rendered_docs, rendered_tokens)


def _build_doc_url(doc_path: str) -> str:
    """Extract documentation URL from file path."""
    if "docs-data" not in doc_path:
        return ""

    parsed_path = doc_path.split("/")[1:]
    file_path = parsed_path[-1].split(".")[0]
    parsed_path[-1] = file_path

    return f'doc-url="https://docs.ton.org/{"/".join(parsed_path)}"'


def _build_doc_score(score: float):
    return f'doc-relativity="{score}"'


def render_single_doc(
    doc: Document, snippet_refs: List[Dict], snippets_cache: Dict[str, Any]
) -> str:
    """Render a single document using pre-fetched snippets."""

    total_delta = len(doc.metadata.get("crumbs", "")) + 2
    doc_content = doc.page_content

    for snip_ref in snippet_refs:
        snip_id = snip_ref["id"]

        # No lookup! Just dictionary access to pre-fetched data
        snippet_obj = snippets_cache.get(snip_id)
        if not snippet_obj:
            continue

        fenced_block = fence_code(snippet_obj)
        start_idx = total_delta + snip_ref["pos"]
        doc_content = doc_content[:start_idx] + fenced_block + doc_content[start_idx:]
        total_delta += len(fenced_block)

    # Don't pollute context with the empty tags
    if len(doc_content.strip()) == 0:
        return ""

    # Build context string
    doc_attributes = [
        f'id="{doc.id}"',
        f'orig-doc="{doc.metadata.get("from", "")}"',
        f'concept="{doc.metadata["crumbs"]}"',
        _build_doc_url(doc.metadata.get("from", "")),
        _build_doc_score(doc.metadata.get("rerank_score", 0.0))
        if settings.EXPOSE_SCORING
        else "",
    ]

    # Python mad ways of doing stuff
    return f"<context {' '.join([attr for attr in doc_attributes if attr])}>{doc_content}</context>"
