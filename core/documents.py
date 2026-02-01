"""
Mainly borrowed from the PrepareData notebook
"""

import pathlib
import markdown2 as markdown
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
import unicodedata
import hashlib
import re
from langchain_core.documents import Document
from typing import Dict, List, Optional, Tuple

_jsx_regex = re.compile(
    r"""^import\s+                     # the word “import” + whitespace
        \{([^}]+)\}\s+                # everything inside the braces → group 1
        from\s+                       # the word “from” + whitespace
        (['"])(/?([^'"]+))\.jsx\2     # quote (group 2), optional leading /,
                                      # the path without extension → group 4,
                                      # then “.jsx” and the same closing quote
        ;?\s*                        # optional trailing semicolon
    """,
    re.VERBOSE | re.MULTILINE,
)


def process_code(code: str, lang: str, concept: str, desc: str):
    doc_id = hashlib.sha256("||".join([code, concept]).encode("utf8")).hexdigest()
    codeDoc = Document(
        page_content=code,
        id=doc_id,
        metadata={
            "lang": lang.lower(),
            "desc": desc,
            "word_count": len(code.split()),
            "token_count": (len(code) + 3) // 4,  # Rough anthropic estimate
            "concept": concept,
        },
    )
    return codeDoc


def filter_jsx(content: str):
    # Let's start with something super dumb here.
    # Import is by far the most anoying token, rest could be ignored for now.
    return _jsx_regex.sub("", content)


def create_document(
    title: str,
    doc_chunks: List[str],
    path: str,
    references: List[str],
    ref_snippets: List[Dict],
    crumbs: List[str],
    snippets: Dict[str, Document],
    children_nodes: Optional[List[str]] = None,
):
    crumb_str = ">>".join(crumbs)
    # Put the crumbs string at the top instead of title, so the
    # whole hierarchy participates in scoring.
    doc_content = f"{crumb_str}\n\n" + filter_jsx(" ".join(doc_chunks))
    doc_id = hashlib.sha256("||".join([doc_content, path]).encode("utf8")).hexdigest()
    file_url = pathlib.Path(*(pathlib.Path(path).parts[1:]))
    url_from = "/" + str(file_url.parent / file_url.stem)

    doc_meta = {
        "concept": title,
        "word_count": len(doc_content.split()),
        "token_count": (len(doc_content) + 3) // 4,  # Rough anthropic estimate
        "from": path,
        "url_from": url_from,
        "child_nodes": children_nodes if children_nodes is not None else [],
        "references": references,
        "snippets": ref_snippets,
        "crumbs": crumb_str,
    }
    new_doc = Document(id=doc_id, page_content=doc_content, metadata=doc_meta)

    for snip_ref in ref_snippets:
        snip = snippets[snip_ref["id"]]
        snip.metadata["parent_doc"] = new_doc.id
        new_doc.metadata["token_count"] += (len(snip.page_content) + 3) // 4
        new_doc.metadata["word_count"] += len(snip.page_content.split())

    return new_doc


def add_to_hierarchy(docId: str, hierarchy: List[Document]):
    for doc in hierarchy:
        doc.metadata["child_nodes"].append(docId)


def process_md(
    path: str, md_text: str, custom_title="", custom_crumbs=None, skip_top=False
) -> Tuple[List[Document], Dict[str, Document]]:
    raw_md = unicodedata.normalize("NFKC", md_text)
    # raw_md  = md_path.read_text(encoding="utf8")
    md = markdown.Markdown(
        extras=["metadata", "fenced-code-blocks", "highlightjs-lang"]
    )
    text = md.convert(raw_md)
    markup = BeautifulSoup(text, "html.parser")
    title = (
        custom_title
        if custom_title
        else md.metadata["title"].strip('"')
        if "title" in md.metadata
        else ""
    )
    crumbs = [title] if title else []
    initial_crumbs = custom_crumbs if custom_crumbs else []
    heading_level = len(crumbs)
    chapter_length = 0
    references = []
    ref_snippets = []
    snippets: Dict[str, Document] = {}
    chapter_chunks = []
    last_added_text = ""
    doc_hierarchy = []
    new_docs = []
    skip_set = set(["code", "h1", "h2", "h3", "h4", "h5", "h6"])
    for node in markup.descendants:
        # skip text that is already extracted
        if isinstance(node, NavigableString):
            if node.parent and node.parent.name in skip_set:
                # print(f"Skipping {node.parent.name}")
                continue
            text = str(node).strip()
            if text:
                chapter_length += len(text) + int(chapter_length > 0)
                chapter_chunks.append(text)
            continue
        if isinstance(node, Tag):
            # print(node.text)
            # Potentially one of the heading tags h1 - h6
            if len(node.name) == 2 and node.name[0] == "h":
                new_lvl = ord(node.name[1]) - 49 + 1  # Char codes 1 - 6 index 0 - 5
                # Should never happen
                if new_lvl < 0 or new_lvl > 6:
                    continue

                new_doc = create_document(
                    title,
                    doc_chunks=chapter_chunks,
                    path=path,
                    references=references,
                    ref_snippets=ref_snippets,
                    snippets=snippets,
                    crumbs=initial_crumbs + crumbs,
                )
                new_docs.append(new_doc)
                # docs.append(new_doc)

                title = (
                    node.get_text(strip=True, separator=" ").replace('"', "").strip()
                )  # Strip quote from titles for escaping simplicity
                # print(title)
                # print(node)
                last_added_text = ""
                references = []
                ref_snippets = []
                chapter_chunks = []
                chapter_length = 0
                next_lvl = new_lvl > heading_level
                prev_lvl = new_lvl < heading_level
                # print(f"New{new_lvl}\n{heading_level}")
                # print(node)
                heading_level = new_lvl
                if next_lvl:
                    crumbs.append(title)
                    add_to_hierarchy(new_doc.id, doc_hierarchy)
                    doc_hierarchy.append(new_doc)
                elif prev_lvl:
                    # print(crumbs)
                    # print(doc_hierarchy)
                    doc_hierarchy = doc_hierarchy[0 : new_lvl - 1]
                    add_to_hierarchy(new_doc.id, doc_hierarchy)
                    doc_hierarchy.append(new_doc)
                    crumbs = crumbs[0 : new_lvl - 1] + [title]
                else:
                    crumbs[-1] = title
                    add_to_hierarchy(new_doc.id, doc_hierarchy)

                    """
                    Hierarchy is for the parents.
                    If lelvel is not changed, should not
                    touch it.
                    if len(doc_hierarchy) > 0:
                        doc_hierarchy[-1] = new_doc
                    else:
                        doc_hierarchy = [new_doc]
                    """

                # print(f"New crumbs {'>'.join(crumbs)} level {new_lvl}")
            elif node.name == "code":
                code_lang = node.get("class")
                code_text = node.text
                lang = ""

                if code_lang:
                    lang = code_lang[0]
                # Leave inlined only oneliners with no language def
                # Theese often can be term, and useful in vector search
                if len(code_text.split("\n")) <= 1 and not lang:
                    last_added_text = f"`{code_text.strip()}`"
                    if last_added_text:
                        chapter_chunks.append(last_added_text)
                        # Data chunk length + delimiter
                        chapter_length += len(last_added_text) + int(chapter_length > 0)
                else:
                    code_doc = process_code(code_text, lang, title, last_added_text)
                    snippets[code_doc.id] = code_doc
                    ref_snippets.append({"id": code_doc.id, "pos": chapter_length})

            elif node.name == "a":
                ref = node.get("href")
                if ref:
                    references.append(ref)

    if len(title) > 0 or len(chapter_chunks) > 0:
        # print(f"Adding last paragraph {title}")
        new_doc = create_document(
            title,
            doc_chunks=chapter_chunks,
            path=path,
            references=references,
            ref_snippets=ref_snippets,
            snippets=snippets,
            crumbs=initial_crumbs + crumbs,
        )
        add_to_hierarchy(new_doc.id, doc_hierarchy)
        new_docs.append(new_doc)

    return (new_docs[1:] if skip_top else new_docs, snippets)
