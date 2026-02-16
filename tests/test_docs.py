import pytest
import re
from pathlib import Path
from core.documents import process_md

test_doc_path = "docs-data/languages/func/known-issues.mdx"
test_doc = Path(test_doc_path).read_text(encoding="utf8")


def test_parse_doc():
    """
    Quickly check that number of paragraphs, links and snippets match.
    Snippet match the contents
    Crumbs match content and level
    """

    headings = re.findall(r"^(#{1,6})\s+(.*)$", test_doc, re.MULTILINE)
    links = re.findall(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)", test_doc, re.MULTILINE)
    docs, snippets = process_md(test_doc_path, test_doc)
    # Initial title is extracted from metadata
    known_title = "Known issues"
    assert headings is not None
    assert len(docs) == len(headings) + 1
    assert docs[0].metadata["concept"] == known_title

    parsed_links = sum(map(lambda doc: doc.metadata["references"], docs), [])

    assert len(parsed_links) == len(links)
    links_set = set(parsed_links)

    for link in links:
        assert link[1] in links_set

    # print(docs)
    for i in range(1, len(docs)):
        # print(i)
        # ` get dropped from the headers
        # It doesn't make any semantic difference in terms of search, so won't fix it.
        heading: str = headings[i - 1][1].replace("`", "")
        doc = docs[i]

        assert doc.metadata["concept"] == heading
        crumbs: list[str] = doc.metadata["crumbs"].split(">>")
        assert crumbs[-1] == heading
        assert crumbs[0] == known_title
        # print(heading)
        # print(crumbs)
        heading_lvl = len(headings[i - 1][0])
        assert len(crumbs) == heading_lvl

    doc_snippets = re.findall(r"^```", test_doc, re.MULTILINE)
    snippet_list = list(snippets.values())
    assert len(snippet_list) == len(doc_snippets) // 2

    for snip in snippet_list:
        snip_idx = test_doc.index(snip.page_content)
        assert snip_idx >= 0
        front_idx = snip_idx + len(snip.page_content)

        # Check that there is nothing missing infront
        while front_idx < len(test_doc):
            if test_doc[front_idx : front_idx + 3] == "```":
                break
            front_idx += 1
            assert test_doc[front_idx].isspace()

        # Or before the snippet and open/closing sequence
        while snip_idx:
            if test_doc[snip_idx - 3 : snip_idx] == "```":
                break

            snip_idx = snip_idx - 1
            if not test_doc[snip_idx].isspace():
                lang_name = test_doc[snip_idx - 3 : snip_idx + 1].lower()
                assert lang_name == "func" or lang_name == "text"
                snip_idx = snip_idx - 3
