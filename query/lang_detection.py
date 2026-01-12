import logging
import re
import string
from typing import List, Tuple, Optional, Callable
from models.domain import CodeExtractionResult
from query.tolk_stdlib_names import tolk_names
from config import settings

log = logging.getLogger(__name__)


def _filter_doc_path(doc, filter_path: str):
    doc_content = doc[0] if isinstance(doc, tuple) else doc
    return filter_path not in doc_content.metadata["from"]


def _split_by_regex(token: str, regex) -> List[str]:
    sub_tokens = regex.split(token)
    return [tok for tok in sub_tokens if tok]


def extract_code_from_query(query: str) -> CodeExtractionResult:
    # Let's start with simple markdown ``` extraction
    code_fence = "```"
    fence_count = 0
    nl_tokens = []
    code_tokens = []
    code_lines = []
    query_lines = []
    lang_labels = set()
    tokens_set = set()
    same_line: bool
    # I don't like it either
    complex_query: bool = bool(
        re.search(
            "|".join(settings.COMPLEX_QUERY_INDICATORS), query, flags=re.IGNORECASE
        )
    )
    # Not really punctuation if you know what i mean, but gonna treat it as such
    preserve_code_punct = [
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        ";",
        ":",
        ",",
        ".",
        "!",
        "&",
        "|",
        "<",
        ">",
        "+",
        "-",
        "=",
    ]
    code_punct_re = re.compile("([" + re.escape("".join(preserve_code_punct)) + "]+)")
    nl_punct_re = re.compile("([" + re.escape(string.punctuation) + "]+)")

    bracket_closed = True
    for line in query.split("\n"):
        same_line = False
        for tok in line.split():
            stripped_tok = tok.strip()
            if code_fence in stripped_tok:
                fence_count += 1
                bracket_closed = fence_count % 2 == 0
                if not bracket_closed:
                    lang_tok = stripped_tok.lower().split(code_fence)[1:1]
                    if lang_tok:
                        lang_labels.add(lang_tok)
                # Skip the token addition if it is in the same line as ```
                same_line = True
                split_lines = line.split(code_fence)
                if bracket_closed:
                    code_lines.append(split_lines[0])
                else:
                    query_lines.append(split_lines[0])
                continue

            if same_line and (not bracket_closed) and tok:
                lang_labels.add(tok.lower())
            else:
                new_tokens = []
                if bracket_closed:
                    new_tokens = _split_by_regex(tok.lower(), nl_punct_re)
                    nl_tokens.extend(new_tokens)
                else:
                    new_tokens = _split_by_regex(tok, code_punct_re)
                    code_tokens.extend(new_tokens)

                tokens_set.update(new_tokens)

        if not same_line:
            if fence_count % 2 > 0 and (not same_line):
                code_lines.append(line.split(code_fence)[0])
            else:
                query_lines.append(line.split(code_fence)[0])

    boost_query = ""

    if not lang_labels:
        if "tolk" in tokens_set or any(elem in tolk_names for elem in tokens_set):
            log.debug(f"Boosting query {query}")
            boost_query = f"[TOLK language] {query}"
            lang_labels.add("tolk")

    if not complex_query:
        complex_query = len(nl_tokens) > 5 or len(code_tokens) > 0

    return CodeExtractionResult(
        complex_query=complex_query,
        query_tokens=nl_tokens,
        code_tokens=code_tokens,
        code_lines=code_lines,
        query_lines=query_lines,
        boost_query=boost_query,
        languages_detected=list(lang_labels),
    )


def exclude_context_by_lang(
    nl_tokens: List[str], lang_list: Optional[List] = None
) -> Tuple[Optional[Callable], Optional[List[str]]]:
    """
    Ugly context exclusion
    """
    skip_languages: Optional[List] = None
    filter_lambda: Optional[Callable] = None
    has_tolk = False
    has_func = False

    if lang_list is not None:
        # Very short list, so who cares
        has_tolk = "tolk" in lang_list
        has_func = "func" in lang_list

    if not (has_tolk and has_func):
        for tok in nl_tokens:
            if tok == "tolk":
                has_tolk = True
            elif tok == "func":
                has_func = True
            if has_tolk and has_func:
                break

    if has_tolk and not has_func:
        log.debug("Excluding FunC context!")
        filter_lambda = lambda doc: _filter_doc_path(doc, "languages/func")
        skip_languages = ["func", "tact"]
    elif has_func and not has_tolk:
        log.debug("Excluding TOLK context!")
        filter_lambda = lambda doc: _filter_doc_path(doc, "languages/tolk")
        skip_languages = ["tolk", "tact"]
    return filter_lambda, skip_languages
