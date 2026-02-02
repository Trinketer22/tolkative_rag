import logging
import asyncio
from models.domain import Message, IntentInfo
from models.response import ContextResponse
from config import settings
from query.lang_detection import extract_code_from_query, exclude_context_by_lang
from core.rendering import render_docs_batch, render_single_doc
from services.llm import llm_service
from services.vector_store import vector_store
from services.embedding import embedding_service

from services.reranking import reranker
from typing import Optional, Callable, List
from langchain_core.documents import Document

log = logging.getLogger(__name__)


class InputError(Exception):
    pass


def _get_system_context() -> str:
    context = settings.SYSTEM_PROMPT
    return Message(role="system", content=context)


async def retrieve_documents_by_headers(
    query: str,
    query_vector: Optional[List[float]] = None,
    threshold: float = settings.HEADER_THRESHOLD,
    filter_lambda=None,
    top_k=settings.DEFAULT_HEADERS_RETRIEAVAL_SIZE,
) -> List[Document]:
    log.debug(f"Headers query: {query}")
    # Don't pre-filter, because header documents don't have metadata
    doc_batch = await vector_store.search(
        query,
        query_vector,
        top_k=top_k,
        from_headers=True,
        with_score=True,
        filter_fn=None,
    )
    header_docs = [
        doc[0]
        for doc in doc_batch
        # No re-ranking with headers, since
        if doc[1] >= threshold
    ]
    if len(header_docs) > 0:
        # But the actual docs do have meta and can be filtered
        result = vector_store.get_by_ids(list(map(lambda doc: doc.id, header_docs)))
        docs_ranked = await reranker.rerank(query=query, documents=result)
        if filter_lambda is not None:
            docs_ranked = list(filter(filter_lambda, docs_ranked))
        return await top_k_tree(docs_ranked, set(), top_k)
    return list()


async def retrieve_documents(
    query: str,
    query_vector: Optional[List[float]] = None,
    filter_lambda: Optional[Callable] = None,
    top_k=settings.DEFAULT_RETRIEVAL_SIZE,
    rerank_against: str = "",
):
    # We retrieve more to re-rank later
    do_rerank = bool(rerank_against)
    with_score = False
    if do_rerank:
        retrieval_k = top_k * settings.RERANK_COEFF
        with_score = False
    else:
        retrieval_k = top_k
        with_score = True

    initial_docs = await vector_store.search(
        query,
        query_vector,
        top_k=retrieval_k,
        with_score=with_score,
        filter_fn=filter_lambda,
    )

    if len(initial_docs) == 0:
        return initial_docs

    # We migh want to re-rank the result based on itent, instead of a full query
    # rank_query = query if intent is None else intent
    if do_rerank:
        log.debug(f"Ranking query {query}")

        # print(f"Before {initial_docs[0].page_content}")
        # print(f"After{rendered_docs[0].page_content}")
        docs_ranked = await reranker.rerank(
            query=rerank_against, documents=initial_docs
        )
        return await top_k_tree(docs_ranked, set(), top_k)
        """
        Leave it for a better day
        parent_set = set([doc.id for doc in docs_ranked])

        # Suboptimal for now
        docs_ranked.extend(
            await reranker.rerank(
                query=rerank_against,
                documents=await top_k_tree(docs_ranked, parent_set, top_k),
            )
        )
        docs_ranked.sort(
            key=lambda doc: doc.metadata.get("rerank_score", 0.0), reverse=True
        )
        # Hard cap top_k per retrieval, including chinlren and references.
        return docs_ranked
        """

    log.debug("Skip reranking")
    docs_ranked = initial_docs
    return docs_ranked[:top_k]


# Traversing the retrieved knowlage graph, and then picking the top k nodes by rank.
# Other than just the order they were retrieved
async def top_k_tree(docs: List[Document], watch_set: set, top_k: int):
    doc_graph = traverse_knowledge_graph(docs, watch_set)
    doc_graph.sort(key=lambda doc: doc.metadata.get("rerank_score", 0.0), reverse=True)
    return doc_graph[:top_k]


"""
    Tranverses the retried documents knowledge graph.
    If document has children,
    they are added to the output set, but with penalty
    to parent ranking.
    If document ranking score after penalty is lower than
    RERANK_THRESHOLD, it is not added to the output set
    Same goes for the referenced documents

    Output is a flat list.
"""


def traverse_knowledge_graph(
    chunks: List[Document],
    watch_set: set,
    cur_depth=0,
    max_depth=settings.CONTEXT_REF_DEPTH,
    parent_score: float = 0,
):
    uniq_chunks = []
    has_depth = cur_depth + 1 <= max_depth
    for chunk in chunks:
        if chunk.id not in watch_set:
            watch_set.add(chunk.id)
            uniq_chunks.append(chunk)

            chunk_score = parent_score
            if "rerank_score" not in chunk.metadata:
                chunk.metadata["rerank_score"] = parent_score
            else:
                chunk_score = chunk.metadata["rerank_score"]

            child_score = chunk_score * (1 - settings.CHILDREN_PENALTY)
            # print(f"Child score: {child_score}")
            ref_score = chunk_score * (1 - settings.REF_PENALTY)
            # print(f"Ref score {ref_score}")

            children_ids = chunk.metadata["child_nodes"]
            ref_ids = chunk.metadata["references"]

            if (
                len(children_ids) > 0
                and child_score >= settings.RERANK_THRESHOLD
                and has_depth
            ):
                log.debug(f"Fetching child docs from {chunk.id}")
                log.debug(f"Direct children {len(children_ids)}")
                child_docs = await traverse_knowledge_graph(
                    await vector_store.get_by_ids(children_ids),
                    watch_set,
                    cur_depth + 1,
                    parent_score=child_score,
                )
                uniq_chunks.extend(child_docs)
            if (
                len(ref_ids) > 0
                and ref_score >= settings.RERANK_THRESHOLD
                and has_depth
            ):
                log.debug(f"Fetching referenced docs from {chunk.id}")
                log.debug(f"{len(ref_ids)} references found")
                ref_docs = await traverse_knowledge_graph(
                    await vector_store.get_by_ids(ref_ids),
                    watch_set,
                    cur_depth + 1,
                    parent_score=ref_score,
                )
                uniq_chunks.extend(ref_docs)

    return uniq_chunks


def add_uniq_context(chunks: List[Document], watch_set: set):
    uniq_context = []
    for chunk in chunks:
        if chunk.id not in watch_set:
            uniq_context.append(chunk)
            watch_set.add(chunk.id)
    return uniq_context


# Idea here that we have sorted by ranking documents
# gathered from various topics.
# We can't just join those in a flat list by ranking,
# because simple topic would dominate over complex ones.
# Threrefore we return current top 1 document from each topic
# till all topics exchausted.
# This way signal is more-less balanced, if you know what i mean.
def _topic_round_robin(max_context_length: int, *topics: List[Document]):
    # All this jazz is in order to not modify arguments
    topic_count = len(topics)
    topic_lengths = [len(elems) for elems in topics]
    total_count = sum(topic_lengths)
    processed_docs = 0
    total_tokens = 0

    while total_count and total_tokens <= max_context_length:
        for topic_idx in range(topic_count):
            # Yeild top dog, else check next topic
            cur_topic_left = topic_lengths[topic_idx]
            if cur_topic_left > 0:
                top_dog = topics[topic_idx][-cur_topic_left]
                topic_lengths[topic_idx] = cur_topic_left - 1
                doc_token_count = top_dog.metadata.get("token_count", 0)
                total_tokens = doc_token_count + settings.CTX_RENDERING_OVERHEAD
                if total_tokens > max_context_length:
                    if processed_docs == 0:
                        raise InputError(
                            f"Token limit is not enough to fit a single result.\nResult_size: {doc_token_count}"
                        )
                    break
                total_count -= 1
                processed_docs += 1
                yield top_dog


async def retrieve_topic(
    topic: str,
    query: str,
    topic_vector: Optional[List[float]] = None,
    filter_context: Optional[Callable] = None,
    top_k: int = settings.DEFAULT_RETRIEVAL_SIZE,
):
    rerank_key: str

    if "example" in topic.lower():
        rerank_key = topic
    else:
        rerank_key = f"Explanation of {topic}"

    docs = await retrieve_documents(
        topic,
        topic_vector,
        filter_lambda=filter_context,
        rerank_against=rerank_key,
        top_k=top_k,
    )
    if len(docs) > 0:
        top_doc_score = docs[0].metadata.get("rerank_score", 0)
        if top_doc_score >= settings.TOP_QUALITY_THRESHOLD:
            log.debug(
                f"Skipping topic {topic} in query {query}. Excellent result found!"
            )
            return docs[: settings.TOP_QUALITY_RETRIEVAL_SIZE]
    return docs


async def pull_context(
    orig_msgs: List[Message], max_tokens: Optional[int] = None
) -> ContextResponse:
    # Basic input checks
    # Could have searched over all messages, but let's not compilcate things here.
    if len(orig_msgs) == 0 or orig_msgs[-1].role != "user":
        raise InputError("No user message found")

    if max_tokens is None:
        max_tokens = settings.CTX_TOKEN_LIMIT

    if max_tokens < settings.CTX_MINIMAL_OUTPUT:
        raise InputError(f"Minimal output tokens length {settings.CTX_MINIMAL_OUTPUT}")
    user_msg = orig_msgs[-1].content
    context = []
    has_system = any(msg.role == "system" for msg in orig_msgs)
    system_msg = None
    intent_topics: IntentInfo = IntentInfo(intent=user_msg, concepts=[])
    # If there is no system message, put it before the user message
    if not has_system:
        log.debug("Adding system message...")
        system_msg = _get_system_context()

    log.debug(f"Received user message: {user_msg}")
    code_detect = extract_code_from_query(user_msg)
    if code_detect.boost_query:
        user_msg = code_detect.boost_query
    filter_context, skip_languages = exclude_context_by_lang(
        code_detect.query_tokens, code_detect.languages_detected
    )
    quick_path = False

    text_ctx = await retrieve_documents(
        user_msg, filter_lambda=filter_context, rerank_against=user_msg
    )
    log.debug(f"Initial text search brought {len(text_ctx)} results")

    if not code_detect.complex_query:
        res_sum = 0.0
        top_n = settings.SKIP_LLM_EXTRACTION[0]
        if len(text_ctx) >= top_n:
            for idx in range(top_n):
                res_sum += text_ctx[idx].metadata["rerank_score"]

        if res_sum / top_n >= settings.SKIP_LLM_EXTRACTION[1]:
            # Allowed to skip llm extraction heavy path
            context = text_ctx
            quick_path = True
            log.debug("Skipping llm extraction")

    if not quick_path:
        # If we have already received too many, no point for LLM extraction
        tokens_sum = sum([doc.metadata.get("token_count", 0) for doc in text_ctx])
        quick_path = tokens_sum >= max_tokens

    if not quick_path:
        intent_topics = await llm_service.extract_intent(
            user_msg
        )  # IntentInfo(intent=user_msg, concepts=[])

        log.debug(f"Topics from intent extraction: {intent_topics}")

        queries_to_embed = [
            intent_topics.intent,  # For language detection + header search
            *intent_topics.concepts,  # For concept searches
        ]

        embeddings = await embedding_service.embed_documents(queries_to_embed)

        # Unpack embeddings
        intent_emb = embeddings[0]
        concept_embs = zip(intent_topics.concepts, embeddings[1:])

        # Exclude the others if it belongs to domain with exclusive concepts
        # Otherwise LLM will produce the code with mix of languages

        # We can further parellalize all the retrieval, but that's a problem for another day
        headers_task = retrieve_documents_by_headers(
            intent_topics.intent, intent_emb, filter_lambda=filter_context
        )
        # Less so for full text search
        # Searchin with intent -> broarder and less noise
        # Re-ranking by full query for better fitting
        text_task = retrieve_documents(
            user_msg,
            intent_emb,
            filter_lambda=filter_context,
            rerank_against=intent_topics.intent,
        )

        concept_tasks = [
            retrieve_topic(
                topic, intent_topics.intent, topic_emb, filter_context=filter_context
            )
            for topic, topic_emb in concept_embs
        ]
        # unified_query = f"{intent_topics.intent} [SEP] Keywords: {",".join(intent_topics.concepts)}"
        # concept_task = retrieve_documents(unified_query, filter_lambda = filter_context, rerank_against = unified_query)

        # I know this is Suboptimal, we can merge headers and texts, and only then re-rank once,
        # but for debugging purposes it will stay for now
        id_set = set([doc.id for doc in text_ctx])
        context_batch = await asyncio.gather(headers_task, text_task, *concept_tasks)
        log.debug(f"Initial header results {len(context_batch[0])}")
        headers_ctx = add_uniq_context(context_batch[0], id_set)
        log.debug(f"From headers added {len(headers_ctx)}")

        text_ctx.extend(add_uniq_context(context_batch[1], id_set))

        log.debug(f"Intent text results {len(context_batch[1])}")
        log.debug(f"From texts added total {len(text_ctx)}")
        # concept_ctx = add_uniq_context(context_batch[2], id_set)
        # print(concept_ctx)
        text_ctx.sort(
            key=lambda doc: doc.metadata.get("rerank_score", 0.0), reverse=True
        )

        topic_ctx = context_batch[2:]

        # context = headers_ctx + text_ctx
        extraction_ctx = [headers_ctx, text_ctx]

        for concept, ctx in zip(intent_topics.concepts, topic_ctx):
            uniq_docs = add_uniq_context(ctx, id_set)
            log.debug(f"From topic {concept} added {len(uniq_docs)}")
            # Appending separate topic in list form
            extraction_ctx.append(uniq_docs)

        # context.sort(key=lambda doc: doc.metadata.get("rerank_score", 0.0), reverse=True)
        # Round robin extraction from the contexts
        context = [doc for doc in _topic_round_robin(max_tokens, *extraction_ctx)]

    (rendered_docs, token_count) = await render_docs_batch(
        context, token_limit=max_tokens, language_filter=skip_languages
    )
    rendered_ctx = "\n".join(rendered_docs).strip()
    raw_context = None

    if settings.ADD_RAW_CONTEXT:
        raw_context = list(map(lambda doc: doc.model_dump(), context))

    return ContextResponse(
        context=Message(role="user", content=f"{rendered_ctx}\n\n{user_msg}"),
        ctx_token_count=token_count,
        raw_context=raw_context,
        system=system_msg,
        intent=intent_topics,
    )
