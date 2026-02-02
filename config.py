"""
Application settings using Pydantic.
Loads from environment variables.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Tuple, Literal


class Settings(BaseSettings):
    """Application settings."""

    # App
    APP_NAME: str = "Tolkative RAG API"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    CHEAP_MODEL: str = (
        "GPT-5.1-Codex-Mini"  # "gpt-4o-mini"#"GPT-5.1-Codex-Mini"# "gpt-4o-mini"
    )
    TOP_MODEL: str = "claude-sonnet-4.5"

    # Paths
    CONFIG_PATH: Path = Path(__file__).resolve().parent
    INDEX_PATH: str = str(CONFIG_PATH / Path("indexes/full_separate_snip"))
    HEADERS_INDEX_PATH: str = str(CONFIG_PATH / Path("indexes/top_level_index"))
    SNIPPETS_PATH: str = str(CONFIG_PATH / Path("rag-data/latest_snippets.jsonl"))
    MODEL_CACHE_DIR: str = str(CONFIG_PATH / ".models")

    # Re-rank coefficient
    # How much more retrieve at pre-rerank stage
    # To get top_k
    RERANK_COEFF: int = 2  # Means retrieve TOP_K * COEFF
    # Similarity or max marginal relevance search types
    SEARCH_TYPE: Literal["SIMILARITY", "MMR"] = "SIMILARITY"

    # How many documents to retrieve by default from MAIN index
    DEFAULT_RETRIEVAL_SIZE: int = 10
    # How many documents to retrieve by default from CRUMBS index
    DEFAULT_HEADERS_RETRIEAVAL_SIZE: int = 5

    """
    If document scores that high,
    reranker model is certain that it's
    a quality doc, that contains the answer.
    No need to pull this topic further and pollute
    context with sub-par docs.
    Mostly happens for contract example requests
    """
    TOP_QUALITY_RETRIEVAL_SIZE: int = 1
    TOP_QUALITY_THRESHOLD: float = 6.5

    # Rough overhead estimate.
    CTX_RENDERING_OVERHEAD: int = 200
    # 100K default total context tokens limit, unless max_tokens is specified in the request
    CTX_TOKEN_LIMIT: int = 100000

    CTX_MINIMAL_OUTPUT: int = CTX_RENDERING_OVERHEAD * 2

    # Scores for the ms-marco-MiniLM-L6-v2 model.
    # It produces scores from -10 to 10
    # Scores below that value is dropped
    RERANK_THRESHOLD: float = 1
    HEADER_THRESHOLD: float = 0
    CHILDREN_PENALTY: float = (
        0.25  # child documents will suffer 25% penalty from parent ranking
    )
    REF_PENALTY: float = (
        0.5  # reference documents will suffer 50% penalty from parent ranking
    )
    CONTEXT_REF_DEPTH: int = 1  # Depth of parent->rel/child chain

    # If top 3 results score >= 4.25 on average
    # And the query is not considered complex.
    # We can skip the LLM extraction
    SKIP_LLM_EXTRACTION: Tuple[int, float] = (3, 4.25)

    # Logging
    LOG_LEVEL: str = "INFO"  # "DEBUG"
    LOG_FILE: str = str(Path("logs/rag.log"))
    OPENAI_API_KEY: str = "secret"

    PROXY_MODEL_NAME: str = "TOLKative model"
    PROXY_MODEL_OWNED_BY: str = "TON Core"
    PROXY_MODEL_CONTEXT_NOT_FOUND_MESSAGE: str = """
    No context found for your request
    """
    # If set to true, documents metadata is added to ContextResponse raw_context field
    ADD_RAW_CONTEXT: bool = False
    # Wether or not to include rerank_score in rendered documents
    EXPOSE_SCORING: bool = True

    # Wordlist for complex query detection.
    COMPLEX_QUERY_INDICATORS: List[str] = [
        "implement",
        "write",
        "code",
        "build",
        "create",
        "provide",
        "generate",
        "explain",
        "develop",
        "setup",
        "configure",
        "fix",
        "issue",
        "debug",
        "troubleshoot",
        "resolve",
        "solve",
        "doesn't",
        "getting",
        "handle",
        "optimize",
        "faster",
        "reduce",
        "efficient",
        "best",
        "improve",
        "refactor",
        "how",
        "steps",
        "procedure",
        "workflow",
        "protocol",
        "strategy",
        "pattern",
        "interact",
        "communicate",
        "bridge",
        "send",
        "receive",
        "call",
        "invoke",
        "connect",
        "integrate",
        "combine",
        "together",
        "behavior",
        "logic",
        "works",
        "mechanism",
        "internals",
        "architecture",
        "difference",
        "compare",
        "better",
        "alternative",
        "why",
        "rationale",
        "purpose",
        "case",
        "when",
        "loop",
        "iterate",
        "conditional",
        "branch",
        "recursive",
        "callback",
        "listener",
        "parse",
        "serialize",
        "deserialize",
        "encode",
        "decode",
        "map",
        "transform",
        "wrap",
        "validate",
        "verify",
        "assert",
        "check",
        "ensure",
        "protect",
    ]
    COMPLEX_QUERY_LENGTH: int = 10

    # LLM Service config
    LLM_AUTH_TYPE: str = "Bearer"
    OPENAI_API_BASE: str = "https://api.ppq.ai/v1"
    INTENT_EXTRACTION_TEMP: float = 0
    INTENT_EXTRACTION_PROMPT: str = """
You are a helpful data retrieval assistant
Goal is to help the engineer to find all information relative to the user query.
Identify the user request from query.
Populate the **FULL** list of search terms (concepts) (for loop, while loop, conditional operations, etc)  helpfull for the engineer
For every programing language specific concept:
- If requested to generate or analyze code, and language name is explicity identified in the query, include language syntax as a separate concept. Otherwise skip this point.
- For every concept, **EXCEPT the language syntax** provide 2-3 most common technical synonyms or function names likely to be used Example:(Time, timestamp, now, etc) as a dedicated concept. **ALWAYS** prepend language name in the concept name <LANGUAGE NAME> <Concept> (<synonym a>, <synonym b>,..)
- If source code is provided in the query or code generation is requested, identify the code control structures (do-while loop, structures, conditional operators, etc) and add as separate concepts.
- If some specific function name mentioned in the query, add it literally as separate context.
- For every language specific concept, include the explicit language name at the begining of the concept name if indicated.

If user is requesting to implement certain functionality, add example to the concepts list. Be specific. (Jetton contract imlementation example, Jetton wrappers example, Jetton unit tests example, etc)
If concept is implied, but not named explicity, add it to the list.

Respond in a following structure {"intent": "<user_intent>", "concepts": [<json list of concepts>]}

DONT include into the concepts:
- Anything related to blockchains other than TON like Ethereum, Solana, Solidity, etc


Examples:
Input: "Create a Jetton contract with custom transfer logic using tolk language. Provide unit tests and wrappers"
Output: {"intent": "Create custom Jetton contract using tolk with unit tests", "concepts": ["Jetton contract example", "TOLK language", "Jetton wrappers example", "Jetton unit tests example"]}

Provide response in the raw json format.

"""
    SYSTEM_PROMPT: str = """
You are an experienced TON developer.
Context is wrapped in <context></context> tags.
First cite the used context "id","doc-url","concept" for every context entry.
Keep close attention to the context provided and don't mix the context from different programming
languages when generating code.

Provide comprehensive answer to the user request using the context supplied.
If no context provided, explicitly indicate that fact and do not respond anything else.

When code examples provided in the context satisfy user request:
- Return code snippets EXACTLY as they appear in the context
- Don't merge snippets comming from different files into a single one, unless explicitly told so.
- Do NOT modify, improve, or fix the code without explicit request.
- If you must reference code, use EXACT copy-paste
    """

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
