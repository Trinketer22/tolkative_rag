# Tolkative RAG server

This project is an attempt to provide the context layer for LLMs over existing TON documentation.

## What it does

### Retrieval augmented generation

According to [wiki](https://en.wikipedia.org/wiki/Retrieval-augmented_generation),

Retrieval-augmented generation is a technique that enables large language models to retrieve and incorporate new information from external data sources.
With RAG, LLMs do not respond to user queries until they refer to a specified set of documents.

These documents supplement information from the LLM's pre-existing training data.
This allows LLMs to use domain-specific and/or updated information that is not available in the training data.

That is precisely what this project is about.

## What it doesn't do

### It is not a model

It uses existing models to provide response.
It's sole purpose is to inject additional context between the consumer and the model.

### It is not an AI Agent or assistant

Agent acts on the data, RAG provides the data.

However, an agent example that uses RAG capabilities is [available](/client).

Recent client implementation details:

- Built with LangGraph (`client/agent.py`, `client/cli_client.py`)
- Uses RAG tool against `/context` with `ContextRequest(query=...)`
- Uses in-memory conversation checkpointing (`MemorySaver`) in CLI mode
- Supports custom model endpoint and prompts via env vars

Simplified snippet from the LangGraph demo client (`client/agent.py`):

```python
class RagInput(BaseModel):
    query: str = Field(description="The input query")

async def call_rag(query: str):
    async with httpx.AsyncClient(timeout=rag_query_timeout) as client:
        response = await client.post(
            rag_url,
            json=ContextRequest(query=query).model_dump(exclude_unset=True),
        )
    return ContextResponse(**response.json()).context

tool_rag = StructuredTool.from_function(
    func=None,
    coroutine=call_rag,
    name="doc_query_tool",
    description="This tool allows to query TON documentation for additional info",
    args_schema=RagInput,
)
tools = [tool_rag]
```

CLI client defaults:

- `CTX_URL`: `http://localhost:8000/context`
- `MODEL_NAME`: `claude-sonnet-4.6`
- `OPENAI_API_BASE`: required in current CLI client setup

For details and local run instructions check [client](/client) and `client/README.md`.

On top of that, there is OpenAPI definition available at `http://<rag_server_url>/openapi.json`

### RAG API vs MCP wrapper

This FastAPI app is the RAG backend API. It is stateless and focuses on retrieval + context assembly.

Separately, this repo now includes a minimal MCP wrapper example that exposes the RAG backend as an MCP tool.

In other words:

- RAG server (`main.py`) exposes HTTP endpoints such as `/context` and `/v1/chat/completions`
- MCP server example (`scripts/mcp_server.py`) exposes MCP tool `get_ton_context(...)` and calls the RAG HTTP API

So this project now contains both pieces: the core RAG API and an optional MCP adapter process.

#### MCP wrapper example

MCP example details:

- `scripts/mcp_server.py` (FastMCP server)
- exposes tool `get_ton_context(query, max_tokens=None)`
- forwards calls to `/context` and returns `{ ok, context, ctx_token_count }`

Useful env vars:

- `CTX_URL` (default: `http://localhost:8000/context`)
- `CTX_TIMEOUT` (default: `30` seconds)
- `MCP_SERVER_NAME` (default: `tolkative-rag-mcp`)

Install the MCP dependencies:

```shell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu -e ".[mcp]"
```

Then add the mcp definition to your agent configuration.

For example for opencode:
```json

{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "...": {}
  },
  "mcp": {
    "tolkative-rag": {
      "type": "local",
      "enabled": true,
      "command": [
        "/home/user/tolkative-rag/scripts/mcp_server.py"
      ],
      "environment": {
        "CTX_URL": "http://localhost:8000/context"
      },
      "timeout": 15000
    }
  }
}
```

For Codex CLI (`~/.codex/config.toml`):

```toml
[mcp_servers.tolkative-rag]
command = "python"
args = ["/home/user/tolkative-rag/scripts/mcp_server.py"]

[mcp_servers.tolkative-rag.env]
CTX_URL = "http://localhost:8000/context"
CTX_TIMEOUT = "30"
MCP_SERVER_NAME = "tolkative-rag-mcp"
```

For Claude Code (`.claude/settings.json` in project root, or `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "tolkative-rag": {
      "command": "python",
      "args": [
        "/home/user/tolkative-rag/scripts/mcp_server.py"
      ],
      "env": {
        "CTX_URL": "http://localhost:8000/context",
        "CTX_TIMEOUT": "30",
        "MCP_SERVER_NAME": "tolkative-rag-mcp"
      }
    }
  }
}
```

If your agent uses a different MCP config schema/version, adapt field names accordingly (`mcp`, `mcpServers`, `mcp_servers`).


### It does not just send all the documents for every query

Modern LLMs claim to have massive context windows, and
sending excessive amounts of context may accomplish some
goals, but this approach doesn't scale.

RAG aims to provide the just right context.

## Data sources

Current list of data sources is rather humble:

- Official [TON documentation](https://github.com/ton-org/doc)
- [TEPs](https://github.com/ton-blockchain/TEPs)
- [TOLK bench](https://github.com/ton-blockchain/tolk-bench) as a source of contract examples
- [TVM Specification](https://github.com/ton-blockchain/tvm-specification/blob/master/gen/tvm-specification.json).

## Quick start

Processed documents attached as release artifacts.
Download and extract to rag-data

### Local env installation

In general it is enough to set up a venv and run:

```shell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu -e .

```

If you also plan to run the bundled MCP example, install with MCP extras:

```shell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu -e ".[mcp]"
```

#### Pull submodules

Local development and tests require git submodules (documentation snapshots, examples, and TEPs data).

```shell
git submodule update --init
```

#### Dev

For development, install additional dependencies required for unit tests:

```shell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu -e ".[test]"
```

#### Prepare data dump

Build `rag-data/docs.jsonl` and `rag-data/latest_snippets.jsonl` before index setup:

```shell
python scripts/prepare_data.py
```

Useful flags:

```shell
python scripts/prepare_data.py --skip-instructions
python scripts/prepare_data.py --output <custom-rag-data-path>
```

The data preparation step also consumes:

- `rag-data/examples_mapping.json` for source-code example mappings
- `rag-data/instructions_desc.json` for optional TVM instruction docs

#### Build release artifacts on GitHub

Workflow `.github/workflows/build-rag-data.yml` supports both manual and tag-driven release flows.

- Manual run (`workflow_dispatch`): runs tests, builds dumps, uploads workflow artifacts.
- Tag run (`push` on `v*`): runs tests first; only if tests pass it builds and publishes release assets.

The workflow:

- checks out repo with submodules
- runs `pytest`
- builds `rag-data/docs.jsonl` and `rag-data/latest_snippets.jsonl`
- uploads the dumps as workflow artifact (`rag-data-dumps`)
- on tag runs, attaches both files to the GitHub release

#### Setup rag index

Then run `rag-setup <path to docs.jsonl>` to build the index.

Example:

```shell
rag-setup rag-data/docs.jsonl
```

### Docker build

[Dockerfile](/docker/Dockerfile) is available.

#### Build the docker

``` shell
docker build -f docker/Dockerfile . -t rag_img
```

#### Run the image

``` shell
docker run --name rag_server -v <Path to your .env file>:/app/.env:ro rag_img
```

This image runs `rag-setup rag-data/docs.jsonl` at build time, so make sure `rag-data/docs.jsonl` exists before `docker build`.

## How to add/modify documents

Currently there is no convenient interface for that.

However, admin REST APIs are now available.
Swagger interface can be checked at `http://<rag_server_url>/docs`.

Note that admin API is protected with Bearer authorization.

Check the following configuration parameters:

```python
# Admin configuration
ENABLE_ADMIN: bool = True
ADMIN_PREFIX: str = "/admin"
ADMIN_AUTH_TOKEN: str = "secret"
ADMIN_UNAUTHORIZED_BANNER: str = "Access denied"
```

Documents are stored as newline separated json objects `jsonl` in `rag-data` directory.
Which means that one could easily add/remove documents by appending/removing lines
with documents content, then just perform index rebuild using `rag-setup` script.

Rebuild will only take a couple of minutes

For a deep dive into documents structure, check [PrepareData](/notebooks/PrepareData.ipynb) jupyter notebook.

## Endpoints

Endpoints top level handlers are implemented at [routes](/api/routes).

The service exposes OpenAI-compatible chat proxy endpoints and dedicated context-retrieval endpoints.

### Context

There are two context endpoints:

- `/chat_context` takes an OpenAI ChatCompletion request and returns the full message-context response.
- `/context` takes a lightweight request payload:

```json
{"query": "...", "max_tokens": 20000}
```

Current `/context` response model:

```python
class ContextResponse(BaseModel):
    context: str
    ctx_token_count: int
```

Current `/chat_context` response model:

```python
class MessageContextResponse(BaseModel):
    prompt_msg: Message
    context: str
    ctx_token_count: int
    raw_context: Optional[List[Dict]] = None
    system: Optional[Message] = None
    intent: Optional[IntentInfo] = None
```

### Chat completion

OpenAI compatible chat completion endpoint.

Mapped to `/v1/chat/completions`

It serves an chat completion proxy with context augmentation on the fly.
So, it can be used plug'n play with any ChatCompletion compatible tool.

All the optional fields are passed as is to the LLM endpoint.

[Streaming](https://platform.openai.com/docs/api-reference/responses-streaming) **is supported**.

### Models

Some tools require the model name, to interact with ChatCompletion endpoint.
Model endpoint is also compatible with the [OpenAI Models](https://platform.openai.com/docs/api-reference/models/list).

Currently just outputs the model name from config.

### Health

Health endpoint is mapped to `/health`.

## How it works

### Configuration

Configuration is stored in [config.py](/config.py)
Most of the parameters a commented, or self-explainatory.

`config.py` contains the default values.
User provided values are picked from `.env` and merged with the defaults.

Most important parameters will be mentioned in the following chapters.

### Documents

`Document` in terms of this project **is not** a file.

File is divided into documents based on semantic boundary.
In current implementation, files are chunked into documents by markdown headers.
This may change in the future.
Header level is determining the document hierarchy within a file.

### Components

Components and project structure description.

#### Vector storage

In order to quickly find the relevant documents,
[FAISS](https://docs.langchain.com/oss/python/integrations/vectorstores/faiss) storage is used.

Implemented in [/services/vector_store.py](/services/vector_store.py) it allows to fetch documents
that contain words semantically close to the query.

Currently, there are two FAISS indexes used.

1. Main index, that contains full documents text and metadata
2. Documents crumbs(breadcrumbs) index for a quick noise-less lookup, that only contain documents titles joined into hierarchy `Languages>>Tolk>>Syntax>>etc`.

Following configuration parameters point to the index locations

``` python
    INDEX_PATH: str = str(CONFIG_PATH / Path("indexes/full_separate_snip"))
    HEADERS_INDEX_PATH: str = str(CONFIG_PATH / Path("indexes/top_level_index"))
```

#### Embedding

In vector storages, data is stored in vectorized (embedded) form.
Converting from natural to vectorized form is an expensive operation.
Therefore the [/services/embedding.py](/services/embedding.py)
implements the caching layer between the retrieval pipeline and the
vector storage.

#### Snippet caching

Note, that documents are stored separately from the code snippets.

Reason behind is that encoder (embedder) and cross-encoder models
are trained on natural language.
Code is mostly noise for it, unless it is a code specialized model like [CodeBERT](https://github.com/microsoft/CodeBERT).

multi line snippets are stripped from the documents, but their position and identifier is stored in the document metadata.

Snippets without line terminators are kept inlined in the document, because the often contain more natural language than code.

At the context rendering stage(last prior to sending the context out), snippet text is pulled from the cache and inlined into the document at it's corresponding position.

Markdown code tags are used.

Snippet cache is implemented in  [/services/snippet_cache.py](/services/snippet_cache.py).

#### Reranker

In order to distinguish between documents that just matched by semantically close
keywords, from ones, that actually answer the user query, a [Cross-Encoder](https://www.sbert.net/examples/cross_encoder/applications/README.html) model is used.

The most CPU intensive operation in the pipeline
Implemented in [/services/reranking.py](/services/reranking.py).

Following configuration parameters a crucial for reranker behavior:

``` python
    # How many documents to retrieve by default from MAIN index
    DEFAULT_RETRIEVAL_SIZE: int = 10
    # Scores for the ms-marco-MiniLM-L6-v2 model.
    # It produces scores from -10 to 10
    # Scores below that value is dropped
    RERANK_THRESHOLD: float = 1

    # Re-rank coefficient
    # How much more retrieve at pre-rerank stage
    # To get top_k
    RERANK_COEFF: int = 2  # Means retrieve TOP_K * COEFF
```

If documents are queried without re-ranking, `DEFAULT_RETRIEVAL_SIZE` elements is returned.

If re-ranking is performed(currently always), `DEFAULT_RETRIEVAL_SIZE * RERANK_COEFF` is retrieved, than ranked.

Results sorted by `rerank_score`, and the elements below `RERANK_THRESHOLD` are dropped completely.

Only top  `DEFAULT_RETRIEVAL_SIZE`  documents is returned per query.

### Context retrieval

Brief description of context retrieval pipeline implemented in `core/retrieval.py` from logical perspective.

#### High level diagram

``` mermaid
graph TD
    Input([User Query + Messages]) --> Validation{Input validation}
    Validation -- Invalid --> Error[400 invalid_input]
    Validation -- Valid --> SystemCheck{System message present?}
    SystemCheck -->|No| AddSystem[Attach default system prompt]
    SystemCheck -->|Yes| LangDetect
    AddSystem --> LangDetect[Code/language detection + context filters]

    LangDetect --> InitialRetrieval[Initial retrieve_documents(user_msg)\n(main index + rerank)]
    InitialRetrieval --> InitTree[Knowledge tree traversal\nadd child_nodes + references\napply CHILDREN_PENALTY / REF_PENALTY\nrespect CONTEXT_REF_DEPTH]
    InitTree --> QuickCheck{QUICKPATH_ONLY\nor simple-query score pass?}

    QuickCheck -- Yes --> UseInitial[Use initial text context]
    QuickCheck -- No --> TokenGate{Initial tokens >= max_tokens?}
    TokenGate -- Yes --> UseInitial
    TokenGate -- No --> IntentExtract[LLM intent + concepts extraction]

    IntentExtract --> Embed[Embed intent and concept queries]
    Embed --> ParallelFetch[Parallel retrieval:\nheaders(intent), text(intent), each concept\n(each path reranked)]
    ParallelFetch --> DeepTree[Knowledge tree traversal per bucket\nadd child_nodes + references with penalties]
    DeepTree --> MergeBuckets[Deduplicate and build buckets\nheaders + text + per-topic]
    MergeBuckets --> RoundRobin[Round-robin selection under token budget]

    UseInitial --> Render[render_docs_batch\n(inline snippets + language filter)]
    RoundRobin --> Render
    Render --> Response[MessageContextResponse:\ncontext, prompt_msg, system, intent, raw_context?]

    style ParallelFetch fill:#bbf,stroke:#333
    style InitTree fill:#cfe,stroke:#333
    style DeepTree fill:#cfe,stroke:#333
    style IntentExtract fill:#f9f,stroke:#333
    style QuickCheck stroke-dasharray: 5 5
```


#### Input validation

In order for the message to be processed it must:

- Comply with [ChatCompletion definition](https://platform.openai.com/docs/api-reference/chat)
- In the message with `user` role should be last in `messages` field.
- `max_tokens` parameter should be at least `setting.CTX_MINIMAL_OUTPUT`

In case of invalid input, status code 400 is returned with error in format:

``` json

{"error": {"code": "invalid_input", "msg": "Error description"}

```

#### Query processing

First, the query text is processed in a following way:

- Code part (if present) is separated from the natural language part of the query.
- Code name tag is extracted(if present)
- If no code tag present, tokens are checked against the know language tokens (currently only tolk)
- If such token found, query is labeled as containing code, and language name is attached
- All query tokens are matched against list of keywords, that supposed to indicate that query is likely complex.

Detected language markers are also used to exclude unrelated language contexts during retrieval (for example, filtering FunC docs for Tolk-focused queries and vice versa).

Query is labled complex if:

- It contains code.
- Popular language tokens.
- One or more keywords from `COMPLEX_QUERY_INDICATORS`

Full list of complex query indicators can be found in configuration.

#### Initial retrieval

Initially the data is retrieved and re-ranked based on full query.

##### For simple query

In case query is considered simple, top n documents average score is calculated.
If average score is above threshold, data goes straight to rendering

This behavior is regulated by following configuration parameter:

```python
# If top 3 results score >= 4.25 on average
# And the query is not considered complex.
# We can skip the LLM extraction
SKIP_LLM_EXTRACTION: Tuple[int, float] = (3, 4.25)

```

There is also an explicit quick-path switch:

```python
# Force retrieval-only flow for all requests
QUICKPATH_ONLY: bool = False
```

When `QUICKPATH_ONLY` is enabled, the pipeline skips LLM intent/concept extraction and uses retrieval + reranking only.

Additionally, topic retrieval has a top-quality shortcut to avoid over-expanding already strong matches:

```python
TOP_QUALITY_RETRIEVAL_SIZE: int = 1
TOP_QUALITY_THRESHOLD: float = 6.5
```

If a topic's top rerank score exceeds `TOP_QUALITY_THRESHOLD`, only the top `TOP_QUALITY_RETRIEVAL_SIZE` document is kept for that topic.

#### LLM powered intent and concepts extraction

For complex queries, or ones that didn't pass the quality threshold,
LLM intent and concepts extraction is performed.

Extraction prompt, model and temperature is regulated by:

``` python
    INTENT_EXTRACTION_TEMP: float = 0
    INTENT_EXTRACTION_PROMPT: str
    CHEAP_MODEL: str
```

Intent extraction works as a noise filter and concept retrieval extracts broad topics relevant to the query.

Example:

``` json
Query:  "How to implement sharded contract?"

Intent extraction result:
'intent': {'intent': 'Implement sharded contract',
  'concepts': ['TON Sharded contract implementation (contract sharding, partitioned contract, state sharding)',
   'TON Sharded contract example (contract sharding sample, shard implementation example, partitioned contract demo)',
   'TON Sharded contract state distribution (state sync, state replication, state migration)',
   'TON Sharded contract shard coordination (cross-shard communication, inter-shard calls, shard messaging)',
   'TON Sharded contract data partition strategy (hash partitioning, range partitioning, deterministic splitting)',
   'TON Sharded contract lifecycle management (shard activation, shard rebalancing, shard deactivation)',
   'TON Sharded contract verification (shard proof, state validation, consistency checks)']}}
```

Next, retrieval by intent is performed over:

- headers index
- full text index again
- for each of the concepts.

Each concept specific topic is ranked against explanation of the concept,
where other data retrieval is ranked against the intent.

#### Knowledge tree traversal

Documents contain two types of child nodes.

##### Direct children

Direct children are documents that are lower in hierarchy.

Example: `Tolk language syntax` parent doc, and `Conditional loops ` is child node.

All document below in header hierarchy are added to the parents metadata

##### References

References are actual hyperlinks.
If those links lead within the documentation structure, description document is added to references in source document metadata

#### Child nodes ranking

Child nodes are ranked by discounting it's parents rank.

Discount coefficient is different for each type and determined by the following parameters:

``` python
    CHILDREN_PENALTY: float = (
        0.25  # child documents will suffer 25% penalty from parent ranking
    )
    REF_PENALTY: float = (
        0.5  # reference documents will suffer 50% penalty from parent ranking
    )
    CONTEXT_REF_DEPTH: int = 1  # Depth of parent->rel/child chain
```

Calculation is performed like this:

``` python
            child_score = chunk_score * (1 - settings.CHILDREN_PENALTY)
            ref_score = chunk_score * (1 - settings.REF_PENALTY)
```

If resulting score is below the `RERANK_THRESHOLD`, children nodes of the failing
type are not added to output.

Finally, the traversed knowledge tree is flattened out, de-duplicated and sorted by ranking,
and top `DEFAULT_RETRIEVAL_SIZE` best performing documents are returned for each retrieval.

#### Final retrieval and rendering

Search results end up in their own `bucket`(list),
and then passed to rendering in a round-robin fashion.

Meaning:

- Remove top 1 element of the first bucket
- Proceed to the second, remove the top 1 element
- Proceed to the third, etc.

This happens till either token budget or all of the `buckets` are exhausted.

Rationale behind this process is that some of materials are ranked against the raw
query, some against the intent, others against the concept topic.
Round robin rendering allows to maintain signal balance over the results.

Token budget is determined either from user input, or from:

``` python
# 100K default total context tokens limit, unless max_tokens is specified in the request
CTX_TOKEN_LIMIT: int = 100000

```

If user omitted `max_tokens`.

Documents are rendered into the text representation and sent to the
client.

**NOTE** that token budget should include the rendering overhead.

##### Rendering configuration

``` python
    # Rough overhead estimate per element.
    CTX_RENDERING_OVERHEAD: int = 200
    # If set to true, documents metadata is added to ContextResponse raw_context field
    ADD_RAW_CONTEXT: bool = False
    # Wether or not to include rerank_score in rendered documents
    EXPOSE_SCORING: bool = True
```

## TODO

### Proper unit testing

Code base grew substantially, it's an obvious must.

### Evaluation tests

Need tests that evaluate the produced results in a quantified ways.

[RAGAS](https://docs.ragas.io/en/latest/) and alike.

Researched the theme, but didn't quite implemented it.

### Performance and latency optimization

There is room to batch some of the search results re-ranking together,
or start LLM extraction in parallel with the initial retrieval for complex query.

### Migrate to chromadb

FAISS was a good choice for rapid development stage, but from managing standpoint chorma is way easier to deal with
In terms of thread safety and document edits/updates.

There will be performance penalty, but as long as there is LLM extraction is used,
and the inference itself takes quite a bit of time, it is tolerable.

### AST based knowledge base

In the `ParseSnippets` notebook most of the TOLK ast node types are already linked
to the documentation topics.

It makes sense to extend it further and make use of AST parser where applicable,
instead of the LLM code structures extraction.

### Build complete knowledge graph over documentation

Build hardwired `subject->predicate->object map` over all of the documentation.

It will allow to ditch LLM extraction for most of the queries and improve performance drastically


### Customize encoder model

Adapting CodeBERT to TON ecosystem might be worth a shot.
It would allow semantic search over code, instead of just natural language description
