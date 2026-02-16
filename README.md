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

However, agent example that uses RAG capabilities is [available](/client).

On top of that, there is openapi definition available at `http://<rag_server_url>/openapi.json`

### It is not an MCP Server

Even though, current implementation
has ability to proxy user request with augmented context on the fly,
It doesn't track the conversation scope , but only
augments the last user message with the relevant context.
It's a **STATELESS** context providing machine.

MCP server can later perform any caching of the context provided.

Bottom line, it serves the `C` in the `MCP`.

RAG should be wrapped in a MCP server tool.

Here is an example implementation:

```python

async def call_rag(query: str):
    cur_retries = 0
    url = rag_url
    if not url:
        raise ValueError("RAG_URL environment variable is not set!")

    while True:
        try:
            async with httpx.AsyncClient(timeout=rag_query_timeout) as client:
                response = await client.post(
                    url,
                    json=ChatCompletionRequest(
                        model=llm.model_name,
                        messages=[Message(role="user", content=query)],
                    ).model_dump(exclude_unset=True),
                )

            ctx_resp = ContextResponse(**response.json())
            return ctx_resp.context.content

        except Exception as e:
            print(f"Unknown error {e}")
            cur_retries += 1
            if cur_retries > max_retries:
                return "RAG request error. Do not retry the tool"

tool_rag = StructuredTool.from_function(
    func=None,
    coroutine=call_rag,
    name="doc_query_tool",
    description="This tool allows to query TON documentation for additional info",
    args_schema=RagInput,
)
tools = [tool_rag]
```

For more details check the [client](/client)

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

### Local env instalation

In general it is enough to setup the venv and run:

```shell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu -e .

```

#### Dev

For the development environment setup additional dependencies required for the unit tests to run:

```shell

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu -e ".[test]"
``
```

#### Setup rag index

Then run `rag-setup <path to docs.jsonl>` to build the index.

### Docker build

[Dockerfile](/docker/Dockerfile) is available.

#### Build the docker

``` shell
docker build -f docker/Dockerfile . -t rag_img
```

#### Run the image

``` shell

sudo docker run --name rag_server --network rag_network -v <Path to your .env file>:/app/.env:ro rag_img

```

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

Endpoints top level handlers are implemented at [routes](/api/routes)

Both endpoints support *OpenAI* [ChatCompletion](https://platform.openai.com/docs/api-reference/chat) as a request format for maximum compatibility.

### Context

Main endpoint for context retrieval.
Mapped to `/context` url.

This endpoint respects the `max_tokens` parameter of the completion request.
It would limit max token output including the rendering overhead.

Takes completion request and returns:

``` python
class ContextResponse(BaseModel):
    # Last user message with context augmented
    context: Message

    # Enabled by settings.ADD_RAW_CONTEXT
    # Raw documents in the internal representation.
    # Note that internal representation doesn't continue
    # code snippets in text, but only their reference ids
    raw_context: Optional[List[Dict]]
    # If system message isn't present, default one is proposed
    system: Optional[Message] = None
    #  User extracted intent and topics, if performed
    intent: Optional[IntentInfo] = None

```

### Chat completion

OpenAI compatible chat completion endpoint.

Mapped to `/v1/chat/completions`

It serves an chat completion proxy with context augmentation on the fly.
So, it can be used plug'n play with any ChatCompletion compatible tool.

All the optional fields are passed as is to the LLM endpoint.

[Streming](https://platform.openai.com/docs/api-reference/responses-streaming) **is supported**.

### Models

Some tools require the model name, to interact with ChatCompletion endpoint.
Model endpoint is also compatible with the [OpenAI Models](https://platform.openai.com/docs/api-reference/models/list).

Currently just outputs the model name from config.

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
    %% Input Layer
    Input([User Query + Messages]) --> Validation{Input Validation}
    Validation -- Invalid --> Error[400 Bad Request]
    Validation -- Valid --> Processor[Query Processor]

    %% Query Processing
    subgraph Query_Analysis [Query Processing & Labeling]
        Processor --> CodeDetect{Code/Token Detection}
        Processor --> KeywordMatch{Complex Indicator Check}
        CodeDetect --> Complexity{Is Complex?}
        KeywordMatch --> Complexity
    end

    %% Initial Retrieval
    Complexity --> InitialRetrieval[Initial Retrieval: Main Index + Reranker]

    subgraph Shortcut_Logic [Simple Query Optimization]
        InitialRetrieval --> ScoreCheck{Simple & Score > Threshold?}
        ScoreCheck -- Yes --> Rendering
    end

    %% Complex Path
    ScoreCheck -- No / Is Complex --> IntentExtraction[LLM Intent & Concept Extraction]

    subgraph Multi_Path_Retrieval [Parallel Retrieval Buckets]
        IntentExtraction --> HeaderSearch[Headers Index Search]
        IntentExtraction --> ConceptSearch[Multi-Concept Search]
        IntentExtraction --> DeepSearch[Full Text Search]
    end

    %% Tree Traversal
    HeaderSearch & ConceptSearch & DeepSearch --> TreeTraversal[Knowledge Tree Traversal]

    subgraph Hierarchy_Logic [Structural Expansion]
        TreeTraversal --> DirectChildren[Add Direct Children]
        TreeTraversal --> References[Add Hyperlink Refs]
        DirectChildren --> Scorer[Apply CHILDREN_PENALTY]
        References --> Scorer
    end

    %% Final Ranking & Output
    Scorer --> FinalRerank[Final Reranking & Thresholding]

    subgraph Formatting [Final Context Assembly]
        FinalRerank --> Buckets[Categorize into Buckets]
        Buckets --> RR[Round-Robin Selection]
        RR --> SnippetInlining[Inline Code Snippets from Cache]
    end

    SnippetInlining --> Rendering([Final Context Output])

    %% Styling
    style IntentExtraction fill:#f9f,stroke:#333
    style InitialRetrieval fill:#bbf,stroke:#333
    style Shortcut_Logic stroke-dasharray: 5 5
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
