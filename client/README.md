# Agentic Client Example

This example demonstrates how simple it is to integrate the RAG API into an agentic workflow.
While currently implemented as a command-line utility, the code is designed to be
easily extensible to other interfaces, such as a Telegram bot.

## Configuration

The agent is configured using two environment variables.

### API Key

`OPENAI_API_KEY` is required for the client to operate.
If you are testing against a local endpoint (or any other provider) that does not
require authentication, set this to any arbitrary string.

### Custom Endpoint

`OPENAI_API_BASE` should point to an OpenAI-compatible endpoint.
If this is not provided, the default OpenAI endpoint will be used.
To use other API standards, minor code modifications are required.

## Installation

To install dependencies, run:
`pip install -r client/requirements.txt`

## Implementation Details

The client is implemented in Python using the [LangGraph SDK](https://docs.langchain.com/oss/python/langgraph/overview).

### Project Structure

- `cli_client.py`: Command-line UI implementation
- `agent.py`: Agent graph implementation and tool definition
- `state.py`: Agent state definition
