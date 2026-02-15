import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from models.domain import Message
from models.request import ChatCompletionRequest
from models.response import ContextResponse
from state import AgentState


def create_app_workflow(
    model_name: str,
    rag_url: str,
    max_retries: int = 3,
    rag_query_timeout: float = 300.0,
    model_endpoint: str = "",
    system_prompt: str = "",
):
    llm = (
        ChatOpenAI(
            model=model_name,
            base_url=model_endpoint,
            max_retries=max_retries,
            verbose=True,
        )
        if model_endpoint
        else ChatOpenAI(model=model_name, max_retries=max_retries)
    )

    class RagInput(BaseModel):
        query: str = Field(description="The input query")

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
    # Bind tools to the LLM for this specific instance

    llm_with_tools = llm.bind_tools(tools)

    # --- Node Logic (Inner Functions) ---

    def agent_node(state: AgentState):
        messages = state["messages"]

        # Logic: If a system prompt exists, prepend it to the history
        # (In production, you might manage this slightly differently to avoid
        # duplicate system messages, but this is the simplest pattern).
        if system_prompt:
            messages = [SystemMessage(content=system_prompt)] + messages

        # Call the LLM
        response = llm_with_tools.invoke(messages)

        # Return the response (Diff) to be added to state
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]

        # If the LLM requests a tool call, route to "tools"
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        # Otherwise, end
        return END

    # --- Graph Construction ---

    # Initialize Graph with the AgentState schema
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    # Add Edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    # Compile and Return
    return workflow
