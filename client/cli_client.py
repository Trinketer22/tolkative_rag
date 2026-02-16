import os
from typing import List
from uuid import uuid4
import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agent import create_app_workflow
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
default_prompt = """
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

workflow = create_app_workflow(
    "claude-sonnet-4.5",
    rag_url="http://localhost:8000/context",
    model_endpoint=os.environ["OPENAI_API_BASE"],
    system_prompt=default_prompt,
)
app = workflow.compile(checkpointer=memory)


async def run_app():
    thread_id = str(uuid4)
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 30,
    }

    while True:
        user_input = input(">:").strip()
        cmd_input = user_input.lower()
        if not cmd_input:
            continue
        if cmd_input in ["q", "quit", "exit"]:
            print("Goodbye!")
            break

        if cmd_input == "help":
            print(
                """
                Just type the query and hit enter.
                In order to quit, type (q, quit or exit)
                """
            )
            continue

        async for event in app.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        ):
            for k, v in event.items():
                if k == "agent":
                    last_msg = v["messages"][-1]
                    for tool_call in last_msg.tool_calls:
                        name = tool_call["name"]
                        args = tool_call["args"]
                        print(f"LOG: Agent requested tool {name} with Args: {args}")

                    if not last_msg.tool_calls:
                        print(f"Fianl response: {last_msg.content}")


if __name__ == "__main__":
    asyncio.run(run_app())
