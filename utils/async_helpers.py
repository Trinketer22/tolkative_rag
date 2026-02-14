import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, TypeVar

# Global thread pool
rag_thread_pool: Optional[ThreadPoolExecutor] = None
_T = TypeVar("_T")


async def execute_async(operation: Callable[..., _T]) -> _T:
    thread_pool = get_thread_pool()
    event_loop = asyncio.get_event_loop()
    return await event_loop.run_in_executor(thread_pool, operation)


def get_thread_pool():
    global rag_thread_pool
    if rag_thread_pool is None:
        rag_thread_pool = ThreadPoolExecutor(
            max_workers=max(1, len(os.sched_getaffinity(0)) - 2),
            thread_name_prefix="rag_worker",
        )
    return rag_thread_pool


def shutdown_thread_pool():
    global rag_thread_pool
    if rag_thread_pool:
        rag_thread_pool.shutdown(wait=True)
        rag_thread_pool = None
