from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from config import Settings, settings
from utils.async_helpers import shutdown_thread_pool

from services.vector_store import vector_store
from services.embedding import embedding_service
from services.reranking import reranker
from services.snippet_cache import snippet_cache

from api.routes import context, completion_proxy, models as llm_models, health, admin
from api.logware import log_requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models_and_index()

    yield

    await shutdown_rag()


def create_app(config: Settings) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.middleware("http")(log_requests)

    app.include_router(context.router, tags=["context"])
    app.include_router(completion_proxy.router, tags=["proxy"])
    app.include_router(llm_models.router, tags=["models"])
    app.include_router(health.router, tags=["health"])

    if config.ENABLE_ADMIN:
        app.include_router(admin.router, prefix=config.ADMIN_PREFIX)
    return app


app = create_app(settings)
log = logging.getLogger("rag_backend")
log.setLevel(settings.LOG_LEVEL)


def setup_logging() -> None:
    from pathlib import Path

    log_path = Path(settings.LOG_FILE)
    log_dir = log_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(settings.LOG_FILE),
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


async def load_models_and_index() -> None:
    """
    Load the models and instantiate
    the services.
    """
    setup_logging()
    embedding_service.initialize()

    # Documents text and headers index
    vector_store.initialize()

    snippet_cache.initialize()
    reranker.initialize()
    log.info("Startup finished – model & index ready.")


async def shutdown_rag():
    shutdown_thread_pool()
    await vector_store.cleanup()
    await snippet_cache.cleanup()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
