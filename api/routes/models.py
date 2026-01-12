from models.domain import Model
from models.response import ModelsResponse
from fastapi import APIRouter
from config import settings

router = APIRouter()


@router.get("/v1/models")
async def list_models():
    return ModelsResponse(
        data=[
            Model(id=settings.PROXY_MODEL_NAME, owned_by=settings.PROXY_MODEL_OWNED_BY),
        ],
    )
