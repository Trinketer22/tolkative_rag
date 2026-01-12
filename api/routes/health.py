from fastapi import APIRouter

# Health check
# TODO statistic reports here and such
router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "openai-proxy"}
