from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from src.api.routes import hello
from src.api.routes import data

router = APIRouter()

@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

router.include_router(hello.router, tags=["Hello"])
router.include_router(data.router, prefix="/data", tags=["Dataset"])