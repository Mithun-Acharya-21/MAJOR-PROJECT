from fastapi import APIRouter

from app.api.routes.system import router as system_router
from app.api.routes.analysis import router as analysis_router

api_router = APIRouter()
api_router.include_router(system_router, tags=["system"])
api_router.include_router(analysis_router)

