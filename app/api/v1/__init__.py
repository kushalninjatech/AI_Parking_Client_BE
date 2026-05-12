from fastapi import APIRouter

from app.api.v1.routes import camera, parking_slot

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(camera.router)
api_v1_router.include_router(parking_slot.router)
