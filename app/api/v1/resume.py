"""
Resume download API route.
Serves the portfolio resume from the controlled knowledge directory.
"""
from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.exceptions import NotFoundException

router = APIRouter()


@router.get("/resume")
def get_resume() -> FileResponse:
    """
    Returns the portfolio resume PDF for download or inline display.
    """
    resume_path = settings.resume_file_path
    if not resume_path.is_file():
        raise NotFoundException("Resume file is not available.")

    return FileResponse(
        path=resume_path,
        media_type="application/pdf",
        filename=resume_path.name,
    )
