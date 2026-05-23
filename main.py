"""Application entry point."""
import uvicorn
from dotenv import load_dotenv
load_dotenv()


from app.config.settings import get_settings
from app.interfaces.api.app import create_app
settings = get_settings()
app = create_app(settings)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_ENV == "development",
        log_level="info",
    )