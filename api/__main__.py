"""`python -m api` starts the HTTP service."""

import uvicorn

from api.app import create_app
from api.settings import load_settings

if __name__ == "__main__":
  settings = load_settings()
  uvicorn.run(create_app(settings=settings), host=settings.host, port=settings.port)
