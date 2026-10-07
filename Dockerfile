FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

WORKDIR /code
COPY pyproject.toml README.md uv.lock ./
COPY app ./app
COPY config ./config
COPY data/lending-loan ./data/lending-loan

# Long timeout for slow networks; the cache keeps finished downloads between builds.
ENV UV_HTTP_TIMEOUT=300
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ARG AGENT_VERSION=0.0.0
ENV AGENT_VERSION=${AGENT_VERSION}

EXPOSE 8080
CMD ["uv", "run", "uvicorn", "app.fast_api_app:app", "--host", "0.0.0.0", "--port", "8080"]
