FROM python:3.10-slim

RUN pip install uv


WORKDIR /app


COPY pyproject.toml uv.lock ./
COPY src ./src
COPY knowledge_base ./knowledge_base
COPY app.py storage.py  ./


RUN uv sync


EXPOSE 8080


CMD ["uv", "run", "app.py"]
