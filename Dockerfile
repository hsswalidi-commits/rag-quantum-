FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY knowledge_base ./knowledge_base
COPY app.py storage.py chat_app.py ./

RUN pip install uv
RUN uv sync

EXPOSE 10000

CMD ["streamlit", "run", "chat_app.py", "--server.port=10000", "--server.address=0.0.0.0"]
