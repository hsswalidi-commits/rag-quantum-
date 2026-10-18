FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY knowledge_base ./knowledge_base
COPY app.py storage.py chat_app.py ./

EXPOSE 10000

CMD ["streamlit", "run", "chat_app.py", "--server.port=10000", "--server.address=0.0.0.0"]
