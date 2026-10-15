FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py app.py ./
COPY ui ./ui

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & exec streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true"]
