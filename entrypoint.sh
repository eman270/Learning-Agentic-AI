#!/bin/sh
set -eu

uvicorn main:app --host 127.0.0.1 --port 8001 &
streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false &

exec nginx -g 'daemon off;'
