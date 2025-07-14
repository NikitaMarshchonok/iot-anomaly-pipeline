FROM ubuntu:latest
LABEL authors="nikitamarshchonok"

ENTRYPOINT ["top", "-b"]

FROM python:3.11-slim

WORKDIR /app

# 1) Copy model and metadata
COPY model/saved_model ./model/saved_model
COPY model/scaler.pkl ./model/scaler.pkl
COPY model/meta.json   ./model/meta.json

# 2) Copy the service code and requirements
COPY src/serving/app.py     ./app.py
COPY requirements.txt       ./requirements.txt

# 3) Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
                   fastapi uvicorn redis tensorflow==2.18.0 joblib numpy

# 4) Open port and set entry point
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
