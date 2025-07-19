FROM python:3.11-slim

WORKDIR /app

# 1) Копируем бандл целиком
COPY model/bundle ./model/bundle

# 2) Копируем код и зависимости
COPY src/serving/app.py ./app.py
COPY requirements.txt   ./requirements.txt

# 3) Устанавливаем зависимости, включая scikit-learn
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi uvicorn redis tensorflow==2.18.0 joblib numpy scikit-learn

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
