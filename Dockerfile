FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev && rm -rf /var/lib/apt/lists/*

# Сначала обновим pip
RUN pip install --upgrade pip

# Используем чистый requirements
COPY requirements_clean.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]