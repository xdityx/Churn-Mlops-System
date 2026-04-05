FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY features/ features/
COPY models/ models/
COPY monitoring/ monitoring/
COPY tests/ tests/
COPY data/ data/

CMD ["python", "-m", "pytest", "tests/", "-v"]
