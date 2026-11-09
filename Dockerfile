FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs

EXPOSE 8000
CMD ["uvicorn", "movie_finder.api.main:app", "--host", "0.0.0.0", "--port", "8000"]